import random
import torch
import io
import os
import cv2
import numpy as np
from PIL import Image
from tvd.transforms import keys_to_transforms
import decord
from decord import cpu, gpu
import imageio

import ftfy
import regex as re
import demoji
import editdistance
import tslearn.metrics
import string
from tvd.transforms.videoaug import VideoTransform


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        num_frames=1,
        draw_options_text=0,
        backend="v100",
        text_preprocessor=None,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir
        self.text_preprocessor = text_preprocessor

        if len(names) != 0:
            dataset_name = names[0].split("_")[0]
            if dataset_name in ["tgif", "tgifqa"]:
                dataset_name = "tgif"
            self.data_dir = os.path.join(self.data_dir, dataset_name)  # e.g. webvid_train -> webvid
            split_name = dataset_name
        if torch.distributed.get_rank() == 0:
            print("*" * 100)
            print("video datasets: {}".format(names))
        self.draw_options_text = draw_options_text
        self.num_frames = num_frames
        if torch.distributed.get_rank() == 0:
            print("# frames for base dataset is: {}".format(self.num_frames))

        self.video_transform = VideoTransform(mode=self.split, crop_size=image_size, backend=backend)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def _get_video_path(self, sample):
        if self.names[0] in ["msrvtt_train", "msrvtt_test", "msrvtt_val"]:
            return (
                os.path.join(self.data_dir, "videos", "all", sample.name + ".mp4"),
                sample.name + ".mp4",
            )
        else:
            return (
                os.path.join(self.data_dir, "videos", "all", str(sample["video_id"]) + ".mp4"),
                str(sample["video_id"]) + ".mp4",
            )

    def get_raw_video(self, sample):
        abs_fp, rel_fp = self._get_video_path(sample)
        imgs, idxs, vlen = read_frames_decord(abs_fp, self.num_frames, mode=self.split)
        if imgs is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return imgs

    def get_video(self, sample):
        imgs = self.get_raw_video(sample).permute(1, 0, 2, 3)  # to cthw
        imgs_tensor = [self.video_transform(imgs).permute(1, 0, 2, 3)]  # to tchw
        return imgs_tensor

    def get_false_video(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        imgs = self.get_raw_video(sample).permute(1, 0, 2, 3)  # to cthw
        assert imgs.size(1) == self.num_frames
        imgs_tensor = [self.video_transform(imgs).permute(1, 0, 2, 3)]  # to tchw
        return {f"false_image_{rep}": imgs_tensor}

    def _get_caption(self, sample):
        if self.names[0] in ["msrvtt_train"]:
            caption = random.choice(sample["captions"])
        else:
            caption = sample["captions"][0]
        return caption

    def get_text(self, raw_index, sample):
        text = self._get_caption(sample)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        # print(encoding.size())
        return {
            "text": (text, encoding),
            "img_index": raw_index,
            "cap_index": raw_index,
            "raw_index": raw_index,
        }

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        text = self._get_caption(sample)
        encoding = self.tokenizer(
            text,
            # padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_suite(self, index):
        result = None
        while result is None:
            # retry_times += 1
            sample = self.metadata.iloc[index]
            # print(sample[1])
            try:
                video_tensor = self.get_video(sample)
                ret = {
                    "image": video_tensor,
                    "img_index": index,
                    "cap_index": index,
                    "raw_index": index,
                }
                if not self.image_only:
                    txt = self.get_text(index, sample)
                    ret.update({"replica": True if txt["cap_index"] > 0 else False})
                    ret.update(txt)

                for i in range(self.draw_false_image):
                    ret.update(self.get_false_video(i))
                for i in range(self.draw_false_text):
                    ret.update(self.get_false_text(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {sample.name} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.metadata) - 1)
        return ret

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            # print(size)
            assert (
                len(size) == 4
            ), f"Collate error, an image should be in shape of (T, 3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[2] for i in img_sizes])
            max_width = max([i[3] for i in img_sizes])

        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [torch.zeros(batch_size, self.num_frames, 3, max_height, max_width) for _ in range(view_size)]
            # print(len(img))
            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        # new_images[vi][bi] = None
                        continue
                    else:
                        orig = img[bi][vi]
                        # print(orig.size())
                        new_images[vi][bi, :, :, : orig.shape[-2], : orig.shape[-1]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        # print(txt_keys)
        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                lm_labels = input_ids[:, 1:]
                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_labels_lm"] = lm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
        return dict_batch


def sample_frames_bak(num_frames, vlen, sample="rand", fix_start=None):
    acc_samples = min(num_frames, vlen)
    # print('acc+num', acc_samples, num_frames, vlen, flush=True)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        assert interv < intervals[idx + 1] - 1, (num_frames, vlen)
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == "rand":
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == "uniform":
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs


def sample_frames(num_frames, vlen, sample="rand", fix_start=None):
    acc_samples = min(num_frames, vlen)
    ranges = []

    if acc_samples < num_frames:
        for i in range(vlen + 1):
            ranges.append([i, i + 1])
    else:
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        for idx, interv in enumerate(intervals[:-1]):
            # assert interv < intervals[idx + 1] - 1, (num_frames, vlen)
            ranges.append((interv, intervals[idx + 1]))

    # print(ranges)
    if sample == "rand":
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == "uniform":
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    # print(
    #     f"range: {ranges}, frame_idxs: {frame_idxs}, num_frames: {num_frames}, vlen: {vlen}"
    # )
    return frame_idxs


def read_frames_gif(video_path, num_frames, mode="train", fix_start=None):
    if mode == "train":
        sample = "rand"
    else:
        sample = "uniform"
    gif = imageio.get_reader(video_path)
    vlen = len(gif)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index, frame in enumerate(gif):
        # for index in frame_idxs:
        if index in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            # frame = Image.fromarray(frame)
            frames.append(frame)
    frames = torch.stack(frames)  # .float() / 255
    # print(frames.size())
    return frames, frame_idxs, vlen


def read_frames_cv2(video_path, num_frames, sample="rand", fix_start=None):
    # print(video_path)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    # for decord
    # cap.set(3, 256)
    # cap.set(4, 256)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            # frame = Image.fromarray(frame)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    # return frames tensor
    # convert cv to PIL
    # img = Image.fromarray(imgs[0])
    frames = torch.stack(frames)  # .float() / 255
    # print(frames.size())
    cap.release()
    return frames, success_idxs, vlen


def read_frames_decord(video_path, num_frames, mode="train", fix_start=None):
    # print("video path: {}".format(video_path))
    if mode in ["train", "val"]:
        sample = "rand"
    else:
        sample = "uniform"
    video_reader = decord.VideoReader(video_path, width=512, height=512, num_threads=1, ctx=cpu(0))
    # video_reader = decord.VideoReader(video_path, width=256, height=256, num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge("torch")
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs).byte()
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, vlen


def read_frames_from_img_dir(
    video_path,
    num_frames,
    mode="train",
    fix_begin=0,
    fix_len=None,
    prefix="",
    suffix=".jpg",
):
    if mode in ["train", "val"]:
        sample = "rand"
    else:
        sample = "uniform"
    vlen = len(os.listdir(video_path)) if fix_len is None else fix_len
    frame_idxs = sample_frames(num_frames, vlen, sample=sample)
    if fix_begin > 0:
        frame_idxs = [idx + fix_begin for idx in frame_idxs]

    if len(frame_idxs) > 0 and len(frame_idxs) < num_frames:
        # print(
        # f"padding frames from {len(frame_idxs)} to {num_frames}, video: {video_path}, frame_idxs: {frame_idxs}, num_frames: {num_frames}, vlen: {vlen}"
        # )
        frame_idxs = frame_idxs + frame_idxs[-1:] * (num_frames - len(frame_idxs))

    frames = []
    for idx in frame_idxs:
        frame = cv2.imread(os.path.join(video_path, prefix + str(idx) + suffix))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frame = torch.from_numpy(frame).byte()
        frame = frame.permute(2, 0, 1)  # chw
        frames.append(frame)
    frames = torch.stack(frames, dim=0)  # tchw
    return frames, frame_idxs, vlen


def sample_frames_2(frame_loc, vlen, frame_end):
    assert frame_loc <= frame_end
    frame_idxs = [frame_loc]
    return frame_idxs


def read_large_frames_decord(video_path, frame_loc, frame_end, num_frames, mode="train", fix_start=None):
    # print('*'*100)
    # print(mode)
    if mode == "train":
        sample = "rand"
    else:
        sample = "uniform"
    # video_reader = decord.VideoReader(video_path, width=256, height=256, num_threads=1, ctx=cpu(0))
    video_reader = decord.VideoReader(video_path, width=512, height=512, num_threads=1, ctx=cpu(0))
    decord.bridge.set_bridge("torch")
    # vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, 120, sample=sample, fix_start=fix_start)
    for i in range(len(frame_idxs)):
        # if random.random() < 0.5:
        #     frame_idxs[i] += frame_loc - 60
        # else:
        #     frame_idxs[i] += frame_loc
        frame_idxs[i] += frame_loc - 60
        frame_idxs[i] = min(frame_idxs[i], frame_end - 1)
        frame_idxs[i] = max(0, frame_idxs[i])
    # print(frame_loc, frame_end, frame_idxs)
    frames = video_reader.get_batch(frame_idxs).byte()
    frames = frames.permute(0, 3, 1, 2)
    return frames


def video_clip_reader(video_path, begin_time, end_time, duration, num_frames):
    cap = cv2.VideoCapture(video_path)
    # print(video_path)
    # assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("vcrr", video_path, begin_time, end_time, duration, num_frames, vlen, flush=True)
    average_fps = vlen / duration
    clip_len = (end_time - begin_time) * average_fps
    frame_idxs = sample_frames(num_frames, int(clip_len), sample="rand")
    frames = []
    success_idxs = []
    rel_index = int(begin_time * average_fps)
    rel_index = max(rel_index, 0)
    rel_index = min(rel_index, vlen - 1)
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rel_index + index)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            # frame = Image.fromarray(frame)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    # print(video_path)
    # print(len(frames))
    frames = torch.stack(frames)
    cap.release()
    if frames.size(0) < num_frames:
        zeros = torch.ones(
            (num_frames - frames.size(1), 3, frames.size(-2), frames.size(-1)),
            dtype=torch.uint8,
        ).byte()
        frames = torch.cat((frames, zeros), axis=1)
    if frames.size(0) != num_frames:
        Exception(RuntimeError)
    return frames


def video_reader(video_path, frame_loc, frame_end, num_frames):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames fps is 30, 4s
    frame_idxs = sample_frames(num_frames, 120, sample="rand")
    # frame_idxs = sample_frames_2(frame_loc, vlen, frame_end)
    frames = []
    success_idxs = []

    for index in frame_idxs:
        if random.random() < 0.5:
            rel_index = index + frame_loc - 120
        else:
            rel_index = index + frame_loc
        rel_index = max(rel_index, 0)
        rel_index = min(rel_index, frame_end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, rel_index)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).byte()
            # # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            # frame = Image.fromarray(frame)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
    frames = torch.stack(frames)
    cap.release()
    return frames


def fast_decode(video_path, num_frames, mode="train", fix_start=None, fps=30):
    cap = cv2.VideoCapture(video_path)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    max_len = vlen / 30
    num_sec = num_frames / float(fps)
    size = 224
    crop_only = True
    random_flip = True
    start_seek = random.randint(0, int(max(max_len, max_len - num_sec)))
    cmd = ffmpeg.input(video_path, ss=start_seek, t=num_sec + 0.1).filter("fps", fps=fps)
    if mode == "train":
        aw, ah = random.uniform(0, 1), random.uniform(0, 1)
    else:
        aw, ah = 0.5, 0.5
    if crop_only:
        cmd = cmd.crop(
            "(iw - {})*{}".format(size, aw),
            "(ih - {})*{}".format(size, ah),
            str(size),
            str(size),
        )
    else:
        cmd = cmd.crop(
            "(iw - min(iw,ih))*{}".format(aw),
            "(ih - min(iw,ih))*{}".format(ah),
            "min(iw,ih)",
            "min(iw,ih)",
        ).filter("scale", size, size)
    if random_flip and random.uniform(0, 1) > 0.5:
        cmd = cmd.hflip()
    out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(capture_stdout=True, quiet=True)
    video = np.frombuffer(out, np.uint8).reshape([-1, size, size, 3])
    video = torch.from_numpy(video)
    video = video.permute(3, 0, 1, 2)
    return video, _, _


# 21
colormaps = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (61, 145, 64),
    (127, 255, 212),
    (0, 201, 87),
    (218, 112, 214),
    (255, 0, 255),
    (112, 128, 105),
    (250, 235, 215),
    (240, 255, 255),
    (252, 230, 201),
    (255, 255, 0),
    (235, 142, 85),
    (255, 97, 0),
    (176, 224, 230),
    (
        65,
        106,
        225,
    ),
    (0, 255, 255),
    (56, 94, 15),
    (8, 46, 84),
    (255, 192, 203),
]


# only_use_relevant_dets ?


def color_img(im, object_meta, relevant_dets, only_use_relevant_dets=True):
    # mask detected region
    # only_use_relevant_dets: if true, we only mask regions that mentioned in question & answers
    # print(relevant_dets)
    if only_use_relevant_dets:
        boxes = []
        for index in relevant_dets:
            boxes.append(object_meta["boxes"][index])
            # print(object_meta['names'][index])
        # object_index = relevant_dets
    else:
        boxes = object_meta["boxes"]
    # print(len(boxes))
    # range(len(boxes))
    for i in range(len(boxes)):
        if i > 20:
            break
        bbox = boxes[i]
        # white_rect = cv2.applyColorMap(white_rect, i)
        # only considering bounding box here (wo fine-grained segmentation)
        sub_img = im[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        white_rect[:, :, 0] = colormaps[i][0]
        white_rect[:, :, 1] = colormaps[i][1]
        white_rect[:, :, 2] = colormaps[i][2]
        res = cv2.addWeighted(sub_img, 0.7, white_rect, 0.3, 1.0)
        im[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])] = res
        cv2.rectangle(
            im,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            colormaps[i],
            3,
        )
    return im


def get_video_len(video_src):
    cap = cv2.VideoCapture(video_src)
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen
