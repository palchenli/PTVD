from .video_base_dataset import BaseDataset, read_frames_from_img_dir
import pandas as pd
import os
import numpy as np
import random
import json


class DramaDataset(BaseDataset):
    """Drama Video-Text loader."""

    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["drama_train"]
        elif split == "val":
            names = ["drama_val"]
        elif split == "test":
            names = ["drama_test"]
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

        self.metadata = None
        self._load_metadata()
        self.min_time = 4.0
        self.size = 224
        self.fps = 2
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = True
        if self.split == "train":
            self.center_crop = False
        else:
            self.center_crop = True
        self.benchmark = False
        self.num_candidates = 1
        self.random_flip = True
        self._load_metadata()

    def _load_metadata(self):
        metadata_dir = "/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata"
        split_files = {
            "train": "wt_split/pretrain_train.csv",
            "val": "wt_split/test.csv",
            "test": "wt_split/test.csv",
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep=",")
        self.metadata = metadata
        self.frame_num_info = json.load(open(os.path.join(metadata_dir, "wt_split/frame_num.json")))
        print("dataset size: {}".format(len(metadata)))

    # sample fix video length
    def get_caption(self, caption_csv, segment_id):
        with open(caption_csv, "r") as f:
            cap = json.load(f)
        video_len = cap["duration"]
        start, end = cap["timestamps"][segment_id]
        text = cap["sentences"][segment_id]
        text = self.text_preprocessor(text)
        return text, start, end, video_len

    def get_text(self, sample):
        caption_csv = self.get_caption_path(sample)
        text, start, end, duration = self.get_caption(caption_csv, sample["segment_id"])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {"text": (text, encoding)}, start, end, duration

    def get_caption_path(self, sample):
        name = sample["Name"]
        annotation_type = sample["type"]
        return os.path.join(
            self.data_dir,
            "videos",
            name.split("/")[0],
            annotation_type,
            name.split("/")[-1][:-4] + ".json",
        )

    def get_false_text(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        caption_csv = self.get_caption_path(sample)
        text, start, end, duration = self.get_caption(caption_csv, sample["segment_id"])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {f"false_text_{rep}": (text, encoding)}

    def get_sample_id(self, index):
        sample = self.metadata.iloc[index]
        return sample["Name"].split("/")[-1].split(".")[0] + "_" + str(sample["segment_id"])

    def _get_video_path(self, sample):
        rel_video_fp = sample["Name"].split("/")[-1].split(".")[0]
        full_video_fp = os.path.join(self.data_dir, "images", rel_video_fp)
        return full_video_fp, rel_video_fp

    def get_raw_video(self, sample, begin, end, duration):
        abs_fp, rel_fp = self._get_video_path(sample)
        vlen = self.frame_num_info[rel_fp]
        seg_len = round(vlen * (end - begin) / duration)
        begin = round(vlen * begin / duration)
        end = round(vlen * end / duration)
        imgs, idxs, idxs_num = read_frames_from_img_dir(
            abs_fp,
            self.num_frames,
            mode=self.split,
            prefix=rel_fp + "-",
            fix_begin=begin,
            fix_len=seg_len,
        )

        if imgs.size(0) != self.num_frames:
            raise Exception("video length not enough!", rel_fp)
        if imgs is None:
            raise Exception("Invalid img!", rel_fp)
        else:
            return imgs

    def get_video(self, sample, start, end, duration):
        imgs = self.get_raw_video(sample, start, end, duration).permute(1, 0, 2, 3)  # to cthw
        imgs_tensor = [self.video_transform(imgs).permute(1, 0, 2, 3)]  # to tchw
        subtitle = ". ".join(self.get_subtitle(sample, start, end)).replace("  ", " ")

        subtitle_encoding = self.tokenizer(
            subtitle,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return imgs_tensor, {"subtitle_text": (subtitle, subtitle_encoding)}

    def get_false_video(self, rep):
        random_index = random.randint(0, len(self.metadata) - 1)
        sample = self.metadata.iloc[random_index]
        caption_csv = self.get_caption_path(sample)
        _, start, end, duration = self.get_caption(caption_csv, sample["segment_id"])
        imgs = self.get_raw_video(sample, start, end, duration).permute(1, 0, 2, 3)  # to cthw
        # can be different augmentation
        imgs_tensor = [self.video_transform(imgs).permute(1, 0, 2, 3)]  # to tchw
        subtitle = ". ".join(self.get_subtitle(sample, start, end)).replace("  ", " ")

        subtitle_encoding = self.tokenizer(
            subtitle,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return {
            f"false_image_{rep}": imgs_tensor,
            f"false_subtitle_text_{rep}": (subtitle, subtitle_encoding),
        }

    def get_subtitle(self, sample, start, end):
        rel_video_fp = sample["Name"].split("/")[-1].split(".")[0]
        video_sub_path = os.path.join(self.data_dir, "subtitles", rel_video_fp + ".json")
        if not os.path.exists(video_sub_path):
            return []
        vid_subtitle = json.load(open(video_sub_path))
        if len(vid_subtitle["timestamps"]) == 0:
            print(f"skipping video {video_sub_path} which has 0 timetstamps")
            return []
        timestamps = np.array(vid_subtitle["timestamps"])
        hit_flag = (
            ((timestamps[:, 0] - start < 0) & (timestamps[:, 1] - start > 0))
            | ((timestamps[:, 0] - end < 0) & (timestamps[:, 1] - end > 0))
            | ((timestamps[:, 0] - start > 0) & (timestamps[:, 1] - end < 0))
        )
        indics = np.nonzero(hit_flag)[0]
        subs = [vid_subtitle["sentences"][idx] for idx in indics]
        return subs

    def get_suite(self, index):
        result = None
        max_try = 5
        try_time = 0
        while result is None:
            try_time += 1
            sample = self.metadata.iloc[index]
            ret = dict()
            text, start, end, duration = self.get_text(sample)
            ret.update(text)
            imgs_tensor, subtitle = self.get_video(sample, start, end, duration)
            ret.update(subtitle)
            ret.update(
                {
                    "image": imgs_tensor,
                    "img_index": index,
                    "cap_index": index,
                    "raw_index": index,
                }
            )
            ret.update({"replica": True if ret["cap_index"] > 0 else False})
            for i in range(self.draw_false_image):
                ret.update(self.get_false_video(i))
            for i in range(self.draw_false_text):
                ret.update(self.get_false_text(i))
            result = True
            if try_time > max_try:
                print(f"Exceed max time Error while read file idx {sample} in {self.names[0]}")
        return ret

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.get_suite(index)
