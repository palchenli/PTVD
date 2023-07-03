import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)


class text_preprocessor:
    def __init__(self, config) -> None:
        self.prepend_bos = config["add_new_bos_token"] and config["prepend_bos_token"]
        self.append_eos = config["add_new_bos_token"] and config["append_eos_token"]

    def __call__(self, text):
        text = text.rstrip().rstrip(".").rstrip() + "."
        if self.prepend_bos:
            text = "<bos>" + " " + text
        if self.append_eos:
            text = text + " " + "<eos>"
        return text


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(from_pretrained, do_lower_case="uncased" in from_pretrained)
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(from_pretrained, do_lower_case="uncased" in from_pretrained)


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()
        self.data_dir = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]
        self.num_frames = _config["num_frames"]
        self.draw_options_text = _config["draw_options_text"]
        self.backend = _config["backend"]
        self.train_transform_keys = (
            ["default_train"] if len(_config["train_transform_keys"]) == 0 else _config["train_transform_keys"]
        )
        self.val_transform_keys = (
            ["default_val"] if len(_config["val_transform_keys"]) == 0 else _config["val_transform_keys"]
        )
        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        if _config["add_new_bos_token"]:
            self.tokenizer.add_tokens(["<bos>", "<eos>"])

        self.vocab_size = self.tokenizer.vocab_size

        collator = DataCollatorForWholeWordMask if _config["whole_word_masking"] else DataCollatorForLanguageModeling

        self.mlm_collator = collator(tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"])
        self.setup_flag = False
        self.text_preprocessor = text_preprocessor(_config)

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            num_frames=self.num_frames,
            draw_options_text=self.draw_options_text,
            backend=self.backend,
            text_preprocessor=self.text_preprocessor,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            num_frames=self.num_frames,
            draw_options_text=self.draw_options_text,
            backend=self.backend,
            text_preprocessor=self.text_preprocessor,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
                num_frames=self.num_frames,
                draw_options_text=self.draw_options_text,
                backend=self.backend,
                text_preprocessor=self.text_preprocessor,
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
            num_frames=self.num_frames,
            draw_options_text=self.draw_options_text,
            backend=self.backend,
            text_preprocessor=self.text_preprocessor,
        )

    def make_no_false_test_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
            num_frames=self.num_frames,
            draw_options_text=self.draw_options_text,
            backend=self.backend,
            text_preprocessor=self.text_preprocessor,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            num_frames=self.num_frames,
            draw_options_text=self.draw_options_text,
            backend=self.backend,
            text_preprocessor=self.text_preprocessor,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader