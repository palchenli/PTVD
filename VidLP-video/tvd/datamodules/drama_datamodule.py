from tvd.datasets import DramaDataset
from .video_datamodule_base import BaseDataModule


class DramaMDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DramaDataset

    @property
    def dataset_cls_no_false(self):
        return DramaDataset

    @property
    def dataset_name(self):
        return "drama"
