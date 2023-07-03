from tvd.datasets import DramaDatasetFT
from .video_datamodule_base import BaseDataModule


class DramaFTMDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DramaDatasetFT

    @property
    def dataset_cls_no_false(self):
        return DramaDatasetFT

    @property
    def dataset_name(self):
        return "dramaFT"
