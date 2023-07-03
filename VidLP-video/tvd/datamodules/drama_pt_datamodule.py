from tvd.datasets import DramaDatasetPT
from .video_datamodule_base import BaseDataModule


class DramaPTMDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DramaDatasetPT

    @property
    def dataset_cls_no_false(self):
        return DramaDatasetPT

    @property
    def dataset_name(self):
        return "dramaPT"
