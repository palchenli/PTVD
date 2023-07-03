from .drama import DramaDataset
import pandas as pd
import os
import json


class DramaDatasetPT(DramaDataset):
    """Drama Video-Text loader."""

    def _load_metadata(self):
        metadata_dir = "/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata"
        split_files = {
            "train": "wt_split/pretrain_train.csv",
            "val": "wt_split/pretrain_test.csv",
            "test": "wt_split/pretrain_test.csv",
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep=",")
        self.metadata = metadata
        self.frame_num_info = json.load(open(os.path.join(metadata_dir, "wt_split/frame_num.json")))
        print("dataset size: {}".format(len(metadata)))
