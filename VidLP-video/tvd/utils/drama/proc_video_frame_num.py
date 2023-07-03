import os
import json
from tqdm import tqdm

vid_dir = "/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/images"
files = os.listdir(vid_dir)
out = {}
for file in tqdm(files):
    file_path = os.path.join(vid_dir, file)
    num = len(os.listdir(file_path))
    out[file] = num
json.dump(
    out,
    open(
        "/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/frame_num.json",
        "w",
    ),
)
