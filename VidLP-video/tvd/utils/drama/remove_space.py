import json
import sys

src_path = sys.argv[1]
dst_path = sys.argv[2]
# src_path='/group/30042/wybertwang/project/VidLP_New/result/drama_pretraining_v0_bs4096_lr5e5_step3k_ft_cap_lr5e5_e30_seed0_from_epoch_010-step_002719-val_score_1.43/caption.json_2'

d= json.load(open(src_path))
for img in d:
    cap = img['caption']
    img['caption'] = cap.replace(' ','')
json.dump(d, open(dst_path, 'w+'), ensure_ascii=False)