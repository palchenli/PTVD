from pycocotools.coco import COCO
# from meter.pycocoevalcap.eval import COCOEvalCap
from tvd.modules.objectives import evaluate

ann_path='/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/test_coco_style.json'
cache_path='/group/30042/wybertwang/project/VidLP_New/result/drama_pretraining_v0_bs4096_lr5e5_step3k_ft_cap_seed0_from_epoch_010-step_002719-val_score_1.43/caption.json'
evaluate(cache_path, ann_path, False)

