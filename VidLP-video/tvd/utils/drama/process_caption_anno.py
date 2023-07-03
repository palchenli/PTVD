import pandas as pd
import json
import os

def generate_gt_anno_coco_style(meta_path, anno_folder, out_path):
    annotations = []
    images = []
    d = pd.read_csv(meta_path)
    idx = 0
    for i in range(len(d)):
        sample = d.iloc[i]
        vid = sample['Name'].split('/')[-1].split('.')[0]
        seg_id = int(sample['segment_id'])
        sample_id = vid + '_' + str(seg_id)
        caption_p = os.path.join(anno_folder, vid + '.json')
        
        info = json.load(open(caption_p))
        caption = info['sentences'][seg_id]
        images_info = {'id': sample_id}
        caption = ' '.join(caption)
        info = {'image_id': sample_id, 'id': sample_id, 'caption': caption}
        annotations.append(info)
        images.append(images_info)

    out = {'info': None, 'images':images, 'licenses':None, 'type':None, 'annotations':annotations}
    json.dump(out, open(out_path, 'w'), ensure_ascii=False)

# test_p= '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/test.csv'
# test_out_p = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/test_coco_style_spaced.json'

# train_p= '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/train.csv'
# train_out_p = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/train_coco_style_spaced.json'

# anno_folder = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/videos/subset_0/annotations-biaozhu25001/'

test_p= '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/pretrain_test.csv'
test_out_p = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/pretrain_test_coco_style_spaced.json'

anno_folder = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/videos/subset_0/annotations'
# generate_gt_anno_coco_style(train_p, anno_folder, train_out_p)
generate_gt_anno_coco_style(test_p, anno_folder, test_out_p)
