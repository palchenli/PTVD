import pandas as pd
import os
src_folder = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/'
out_folder = '/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/'

train_p = os.path.join(src_folder, 'train.csv')
df = pd.read_csv(train_p)
df.insert(2, 'type', value=['annotations']*len(df))
df.to_csv(os.path.join(out_folder, 'pretrain_train.csv'), index=False)


train_p = os.path.join(src_folder, 'val.csv')
df = pd.read_csv(train_p)
df.insert(2, 'type', value=['annotations']*len(df))
df.to_csv(os.path.join(out_folder, 'pretrain_val.csv'), index=False)