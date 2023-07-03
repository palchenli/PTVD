#!/usr/bin/env bash
# PROJECT_ROOT=/group/30042/wybertwang/project/VidLP_New
PROJECT_ROOT=/group/30042/wybertwang/project/VidLP_subtitle/
DATA_ROOT=/group/30042/palchenli/projects/meter_pretrain/dataset_cn
cd $PROJECT_ROOT

sudo chmod -R 777 .

pip install --upgrade pip
. /data/miniconda3/etc/profile.d/conda.sh
which conda
conda deactivate
conda activate env-3.8.8
PYTHON=${PYTHON:-"/data/miniconda3/envs/env-3.8.8/bin/python"}
nvidia-smi
$PYTHON -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.backends.cudnn.version()); print(torch.cuda.is_available())"
time_start=$(date "+%Y-%m-%d-%H-%M-%S")

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1 
export PYTHONPATH=./
export TORCH_HOME=$PROJECT_ROOT/.cache

conda install -y ffmpeg
pip install -r requirements.txt --user
pip install -e .  --user
pip install sacred  --user
export PYTHONPATH=./


###################### Arguments ##########################
_num_nodes=1
_num_gpus=`nvidia-smi -L |wc -l`
export NNODES=$_num_nodes
per_gpu_batchsize=5
config_name=ft_mlm_cap_dramacap
# exp_name=drama_ft_BSCcap_lr5e5_from_Scratch_e30_seed0_from_TEST_v2
exp_name=drama_ft_BSCcap_TEST
debug=False

load_path="result/wrong_0524/drama_ft_BSCcap_seed0_from_/version_2/checkpoints/epoch_002-step_006601-val_score_0.57.ckpt"
config_path=`python generate_configs.py with data_root=${DATA_ROOT} num_gpus=$_num_gpus num_nodes=$_num_nodes $config_name \
    per_gpu_batchsize=$per_gpu_batchsize val_check_interval=0.25 exp_name=$exp_name debug=$debug load_path=$load_path gt_caption_anno_path=/group/30042/palchenli/projects/meter_pretrain/dataset_cn/drama/metadata/wt_split/pretrain_test_coco_style_spaced_multi_5.json test_only=True skip_test_step=True`
echo "config_path: $config_path"

log_file=logs/${exp_name}_${time_start}.log

###################### Start Training ##########################
# python  -m torch.distributed.launch $@ run_dist.py --cfg $config_path |& tee -a  ${log_file}_${NODE_RANK}
python  -m torch.distributed.launch --master_port 23456 --nproc_per_node=$_num_gpus run_dist.py --cfg $config_path |& tee -a  ${log_file}_${NODE_RANK}
# NNODES=1 python  -m torch.distributed.launch --nproc_per_node=$_num_gpus run_dist.py --cfg configs/debug_drama.json
