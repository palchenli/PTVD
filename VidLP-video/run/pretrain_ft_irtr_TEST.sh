#!/usr/bin/env bash
PROJECT_ROOT=/group/30042/wybertwang/project/VidLP_subtitle
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
per_gpu_batchsize=4
config_name=ft_irtr_drama
exp_name=drama_pt_ft_irtr_TEST
debug=False
# load_path="result/drama_pretraining_v0_bs4096_lr5e5_step3k_ft_irtr_false1_bs4_NewData_seed0_from_epoch_010-step_002719-val_score_1.43/version_0/checkpoints/epoch_006-step_000265-val_score_0.87.ckpt"
# load_path="result/drama_pt_ft_irtr_TEST_seed0_from_epoch_010-step_002719-val_score_1.43/version_0/checkpoints/epoch_006-step_000265-val_score_0.87.ckpt"
load_path="result/drama_pt_ft_irtr_seed0_from_epoch_013-step_001794-val_score_1.45/version_4/checkpoints/epoch_009-step_000388-val_score_0.89.ckpt"
# load_path="/group/30042/wybertwang/project/VidLP_New/result/drama_ft_irtr_false1_bs4_seed0_from_epoch_010-step_001332-val_score_1.36/version_0/checkpoints/epoch_001-step_002063-val_score_0.81.ckpt" # model weights after pretraining + irtr fintuning

###################### generate a config json file ##########################
config_path=`python generate_configs.py with data_root=${DATA_ROOT} num_gpus=$_num_gpus num_nodes=$_num_nodes $config_name \
    per_gpu_batchsize=$per_gpu_batchsize val_check_interval=0.25 exp_name=$exp_name debug=$debug load_path=$load_path get_recall_metric=True draw_false_text=1 test_only=True skip_test_step=True `
echo "config_path: $config_path"

log_file=logs/${exp_name}_${time_start}.log

###################### Start Training ##########################
# python  -m torch.distributed.launch $@ run_dist.py --cfg $config_path |& tee -a  ${log_file}_${NODE_RANK}
python  -m torch.distributed.launch --master_port 23456 --nproc_per_node=$_num_gpus run_dist.py --cfg $config_path |& tee -a  ${log_file}_${NODE_RANK}
# NNODES=1 python  -m torch.distributed.launch --nproc_per_node=$_num_gpus run_dist.py --cfg configs/debug_drama.json
