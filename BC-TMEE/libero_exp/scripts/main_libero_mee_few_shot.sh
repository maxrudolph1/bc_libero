#!/bin/cd /media/code/BC-IB && bash

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0      # if libgpu_partition.so confilts with gym and robosuite
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
# export MUJOCO_GL=egl
# export PYOPENGL_PLATFORM=egl

source /opt/conda/etc/profile.d/conda.sh
conda activate bcib

ENV_NAME=$1             # ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
POLICY_NAME=$2          # ['bc_policy', 'bc_mee_policy']
CONFIG_NAME=$3          # backbone_name: ['dp', 'rnn', 'transformer']
TRAIN_RATIO=$4         
SEED=$5

MI=1e-3
MINE=0.1

#bash libero_exp/scripts/main_libero_mee_few_shot.sh 'libero_spatial' 'bc_mee_policy' 'transformer' 0.2 0

python train_libero.py \
    --config-path=libero_exp/configs/${POLICY_NAME} \
    --config-name=${CONFIG_NAME} \
    data.env_name=${ENV_NAME} \
    train.seed=${SEED} \
    data.train_ratio=${TRAIN_RATIO} \
    train.mine_mi_loss_scale=${MINE} \
    train.mi_loss_scale=${MI}
