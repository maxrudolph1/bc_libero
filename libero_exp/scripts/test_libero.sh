#!/bin/bash

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0      # if libgpu_partition.so confilts with gym and robosuite
# export MUJOCO_GL=osmesa
# export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

source .venv/bin/activate

ENV_NAME=libero_spatial             # ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
POLICY_NAME=bc_policy          # ['bc_policy', 'bc_ib_policy']
CONFIG_NAME=transformer          # backbone_name: ['mlp', 'rnn', 'transformer', 'vilt']
TRAIN_RATIO=0.9          
SEED=0

MI=1e-3
MINE=0.1

# bash libero_exp/scripts/main_libero.sh 'libero_spatial' 'bc_policy' 'transformer' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_object' 'bc_policy' 'vilt' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_goal' 'bc_policy' 'rnn' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_10' 'bc_policy' 'mlp' 0.9 0

# bash libero_exp/scripts/main_libero.sh 'libero_spatial' 'bc_ib_policy' 'transformer' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_object' 'bc_ib_policy' 'vilt' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_goal' 'bc_ib_policy' 'rnn' 0.9 0
# bash libero_exp/scripts/main_libero.sh 'libero_10' 'bc_ib_policy' 'mlp' 0.9 0


python train_libero.py \
    --config-path=libero_exp/configs/${POLICY_NAME} \
    --config-name=${CONFIG_NAME} \
    data.env_name=${ENV_NAME} \
    train.seed=${SEED} \
    data.train_ratio=${TRAIN_RATIO} \
    train.mine_mi_loss_scale=${MINE} \
    train.mi_loss_scale=${MI} \
    train.train_gpus=[0,1]
