#!/bin/bash

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0      # if libgpu_partition.so confilts with gym and robosuite
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export CUDA_VISIBLE_DEVICES=0

source activate bcib

POLiCY_NAME=$1
CONFIG_NAME=$2
MODEL_TYPE=$3
SEED=$4

ENV_NAME=adroit
TASK_NAMES=(pen-v0 relocate-v0)

MINE=0.1
MI=5e-3

# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_policy full_ft_temporal_fuse ResNet
# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_policy full_ft_spatial_fuse ResNet
# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_policy partial_ft_temporal_fuse VC1
# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_policy partial_ft_spatial_fuse VC1

# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_ib_policy full_ft_temporal_fuse ResNet
# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_ib_policy full_ft_spatial_fuse ResNet
# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_ib_policy partial_ft_temporal_fuse VC1
# bash cortexbench_exp/scripts/main_cortex_adroit_all.sh bc_ib_policy partial_ft_spatial_fuse VC1

SEEDS=(0 1 2)
for SEED in "${SEEDS[@]}"; do
    for TASK_NAME in "${TASK_NAMES[@]}"; do
        python train_cortex.py \
            --config-path=cortexbench_exp/configs/${ENV_NAME}/${POLiCY_NAME} \
            --config-name=${CONFIG_NAME} \
            env.env_name=${ENV_NAME} \
            env.task_name=${TASK_NAME} \
            train.seed=${SEED} \
            policy.embedding_type=${MODEL_TYPE} \
            train.mine_mi_loss_scale=${MINE} \
            train.mi_loss_scale=${MI} 
    done
done    