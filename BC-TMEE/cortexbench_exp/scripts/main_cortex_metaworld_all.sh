#!/bin/bash

# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0      # if libgpu_partition.so confilts with gym and robosuite
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export CUDA_VISIBLE_DEVICES=0

cd /baishuanghao/code/BC-IB
source activate bcib

POLiCY_NAME=$1
CONFIG_NAME=$2
MODEL_TYPE=$3

ENV_NAME=metaworld
TASK_NAMES=(assembly-v2-goal-observable bin-picking-v2-goal-observable button-press-topdown-v2-goal-observable drawer-open-v2-goal-observable hammer-v2-goal-observable)

MINE=0.1
MI=1e-3

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_policy full_ft_temporal_fuse ResNet
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_policy full_ft_spatial_fuse ResNet 
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_policy partial_ft_temporal_fuse VC1 
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_policy partial_ft_spatial_fuse VC1 

# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_ib_policy full_ft_temporal_fuse ResNet 
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_ib_policy full_ft_spatial_fuse ResNet 
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_ib_policy partial_ft_temporal_fuse VC1 
# /baishuanghao/code/BC-IB/cortexbench_exp/scripts/main_cortex_metaworld_all.sh bc_ib_policy partial_ft_spatial_fuse VC1

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
    