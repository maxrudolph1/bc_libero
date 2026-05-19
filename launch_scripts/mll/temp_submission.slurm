#!/bin/bash
#SBATCH --job-name=libero-train
#SBATCH --output=/u/mrudolph/documents/BC-IB/slurm_jobs/libero/job_%j/job_%j.out
#SBATCH --error=/u/mrudolph/documents/BC-IB/slurm_jobs/libero/job_%j/job_%j.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=8:00:00

source ~/.bashrc
cd /u/mrudolph/documents/BC-IB
source .venv/bin/activate

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

echo "MUJOCO_GL=$MUJOCO_GL"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

python train_libero.py --config-path=libero_exp/configs/bc_cardpol_policy --config-name=transformer data.env_name=libero_spatial data.train_ratio=0.9 train.seed=4 train.mine_mi_loss_scale=0.1 train.mi_loss_scale=0.001 train.train_gpus=[0] train.rep_loss_scale=0.0 train.rep_classifier_hidden=256 data.dual_task.enable=true data.dual_task.focused_task_id=7 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 env.task_id=7 wandb.project=bc-cardpol-transformer wandb.group=cardpol_logging_05172026
python train_libero.py --config-path=libero_exp/configs/bc_cardpol_policy --config-name=transformer data.env_name=libero_spatial data.train_ratio=0.9 train.seed=5 train.mine_mi_loss_scale=0.1 train.mi_loss_scale=0.001 train.train_gpus=[0] train.rep_loss_scale=0.0 train.rep_classifier_hidden=256 data.dual_task.enable=true data.dual_task.focused_task_id=7 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 env.task_id=7 wandb.project=bc-cardpol-transformer wandb.group=cardpol_logging_05172026

echo "Job ended at $(date)"
