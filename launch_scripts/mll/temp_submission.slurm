#!/bin/bash
#SBATCH --job-name=libero-train
#SBATCH --output=/u/mrudolph/documents/BC-IB/slurm_jobs/libero/job_%j/job_%j.out
#SBATCH --error=/u/mrudolph/documents/BC-IB/slurm_jobs/libero/job_%j/job_%j.err
#SBATCH --partition=allnodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=24:00:00

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

python train_libero.py --config-path=libero_exp/configs/bc_cardpol_policy --config-name=transformer data.env_name=libero_object data.train_ratio=0.9 train.seed=4 train.train_gpus=[0] train.rep_loss_scale=0.01 train.rep_classifier_hidden=256 data.dual_task.enable=true data.dual_task.focused_task_id=3 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 policy.use_language_conditioning=false env.task_id=3 wandb.project=bc-cardpol-transformer wandb.group=cardpol_test_img+proprio wandb.policy_arch=transformer data.data_modality=[image,proprio]

echo "Job ended at $(date)"
