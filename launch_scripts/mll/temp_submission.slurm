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

python train_libero.py --config-path=libero_exp/configs/bc_vip_policy --config-name=vilt_distract data.env_name=libero_goal data.train_ratio=0.9 train.n_epochs=100 train.seed=0 train.train_gpus=[0] data.dual_task.enable=true data.dual_task.focused_task_id=9 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 data.dual_task.mixed_mode=vip policy.use_language_conditioning=false env.task_id=9 wandb.project=bc-vip-vilt wandb.group=bc-vip-alltasks-long-train_img_cam-agent_distract wandb.policy_arch=vilt_distract eval.enable_rollout=true eval.post_train_rollout.enable=true data.obs.modality.rgb=[agentview_rgb] data.data_modality=[image] data.obs.modality.low_dim=[] data.use_gripper=false data.use_joint=false data.use_ee=false
python train_libero.py --config-path=libero_exp/configs/bc_vip_policy --config-name=vilt_distract data.env_name=libero_goal data.train_ratio=0.9 train.n_epochs=100 train.seed=1 train.train_gpus=[0] data.dual_task.enable=true data.dual_task.focused_task_id=9 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 data.dual_task.mixed_mode=vip policy.use_language_conditioning=false env.task_id=9 wandb.project=bc-vip-vilt wandb.group=bc-vip-alltasks-long-train_img_cam-agent_distract wandb.policy_arch=vilt_distract eval.enable_rollout=true eval.post_train_rollout.enable=true data.obs.modality.rgb=[agentview_rgb] data.data_modality=[image] data.obs.modality.low_dim=[] data.use_gripper=false data.use_joint=false data.use_ee=false
python train_libero.py --config-path=libero_exp/configs/bc_vip_policy --config-name=vilt_distract data.env_name=libero_goal data.train_ratio=0.9 train.n_epochs=100 train.seed=2 train.train_gpus=[0] data.dual_task.enable=true data.dual_task.focused_task_id=9 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 data.dual_task.mixed_mode=vip policy.use_language_conditioning=false env.task_id=9 wandb.project=bc-vip-vilt wandb.group=bc-vip-alltasks-long-train_img_cam-agent_distract wandb.policy_arch=vilt_distract eval.enable_rollout=true eval.post_train_rollout.enable=true data.obs.modality.rgb=[agentview_rgb] data.data_modality=[image] data.obs.modality.low_dim=[] data.use_gripper=false data.use_joint=false data.use_ee=false
python train_libero.py --config-path=libero_exp/configs/bc_vip_policy --config-name=vilt_distract data.env_name=libero_goal data.train_ratio=0.9 train.n_epochs=100 train.seed=3 train.train_gpus=[0] data.dual_task.enable=true data.dual_task.focused_task_id=9 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 data.dual_task.mixed_mode=vip policy.use_language_conditioning=false env.task_id=9 wandb.project=bc-vip-vilt wandb.group=bc-vip-alltasks-long-train_img_cam-agent_distract wandb.policy_arch=vilt_distract eval.enable_rollout=true eval.post_train_rollout.enable=true data.obs.modality.rgb=[agentview_rgb] data.data_modality=[image] data.obs.modality.low_dim=[] data.use_gripper=false data.use_joint=false data.use_ee=false
python train_libero.py --config-path=libero_exp/configs/bc_vip_policy --config-name=vilt_distract data.env_name=libero_goal data.train_ratio=0.9 train.n_epochs=100 train.seed=4 train.train_gpus=[0] data.dual_task.enable=true data.dual_task.focused_task_id=9 data.dual_task.future_step_min=1 data.dual_task.future_step_max=10 data.dual_task.mixed_mode=vip policy.use_language_conditioning=false env.task_id=9 wandb.project=bc-vip-vilt wandb.group=bc-vip-alltasks-long-train_img_cam-agent_distract wandb.policy_arch=vilt_distract eval.enable_rollout=true eval.post_train_rollout.enable=true data.obs.modality.rgb=[agentview_rgb] data.data_modality=[image] data.obs.modality.low_dim=[] data.use_gripper=false data.use_joint=false data.use_ee=false

echo "Job ended at $(date)"
