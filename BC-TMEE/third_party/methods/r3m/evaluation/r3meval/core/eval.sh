cd /baishuanghao/code/r3m/evaluation/r3meval/core
source activate /root/miniconda3/envs/libero

python hydra_launcher.py \
    hydra/launcher=local \
    hydra/output=local \
    env="kitchen_sdoor_open-v3" \
    camera="left_cap2" \
    pixel_based=true \
    embedding=resnet50 \
    num_demos=5 \
    env_kwargs.load_path=r3m \
    bc_kwargs.finetune=false \
    proprio=9 \
    job_name=r3m_repro \
    seed=125