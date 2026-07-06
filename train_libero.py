import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa' 
# os.environ['MUJOCO_GL'] = 'egl'             
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import hydra
import warnings
import lightning
from omegaconf import DictConfig

from libero_exp.algos import *
from libero_exp.utils.run_utils import save_run_configs, setup_run_output_dir


@hydra.main(config_path="libero_exp/configs/bc_policy", config_name="vilt", version_base=None)
def main(cfg: DictConfig):
    run_dir = setup_run_output_dir(cfg)
    warnings.simplefilter("ignore")
    lightning.seed_everything(cfg.train.seed)
    save_run_configs(cfg, run_dir)

    algo = get_algo_class(cfg.algo.algo_type)(cfg)
    algo.train()


if __name__ == "__main__":
    main()
