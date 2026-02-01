import os
import gc
import glob
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import logging
import numpy as np

from .base import AlgoMeta
from ..data.get_dataset import get_dataset_trifinger
from ..models_trifinger import ResNet, ViT, R3M, VC1, MVP, Voltron, MPI
from ..utils.env_trifinger_utils import build_env
from ..utils.train_utils import setup_optimizer, setup_lr_scheduler
from ..utils.record_utils import init_wandb, MetricLogger, AverageMeter, MetricMeter
from ..utils.results_trifinger_utils import rollout, merge_results


class BaseAlgo_Trifinger(nn.Module, metaclass=AlgoMeta):
    def __init__(self, cfg, inference=False):
        super().__init__()
        if cfg.env.task_name == 'move':
            cfg.env.proprio_dim = 9
            cfg.data.action_dim = 9
            cfg.data.goal_type = 'goal_cond'
            if cfg.train.ft_method == 'partial_ft':
                cfg.train.epochs = 1000
                cfg.train.save_frequency = 100
        elif cfg.env.task_name == 'reach':
            cfg.env.proprio_dim = 9
            cfg.data.action_dim = 3
            cfg.data.goal_type = 'goal_none'

        if inference:
            self.device = cfg.train.device
            load_path = cfg.eval.load_path
            self.eval_result_dir = os.path.join(load_path, "eval_results")
            os.makedirs(self.eval_result_dir, exist_ok=True)
            if cfg.eval.eval_all:
                ckpt_files = glob.glob(os.path.join(load_path, "*.ckpt"))
                filtered_ckpt_files = [os.path.basename(f) for f in ckpt_files if os.path.basename(f) != "model_final.ckpt"]
                model_list = sorted(filtered_ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=False)
            else:
                model_list = ["model_final.ckpt"]
            self.ckp_paths_to_eval = [os.path.join(load_path, model_i) for model_i in model_list]
            self.summary_file_path = os.path.join(self.eval_result_dir, f"summary_{cfg.env.task_name}.csv")
            self.final_summary_file_path = os.path.join(self.eval_result_dir, f"final_summary_{cfg.env.task_name}.csv")
        else:   # train
            self.device = cfg.train.device
            self.build_model(cfg)
            self.build_dataloader(cfg)
            self.env = build_env(cfg, self.traj_info)
            
        self.highest_tr_score, self.highest_score = -np.inf, -np.inf
        self.highest_tr_success, self.highest_success = 0.0, 0.0
        
        self.max_dict = {}
        for sim_env_name in cfg.env.eval_envs:
            self.max_dict[sim_env_name] = {"train": {}, "test": {}}

        self.cfg = cfg

    def build_dataloader(self, cfg): 
        train_dataset, test_dataset, traj_info = get_dataset_trifinger(cfg, self.model)
        self.traj_info = traj_info
        self.train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    
    def build_model(self, cfg, use_downloaded=True):
        self.model = eval(cfg.policy.embedding_type)(cfg)
        self.model.to(self.device)
        
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)
        self.scheduler = setup_lr_scheduler(self.optimizer, cfg.train.scheduler)

        # resume
        self.start_epoch = 1
        if cfg.train.resume:
            self.start_epoch = self.model.load(cfg.train.resume_path, self.optimizer, self.scheduler)
            self.model.to(self.device)

    def train(self):
        cfg = self.cfg
        self.before_train()

        print('\nTraining...')
        for self.epoch in self.metric_logger.log_every(range(self.start_epoch, cfg.train.epochs+1), cfg.eval.eval_frequency, ""):
            self.before_epoch()
            train_metrics = self.run_epoch()
            self.after_epoch(train_metrics)

        self.after_train()

    def before_train(self):
        cfg = self.cfg
        self.metric_logger = MetricLogger(delimiter=" ")
        None if cfg.dry_run else init_wandb(cfg)

        self.losses = MetricMeter()
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()

    def before_epoch(self):
        self.model.train()

    def run_epoch(self):
        cfg = self.cfg
        tot_loss_dict, tot_items = {}, 0

        end = time.time()
        for batch_id, batch_data in enumerate(self.train_loader):
            self.data_time.update(time.time() - end)
            ret_dict = self.forward_backward(batch_data)          # backward propagation

            self.batch_time.update(time.time() - end)
            self.losses.update(ret_dict)
            
            for k, v in ret_dict.items():
                if k not in tot_loss_dict:
                    tot_loss_dict[k] = 0
                tot_loss_dict[k] += v
            tot_items += 1

            if (batch_id + 1) % cfg.train.print_frequency == 0 or len(self.train_loader) < cfg.train.print_frequency:
                nb_remain = 0
                nb_remain += len(self.train_loader) - batch_id - 1
                nb_remain += (cfg.train.epochs - self.epoch) * len(self.train_loader)
                eta_seconds = self.batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                logging.info("epoch [{0}/{1}][{2}/{3}]\t"
                    "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "eta {eta}\t"
                    "{losses}\t"
                    "lr {lr:.6e}".format(
                        self.epoch,
                        cfg.train.epochs,
                        batch_id + 1,
                        len(self.train_loader),
                        batch_time=self.batch_time,
                        data_time=self.data_time,
                        eta=eta,
                        losses=self.losses,
                        lr=self.optimizer.param_groups[0]['lr'],
                    ))
                    
            end = time.time()
                    
        out_dict = {}
        for k, v in tot_loss_dict.items():
            out_dict[f"train/{k}"] = tot_loss_dict[f"{k}"] / tot_items

        if self.scheduler is not None:
            self.scheduler.step()

        return out_dict
    
    def forward_backward(self, data):
        loss = self.model.compute_loss(data)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.)
        self.optimizer.step()

        ret_dict = {
            "loss": loss.item(),
        }

        return ret_dict

    def after_epoch(self, train_metrics):
        self.model.eval()
        cfg = self.cfg
        train_metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
        self.metric_logger.update(**train_metrics)

        None if cfg.dry_run else wandb.log(train_metrics, step=self.epoch)

        if self.epoch % cfg.eval.eval_frequency == 0 or self.epoch == cfg.train.epochs:
            if cfg.eval.enable_rollout_eval:
                self.rollout_tasks()

        if self.epoch % cfg.train.save_frequency == 0:
            self.model.save(f"{cfg.experiment_dir}/model_{self.epoch}.ckpt", self.epoch, self.optimizer, self.scheduler)

    def after_train(self):
        cfg = self.cfg
        self.model.save(f"{cfg.experiment_dir}/model_final.ckpt", self.epoch, self.optimizer, self.scheduler)
        None if cfg.dry_run else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry_run else wandb.finish()

    def rollout_tasks(self):
        cfg = self.cfg
        if cfg.eval.enable_rollout_eval:
            results_eval, max_eval, best_list = rollout(cfg, self.env, self.model, self.traj_info, self.max_dict, 
                                                        mode='test', epoch=self.epoch)
            self.max_dict = max_eval

        if cfg.eval.enable_rollout_train:
            results_train, _, _ = rollout(cfg, self.env, self.model, self.traj_info, self.max_dict, 
                                          mode='train', epoch=self.epoch)
        
        if cfg.eval.enable_rollout_eval and cfg.eval.enable_rollout_train:
            gathered_results = [results_eval, results_train]
        elif cfg.eval.enable_rollout_eval and not cfg.eval.enable_rollout_train:
            gathered_results = [results_eval]
        gathered_results = merge_results(gathered_results)
        
        final_results = {}
        for env_name, env_results in gathered_results.items():
            if not isinstance(env_results, list):
                env_results = [env_results]
            for env_result in env_results:
                for mode, mode_results in env_result.items():
                    for key in list(mode_results.keys()): 
                        if "traj-" in key:
                            continue
                        else:
                            final_results[f"{env_name}_{mode}/{key}"] = mode_results[key]

            if best_list[env_name]:
                # self.model.save(f"{cfg.experiment_dir}/{env_name}_model_best.ckpt", self.epoch, self.optimizer, self.scheduler)
                with open(f"{cfg.experiment_dir}/{env_name}_best_epoch.txt", "a") as f:
                    f.write(
                        "Best epoch: %d, Best %s: %.4f\n"
                        % (self.epoch, "success", list(self.max_dict[env_name]['test'].values())[0])
                    )
                    
        None if cfg.dry_run else wandb.log(final_results, step=self.epoch)
        self.metric_logger.update(**final_results)

    def inference(self):
        cfg = self.cfg
        for ckp_path in self.ckp_paths_to_eval:
            epoch_name = os.path.basename(ckp_path).split('.ckpt')[0].split('_')[-1]
            video_save_dir=os.path.join(self.eval_result_dir, cfg.env.eval_envs[0], 'test', "epoch_"+str(epoch_name))
            os.makedirs(video_save_dir, exist_ok=True)
            has_gif = any(file.endswith(".gif") for file in os.listdir(video_save_dir))
            if has_gif:
                print(f"Video of **{cfg.env.task_name}** of {cfg.env.env_name} in **epoch_{epoch_name}** has done!")
                continue

            results = self.inference_evaluate(
                checkpoint=ckp_path,
                epoch_name=epoch_name,
            )
            gathered_results = [results]
            gathered_results = merge_results(gathered_results)
            for env_name, env_results in gathered_results.items():
                with open(f"{self.eval_result_dir}/{env_name}_best_epoch.txt", "a") as f:
                    f.write(
                        "epoch: %d, %s: %.4f\n"
                        % (int(epoch_name), "success", list(self.max_dict[env_name]['test'].values())[0])
                    )

    def inference_evaluate(self, checkpoint, epoch_name):
        cfg = self.cfg

        # load model and set optimizer
        model = eval(cfg.policy.embedding_type)(cfg)
        _ = model.load(checkpoint)
        model.to(self.device)
        cfg.train.optimizer.kwargs.lr = 0.
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.optimizer.kwargs.lr)

        traj_info = get_dataset_trifinger(cfg, model, return_traj_info=True)
        env = build_env(cfg, traj_info)

        results_eval, max_eval, _ = rollout(
            cfg, env, model, traj_info, self.max_dict, 
            mode='test', epoch=epoch_name,
            eval_dir=self.eval_result_dir,
        )
        self.max_dict = max_eval
        
        del env
        del model
        del optimizer
        # torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()

        return results_eval
