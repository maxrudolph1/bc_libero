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

from ..data.get_dataset import get_dataset
from ..models import ResNet, ViT, R3M, VC1, MVP, Voltron, MPI
from ..utils.env_utils import build_env
from ..utils.train_utils import configure_cluster_GPUs, setup_optimizer, setup_lr_scheduler
from ..utils.record_utils import init_wandb, MetricLogger, AverageMeter, MetricMeter
from ..utils.results_utils import rollout_eval, rollout_train, save_success_rate, save_final_success_rate, merge_results

REGISTERED_ALGOS = {}


def register_algo(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_ALGOS:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_ALGOS[policy_name] = policy_class


def get_algo_class(algo_name):
    """Get the policy class from the registry."""
    if algo_name.lower() not in REGISTERED_ALGOS:
        raise ValueError(
            "Policy class with name {} not found in registry".format(algo_name)
        )
    return REGISTERED_ALGOS[algo_name.lower()]


def get_algo_list():
    return REGISTERED_ALGOS


class AlgoMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all algorithms that should not be registered here.
        _unregistered_algos = []

        if cls.__name__ not in _unregistered_algos:
            register_algo(cls)
        return cls


class BaseAlgo(nn.Module, metaclass=AlgoMeta):
    def __init__(self, cfg, inference=False):
        super().__init__()
        if inference:
            self.device = cfg.train.device
            load_path = cfg.eval.load_path
            self.eval_result_dir = os.path.join(load_path, "eval_results")
            os.makedirs(self.eval_result_dir, exist_ok=True)
            if cfg.eval.eval_all:
                ckpt_files = glob.glob(os.path.join(load_path, "*.ckpt"))
                filtered_ckpt_files = [os.path.basename(f) for f in ckpt_files if os.path.basename(f) != "model_final.ckpt"]
                model_list = sorted(filtered_ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            else:
                model_list = ["model_final.ckpt"]
            self.ckp_paths_to_eval = [os.path.join(load_path, model_i) for model_i in model_list]
            self.summary_file_path = os.path.join(self.eval_result_dir, f"summary_{cfg.env.task_name}.csv")
            self.final_summary_file_path = os.path.join(self.eval_result_dir, f"final_summary_{cfg.env.task_name}.csv")
            
        else:   # train
            physical_gpu_id = configure_cluster_GPUs(cfg.env.render_gpu_id)
            cfg.env.render_gpu_id = physical_gpu_id
            self.device = cfg.train.device
            if cfg.policy.embedding_type in ["ViT", "ViT_2"]:
                cfg.train.epochs = cfg.train.epochs * 2
                cfg.train.save_frequency = cfg.train.save_frequency * 2
            
            self.env = build_env(cfg)
            cfg.env.proprio_dim = self.env.env.proprio_dim
            cfg.policy.policy_head.output_size = self.env.spec.action_dim
            self.build_model(cfg)
            self.build_dataloader(cfg)
            
        self.highest_tr_score, self.highest_score = -np.inf, -np.inf
        self.highest_tr_success, self.highest_success = 0.0, 0.0

        self.cfg = cfg

    def build_dataloader(self, cfg):
        dataset, self.init_states, self.demo_score = get_dataset(cfg, self.model)
        self.train_loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=0,
            # pin_memory=True,
        )
    
    def build_model(self, cfg, use_downloaded=True):
        self.model = eval(cfg.policy.embedding_type)(cfg)
        self.model.to(self.device)
        
        self.optimizer = setup_optimizer(cfg.train.optimizer, self.model)
        self.scheduler = setup_lr_scheduler(self.optimizer, cfg.train.scheduler)

        # resume
        self.start_epoch = 1
        if cfg.train.resume:
            ckpt_files = glob.glob(os.path.join(cfg.train.resume_path, "*.ckpt"))
            ckpt_files = [os.path.basename(f) for f in ckpt_files]
            model_id = sorted(ckpt_files, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)[0]
            self.start_epoch = self.model.load(os.path.join(cfg.train.resume_path, model_id), self.optimizer, self.scheduler)
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

        if self.epoch == 1 or self.epoch % cfg.eval.eval_frequency == 0 or self.epoch == cfg.train.epochs:
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
            results_eval, is_best = rollout_eval(cfg, self.env, self.model, self.highest_score, self.highest_success, self.demo_score)
            self.highest_score = results_eval[f"rollout_eval/score_env_{cfg.env.suite}_task_{cfg.env.task_name}"]
            self.highest_success = results_eval[f"rollout_eval/success_env_{cfg.env.suite}_task_{cfg.env.task_name}"]

        if cfg.eval.enable_rollout_train:
            results_train = rollout_train(cfg, self.env, self.init_states, self.model, 
                                        self.highest_tr_score, self.highest_tr_success, self.demo_score)
            self.highest_tr_score = results_train[f"rollout_train/score_env_{cfg.env.suite}_task_{cfg.env.task_name}"]
            self.highest_tr_success = results_train[f"rollout_train/success_env_{cfg.env.suite}_task_{cfg.env.task_name}"]
        
        if cfg.eval.enable_rollout_eval and cfg.eval.enable_rollout_train:
            results = {k: v for d in (results_train, results_eval) for k, v in d.items()}
        elif cfg.eval.enable_rollout_eval and not cfg.eval.enable_rollout_train:
            results = results_eval
        
        gathered_results = [results]
        gathered_results = merge_results(gathered_results)
        None if cfg.dry_run else wandb.log(gathered_results, step=self.epoch)

        for k in list(gathered_results.keys()):
            if k.startswith("rollout_train/vis_") or k.startswith("rollout_eval/vis_"):
                gathered_results.pop(k)

        self.metric_logger.update(**gathered_results)

        if is_best:
            # self.model.save(f"{cfg.experiment_dir}/model_best.ckpt", self.epoch, self.optimizer, self.scheduler)
            with open(f"{cfg.experiment_dir}/best_epoch.txt", "a") as f:
                f.write(
                    "Best epoch: %d, Best %s: %.4f, Best %s: %.4f\n"
                    % (self.epoch, "score", self.highest_score, "success", self.highest_success)
                )
    
    def reset(self):
        self.policy.reset()

    def inference(self):
        cfg = self.cfg
        for ckp_path in self.ckp_paths_to_eval:
            epoch_name = os.path.basename(ckp_path).split('.ckpt')[0].split('_')[-1]
            video_save_dir=os.path.join(self.eval_result_dir, f"video_{cfg.env.env_name}_{cfg.env.task_name}/epoch_{epoch_name}")
            os.makedirs(video_save_dir, exist_ok=True)
            has_mp4 = any(file.endswith(".mp4") for file in os.listdir(video_save_dir))
            if has_mp4:
                print(f"Video of **{cfg.env.task_name}** of {cfg.env.env_name} in **epoch_{epoch_name}** has done!")
                continue

            results = self.inference_evaluate(
                checkpoint=ckp_path,
                video_save_dir=video_save_dir,
            )
            gathered_results = [results]
            gathered_results = merge_results(gathered_results)
            save_success_rate(epoch_name, gathered_results, self.summary_file_path)
        
        save_final_success_rate(self.summary_file_path, self.final_summary_file_path)

    def inference_evaluate(self, checkpoint, video_save_dir):
        cfg = self.cfg

        # set environment
        env = build_env(cfg)
        cfg.policy.policy_head.output_size = env.spec.action_dim
        cfg.env.proprio_dim = env.env.proprio_dim

        # load model and set optimizer
        model = eval(cfg.policy.embedding_type)(cfg)
        _ = model.load(checkpoint)
        model.to(self.device)
        cfg.train.optimizer.kwargs.lr = 0.
        optimizer = torch.optim.Adam(model.parameters(), cfg.train.optimizer.kwargs.lr)
        demo_score = get_dataset(cfg, model, return_demo_score=True)

        results_eval, _ = rollout_eval(cfg, env, model, self.highest_score, self.highest_success, demo_score,
                                       return_wandb_video=False, return_local=True, video_save_dir=video_save_dir)
        
        del env
        del model
        del optimizer
        # torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()

        return results_eval
