"""
    Copied from <libero.lifelong.datasets>

    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
    we should in principle use seq_len, but the paddings of the two are different.
    So that's why we currently use frame_stack instead of seq_len.
"""

import random

import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import default_collate
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from .dataset import SequenceDataset


def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    train_ratio=None,
    train=True,
    return_shape_meta=False,
    val_demo_num=None,
    *args,
    **kwargs
):
    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    if return_shape_meta:
        return shape_meta

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        train_ratio=train_ratio,
        train=train,
        val_demo_num=val_demo_num,
    )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict


class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        self.sequence_dataset = sequence_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)


def validate_dual_task_cfg(cfg):
    """Require env.task_id to match data.dual_task.focused_task_id when dual-task is on."""
    if not cfg.data.dual_task.enable:
        return

    focused_task_id = cfg.data.dual_task.focused_task_id
    env_task_id = cfg.env.task_id
    if env_task_id is None:
        raise ValueError(
            "data.dual_task.enable=true requires env.task_id to be set to the "
            f"focused task index (got None). Set env.task_id=[{focused_task_id}]."
        )

    if isinstance(env_task_id, int):
        env_task_ids = [env_task_id]
    else:
        env_task_ids = list(env_task_id)

    if len(env_task_ids) != 1:
        raise ValueError(
            "data.dual_task.enable=true requires env.task_id to contain exactly one "
            f"task index, got {env_task_ids}. "
            f"Set env.task_id=[{focused_task_id}]."
        )

    if env_task_ids[0] != focused_task_id:
        raise ValueError(
            f"data.dual_task.focused_task_id ({focused_task_id}) must match "
            f"env.task_id ({env_task_ids[0]}). Set both to the same task index."
        )


class DualTaskBatchDataset(Dataset):
    """
    Pairs one sample from a fixed task with one mixed sample from all tasks.

    Each __getitem__ returns {"focused": sample, "mixed": sample}. Use
    collate_dual_task_batch in the DataLoader to obtain two batches per step:

        data["focused"]  # standard BC sequence batch
        data["mixed"]    # obs at t and obs at t+K (same trajectory, random K)

    Mixed samples contain:
        obs, obs_future, task_emb, task_id, future_step_k
    """

    def __init__(
        self,
        task_datasets,
        focused_task_id=0,
        future_step_min=1,
        future_step_max=10,
    ):
        if not task_datasets:
            raise ValueError("task_datasets must be a non-empty list of per-task datasets")
        if focused_task_id < 0 or focused_task_id >= len(task_datasets):
            raise ValueError(
                f"focused_task_id={focused_task_id} out of range for "
                f"{len(task_datasets)} tasks"
            )
        if future_step_min < 1:
            raise ValueError(f"future_step_min must be >= 1, got {future_step_min}")
        if future_step_max < future_step_min:
            raise ValueError(
                f"future_step_max ({future_step_max}) must be >= "
                f"future_step_min ({future_step_min})"
            )

        self.task_datasets = task_datasets
        self.focused_task_id = focused_task_id
        self.focused_dataset = task_datasets[focused_task_id]
        self.all_tasks_dataset = ConcatDataset(task_datasets)
        self.n_tasks = len(task_datasets)
        self._mixed_pool_size = len(self.all_tasks_dataset)
        self.future_step_min = future_step_min
        self.future_step_max = future_step_max
        self._concat_cumulative_sizes = self.all_tasks_dataset.cumulative_sizes

    def __len__(self):
        return len(self.focused_dataset)

    def _sample_mixed_index(self):
        return random.randrange(self._mixed_pool_size)

    def _resolve_concat_index(self, concat_idx):
        if concat_idx < 0:
            raise IndexError(f"concat_idx must be non-negative, got {concat_idx}")
        dataset_idx = 0
        if self._concat_cumulative_sizes:
            dataset_idx = int(
                np.searchsorted(self._concat_cumulative_sizes, concat_idx, side="right")
            )
        local_idx = concat_idx
        if dataset_idx > 0:
            local_idx = concat_idx - self._concat_cumulative_sizes[dataset_idx - 1]
        return dataset_idx, local_idx, self.task_datasets[dataset_idx]

    def _build_mixed_future_pair(self, concat_idx, rng):
        task_id, local_idx, vl_dataset = self._resolve_concat_index(concat_idx)
        seq_dataset = vl_dataset.sequence_dataset
        _, index_in_demo, demo_length = seq_dataset.get_index_location(local_idx)

        window_end = min(index_in_demo + seq_dataset.seq_length - 1, demo_length - 1)
        t = int(rng.randint(index_in_demo, window_end + 1))

        future_k = int(rng.randint(self.future_step_min, self.future_step_max + 1))
        future_t = int(np.clip(t + future_k, 0, demo_length - 1))
        actual_k = future_t - t

        obs = seq_dataset.get_single_obs(local_idx, timestep_offset=t - index_in_demo)
        obs_future = seq_dataset.get_single_obs(
            local_idx, timestep_offset=future_t - index_in_demo
        )

        return {
            "obs": obs,
            "obs_future": obs_future,
            "task_emb": vl_dataset.task_emb,
            "task_id": task_id,
            "future_step_k": actual_k,
        }

    def __getitem__(self, idx):
        focused_idx = idx % len(self.focused_dataset)
        mixed_idx = self._sample_mixed_index()
        rng = np.random.RandomState(seed=(int(idx) + 1) * 9973 + int(mixed_idx))
        return {
            "focused": self.focused_dataset[focused_idx],
            "mixed": self._build_mixed_future_pair(mixed_idx, rng),
        }


def collate_dual_task_batch(batch):
    """Collate a list of dual-task samples into two batched dicts."""
    focused_samples = [sample["focused"] for sample in batch]
    mixed_samples = [sample["mixed"] for sample in batch]
    return {
        "focused": default_collate(focused_samples),
        "mixed": default_collate(mixed_samples),
    }
