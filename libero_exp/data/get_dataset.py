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


def dual_task_dataset_kwargs(cfg):
    """Keyword args for DualTaskBatchDataset from ``cfg.data.dual_task``."""
    dt = cfg.data.dual_task
    return dict(
        focused_task_id=dt.focused_task_id,
        future_step_min=dt.future_step_min,
        future_step_max=dt.future_step_max,
        mixed_mode=dt.get("mixed_mode", "future_pair"),
        icvf_p_randomgoal=dt.get("icvf_p_randomgoal", 0.3),
        icvf_p_trajgoal=dt.get("icvf_p_trajgoal", 0.5),
        icvf_p_currgoal=dt.get("icvf_p_currgoal", 0.2),
        icvf_p_samegoal=dt.get("icvf_p_samegoal", 0.5),
        icvf_reward_scale=dt.get("icvf_reward_scale", 1.0),
        icvf_reward_shift=dt.get("icvf_reward_shift", -1.0),
    )


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
        (or VIP / ICVF frame tuples when mixed_mode is set accordingly)
    """

    _MIXED_MODES = ("future_pair", "vip", "icvf")

    def __init__(
        self,
        task_datasets,
        focused_task_id=0,
        future_step_min=1,
        future_step_max=10,
        mixed_mode="future_pair",
        icvf_p_randomgoal=0.3,
        icvf_p_trajgoal=0.5,
        icvf_p_currgoal=0.2,
        icvf_p_samegoal=0.5,
        icvf_reward_scale=1.0,
        icvf_reward_shift=-1.0,
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
        if mixed_mode not in self._MIXED_MODES:
            raise ValueError(
                f"mixed_mode must be one of {self._MIXED_MODES}, got {mixed_mode!r}"
            )
        icvf_p_sum = icvf_p_randomgoal + icvf_p_trajgoal + icvf_p_currgoal
        if mixed_mode == "icvf" and not np.isclose(icvf_p_sum, 1.0):
            raise ValueError(
                "icvf goal probabilities must sum to 1.0, got "
                f"random={icvf_p_randomgoal}, traj={icvf_p_trajgoal}, "
                f"curr={icvf_p_currgoal} (sum={icvf_p_sum})"
            )

        self.task_datasets = task_datasets
        self.focused_task_id = focused_task_id
        self.focused_dataset = task_datasets[focused_task_id]
        self.all_tasks_dataset = ConcatDataset(task_datasets)
        self.n_tasks = len(task_datasets)
        self._mixed_pool_size = len(self.all_tasks_dataset)
        self.future_step_min = future_step_min
        self.future_step_max = future_step_max
        self.mixed_mode = mixed_mode
        self.icvf_p_randomgoal = icvf_p_randomgoal
        self.icvf_p_trajgoal = icvf_p_trajgoal
        self.icvf_p_currgoal = icvf_p_currgoal
        self.icvf_p_samegoal = icvf_p_samegoal
        self.icvf_reward_scale = icvf_reward_scale
        self.icvf_reward_shift = icvf_reward_shift
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

    def _build_mixed_vip_frames(self, concat_idx, rng):
        """Sample a VIP training tuple (o_0, g, o_t, o_{t+1}) from one trajectory.

        Mirrors the sub-trajectory sampling in facebookresearch/vip: an initial
        frame ``o_0`` (start_ind), a goal frame ``g`` (end_ind >= start_ind), and a
        single-step transition (``o_t``, ``o_{t+1}``) with start_ind <= t < t+1 <= g.
        The self-supervised goal-reaching reward is R(s; g) = (s == g) - 1, i.e. -1
        for every non-goal state (VIP's constant living penalty).
        """
        task_id, local_idx, vl_dataset = self._resolve_concat_index(concat_idx)
        seq_dataset = vl_dataset.sequence_dataset
        _, index_in_demo, demo_length = seq_dataset.get_index_location(local_idx)

        last = demo_length - 1
        if demo_length <= 2:
            start_ind, end_ind = 0, last
        else:
            start_ind = int(rng.randint(0, demo_length - 2))        # [0, L-3]
            end_ind = int(rng.randint(start_ind + 1, demo_length))  # [start+1, L-1]
        if end_ind > start_ind:
            s0_ind = int(rng.randint(start_ind, end_ind))           # [start_ind, end_ind-1]
        else:
            s0_ind = start_ind
        s1_ind = min(s0_ind + 1, end_ind)

        reward = np.float32(float(s0_ind == end_ind) - 1.0)

        def _obs_at(abs_idx):
            return seq_dataset.get_single_obs(
                local_idx, timestep_offset=abs_idx - index_in_demo
            )

        return {
            "obs": _obs_at(s0_ind),             # o_t
            "obs_next": _obs_at(s1_ind),        # o_{t+1}
            "obs_initial": _obs_at(start_ind),  # o_0
            "obs_goal": _obs_at(end_ind),       # g
            "reward": reward,
            "task_emb": vl_dataset.task_emb,
            "task_id": task_id,
        }

    def _obs_at_concat(self, concat_idx, abs_timestep):
        """Fetch a single-timestep obs from a ConcatDataset index at an absolute demo time."""
        _, local_idx, vl_dataset = self._resolve_concat_index(concat_idx)
        seq_dataset = vl_dataset.sequence_dataset
        _, index_in_demo, demo_length = seq_dataset.get_index_location(local_idx)
        abs_timestep = int(np.clip(abs_timestep, 0, demo_length - 1))
        return seq_dataset.get_single_obs(
            local_idx, timestep_offset=abs_timestep - index_in_demo
        ), vl_dataset

    def _sample_icvf_goal(
        self,
        rng,
        concat_idx,
        t,
        demo_length,
        *,
        p_randomgoal,
        p_trajgoal,
        p_currgoal,
    ):
        """Sample an ICVF goal observation and its absolute timestep (GCSDataset-style).

        Returns (goal_obs, goal_abs_t_or_none). ``goal_abs_t_or_none`` is the absolute
        timestep in the *current* demo when the goal is on-trajectory (curr / traj);
        ``None`` when the goal is drawn from a random other demo (never equals ``t``).
        """
        u = float(rng.rand())
        if u < p_currgoal:
            goal_obs, _ = self._obs_at_concat(concat_idx, t)
            return goal_obs, t

        if u < p_currgoal + p_trajgoal:
            final_t = demo_length - 1
            distance = float(rng.rand())
            goal_t = int(np.round(t * distance + final_t * (1.0 - distance)))
            goal_t = int(np.clip(goal_t, 0, final_t))
            goal_obs, _ = self._obs_at_concat(concat_idx, goal_t)
            return goal_obs, goal_t

        # Random goal from another trajectory in the mixed pool.
        other_idx = int(rng.randint(0, self._mixed_pool_size))
        _, other_local, other_vl = self._resolve_concat_index(other_idx)
        other_seq = other_vl.sequence_dataset
        _, other_index_in_demo, other_demo_length = other_seq.get_index_location(
            other_local
        )
        other_t = int(rng.randint(0, other_demo_length))
        goal_obs = other_seq.get_single_obs(
            other_local, timestep_offset=other_t - other_index_in_demo
        )
        return goal_obs, None

    def _build_mixed_icvf_frames(self, concat_idx, rng):
        """Sample an ICVF training tuple (s, s', s_+, z) from passive demos.

        Mirrors ``GCSDataset`` in dibyaghosh/icvf_release: outcome ``goals`` (s_+) and
        intention ``desired_goals`` (z) are sampled with curr / same-traj / random
        probabilities; with probability ``p_samegoal`` the outcome equals the intention.
        Rewards use index equality: R = scale * 1[s == g] + shift (defaults: 0 at goal,
        -1 elsewhere), with masks = 1 - success for TD bootstrapping.
        """
        task_id, local_idx, vl_dataset = self._resolve_concat_index(concat_idx)
        seq_dataset = vl_dataset.sequence_dataset
        _, index_in_demo, demo_length = seq_dataset.get_index_location(local_idx)

        if demo_length <= 1:
            t = 0
            t_next = 0
        else:
            t = int(rng.randint(0, demo_length - 1))
            t_next = t + 1

        desired_obs, desired_t = self._sample_icvf_goal(
            rng,
            concat_idx,
            t,
            demo_length,
            p_randomgoal=self.icvf_p_randomgoal,
            p_trajgoal=self.icvf_p_trajgoal,
            p_currgoal=self.icvf_p_currgoal,
        )
        if float(rng.rand()) < self.icvf_p_samegoal:
            goal_obs, goal_t = desired_obs, desired_t
        else:
            goal_obs, goal_t = self._sample_icvf_goal(
                rng,
                concat_idx,
                t,
                demo_length,
                p_randomgoal=self.icvf_p_randomgoal,
                p_trajgoal=self.icvf_p_trajgoal,
                p_currgoal=self.icvf_p_currgoal,
            )

        success = goal_t is not None and int(goal_t) == int(t)
        desired_success = desired_t is not None and int(desired_t) == int(t)
        reward = np.float32(
            float(success) * self.icvf_reward_scale + self.icvf_reward_shift
        )
        desired_reward = np.float32(
            float(desired_success) * self.icvf_reward_scale + self.icvf_reward_shift
        )
        mask = np.float32(1.0 - float(success))
        desired_mask = np.float32(1.0 - float(desired_success))

        def _obs_at(abs_idx):
            return seq_dataset.get_single_obs(
                local_idx, timestep_offset=abs_idx - index_in_demo
            )

        return {
            "obs": _obs_at(t),
            "obs_next": _obs_at(t_next),
            "obs_goal": goal_obs,                    # s_+ (outcome)
            "obs_desired_goal": desired_obs,         # z (intention)
            "reward": reward,
            "desired_reward": desired_reward,
            "mask": mask,
            "desired_mask": desired_mask,
            "task_emb": vl_dataset.task_emb,
            "task_id": task_id,
        }

    def _build_mixed_sample(self, concat_idx, rng):
        if self.mixed_mode == "vip":
            return self._build_mixed_vip_frames(concat_idx, rng)
        if self.mixed_mode == "icvf":
            return self._build_mixed_icvf_frames(concat_idx, rng)
        return self._build_mixed_future_pair(concat_idx, rng)

    def __getitem__(self, idx):
        focused_idx = idx % len(self.focused_dataset)
        mixed_idx = self._sample_mixed_index()
        rng = np.random.RandomState(seed=(int(idx) + 1) * 9973 + int(mixed_idx))
        return {
            "focused": self.focused_dataset[focused_idx],
            "mixed": self._build_mixed_sample(mixed_idx, rng),
        }


def collate_dual_task_batch(batch):
    """Collate a list of dual-task samples into two batched dicts."""
    focused_samples = [sample["focused"] for sample in batch]
    mixed_samples = [sample["mixed"] for sample in batch]
    return {
        "focused": default_collate(focused_samples),
        "mixed": default_collate(mixed_samples),
    }
