from typing import Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

def create_dataloader(
    buffer, 
    batch_size: int, 
    num_workers=4, 
):
    return iter(DataLoader(buffer, batch_size=batch_size, num_workers=num_workers))

from typing import Dict
from dataclasses import dataclass

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

@dataclass
class DataStatistics:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray

    def __post_init__(self):
        self.mean = np.array(self.mean, dtype=np.float32)
        self.std = np.array(self.std, dtype=np.float32)
        self.min = np.array(self.min, dtype=np.float32)
        self.max = np.array(self.max, dtype=np.float32)

        # check shapes
        assert self.mean.shape == self.std.shape == self.min.shape == self.max.shape

        # check ordering
        assert np.all(self.min <= self.max)


class OnlineTrajectoryBuffer():
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            seq_len: int,
            dataset = None,
            buffer_size: int = 1000,
            episode_len: int = 1001,
            discount: float=1.0,
            return_scale: float = 1.0,
            max_len: int = 1000,
            device: Union[str, torch.device] = "cuda"
    ) -> None:
        self.buffer_size = buffer_size
        self.seq_len = seq_len

        if discount > 1.0:
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount
            self.use_avg = False

        self.return_scale = return_scale
        self.device = device
        self.max_len = max_len

        # max_seq_len = episode_len + seq_len
        #self.offline_data_stastic = self.trajectory_statistics
        self.data = {
            "observations": np.zeros([buffer_size, max_len, state_dim], dtype=np.float32),
            "actions": np.zeros([buffer_size, max_len, action_dim], dtype=np.float32),
            "next_observations": np.zeros([buffer_size, max_len, state_dim], dtype=np.float32),
            "rewards": np.zeros([buffer_size, max_len, 1], dtype=np.float32),
            "terminals": np.zeros([buffer_size, max_len, 1], dtype=np.float32),
            "masks": np.zeros([buffer_size, max_len], dtype=np.float32),
        # "values": np.zeros([buffer_size, max_seq_len, 1], dtype=np.float32),
        }
        self.cur_traj = 0
        self.traj_num = 0
        self.sample_num = 0
        self.traj_len = np.zeros([buffer_size, ], dtype=np.int32)
        self.sample_prob = np.zeros([buffer_size, ])  #

    # def add_offline_traj(self, dataset, min_return = 2000):
    #     converted_dataset = {
    #         "observations": dataset["observations"],
    #         "actions": dataset["actions"],
    #         "rewards": dataset["rewards"],
    #         "terminals": dataset["terminals"],
    #         "next_observations": dataset["next_observations"],
    #     }
    #
    #     traj_start = 0
    #     for i in range(dataset["rewards"].shape[0]):
    #         if dataset["ends"][i]:
    #             this_traj_len = i + 1 - traj_start
    #             total_return = np.sum(dataset["rewards"][traj_start:i + 1])
    #             if total_return > min_return:
    #                 for k, v in converted_dataset.items():
    #                     self.data[k][self.cur_traj, :this_traj_len] = v[traj_start:traj_start + this_traj_len]
    #                 self.data["masks"][self.cur_traj, :this_traj_len] = 1
    #                 self.traj_num = min(self.traj_num + 1, self.buffer_size)
    #                 self.sample_num = self.sample_num - self.traj_len[self.cur_traj] + this_traj_len
    #                 self.traj_len[self.cur_traj] = this_traj_len
    #                 self.cur_traj = (self.cur_traj + 1) % self.buffer_size
    #             traj_start = i + 1
    #     # self.sample_prob = self.traj_len / self.sample_num
    #     self._setup()

    def add_offline_traj(self, dataset, per = 0.05):
        converted_dataset = {
            "observations": dataset["observations"],
            "actions": dataset["actions"],
            "rewards": dataset["rewards"],
            "terminals": dataset["terminals"],
            "next_observations": dataset["next_observations"],
        }
        traj_start = 0
        traj_infos = []
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                traj_end = i + 1
                this_traj_len = traj_end - traj_start
                total_return = np.sum(dataset["rewards"][traj_start:traj_end])
                traj_infos.append({
                    'start': traj_start,
                    'end': traj_end,
                    'length': this_traj_len,
                    'return': total_return
                })
                traj_start = traj_end

        traj_infos.sort(key=lambda x: x['return'], reverse=True)

        num_trajs_to_keep = int(len(traj_infos) * per)
        print("num_trajs_to_keep",num_trajs_to_keep)
        top_trajs = traj_infos[:num_trajs_to_keep]
        print("visual",traj_infos[:5])
        for traj_info in top_trajs:
            start = traj_info['start']
            end = traj_info['end']
            this_traj_len = traj_info['length']
            for k, v in converted_dataset.items():
                self.data[k][self.cur_traj, :this_traj_len] = v[start:end]
            self.data["masks"][self.cur_traj, :this_traj_len] = 1
            self.traj_num = min(self.traj_num + 1, self.buffer_size)
            self.sample_num = self.sample_num - self.traj_len[self.cur_traj] + this_traj_len
            self.traj_len[self.cur_traj] = this_traj_len
            self.cur_traj = (self.cur_traj + 1) % self.buffer_size

        self._setup()


    def add_online_traj(self, online_trajs):
        num, traj_len = online_trajs["observations"].shape[0], online_trajs["observations"].shape[1]
        index_to_go = (self.cur_traj + np.arange(num)) % self.buffer_size
        old_num_sample = self.traj_len[index_to_go].sum()
        self.traj_len[index_to_go] = online_trajs["masks"].sum(-1)
        new_num_sample = self.traj_len[index_to_go].sum()
        self.sample_num = self.sample_num - old_num_sample + new_num_sample
        self.cur_traj = (self.cur_traj + num) % self.buffer_size
        self.traj_num = min(self.traj_num + num, self.buffer_size)
        self.data["masks"][index_to_go] = 0

        for k, v in online_trajs.items():
            v_len = v.shape[1]
            self.data[k][index_to_go, :v_len, ...] = v
        self.sample_prob = self.traj_len / self.sample_num
        self._setup()

    def _setup(self):
        # self.use_avg = True
        keep_idx = []
        index_map = {}
        count = 0
        traj_count = 0
        for idx, pl in enumerate(self.traj_len):
            if pl < self.seq_len:
                pass
            else:
                keep_idx.append(idx)
                for i in range(pl - self.seq_len + 1):
                    index_map[count] = (traj_count, i)
                    count += 1
                traj_count += 1
        self.index_map = index_map
        self.observations_segmented = self.data["observations"]
        self.actions_segmented = self.data["actions"]
        self.rewards_segmented = self.data["rewards"]
        self.terminals_segmented = self.data["terminals"]
        self.next_observations_segmented = self.data["next_observations"]
        self.masks_segmented = self.data["masks"]

        self.observations = self.observations_segmented[keep_idx]
        self.actions = self.actions_segmented[keep_idx]
        self.rewards = self.rewards_segmented[keep_idx]
        self.terminals = self.terminals_segmented[keep_idx]
        self.next_observations = self.next_observations_segmented[keep_idx]
        self.masks = self.masks_segmented[keep_idx]


        self.discounts = (self.discount ** np.arange(self.max_len))[:, None]
        self.values_segmented = np.zeros(self.rewards.shape)
        for t in range(self.max_len):
            V = (self.rewards[:, t + 1 :] * self.discounts[: -t - 1]).sum(axis=1)
            self.values_segmented[:, t] = V
        N_p, Max_Path_Len, _ = self.values_segmented.shape
        if self.use_avg:
            divisor = np.arange(1, Max_Path_Len + 1)[::-1][None, :, None]
            self.values_segmented = self.values_segmented / divisor

    def __len__(self):
        return len(self.index_map)
        # return self.size

    def get_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[traj_index],
            "actions": self.actions[traj_index],
            "rewards": self.rewards[traj_index],
            "terminals":self.terminals[traj_index],
            "next_observations": self.next_observations[traj_index],
            "returns": self.values_segmented[traj_index],
            "masks": self.masks[traj_index],
        }

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        idx, start_idx = self.index_map[index]
        traj = self.get_trajectory(idx)
        return {
            k: v[start_idx: start_idx + self.seq_len] for k, v in traj.items()
        }

    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        """Shapes of the trajectories in the dataset."""

        trajectories = {
            "states": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "returns": self.values_segmented,
        }

        # average over samples and time
        ret_dict = {
            k: DataStatistics(
                mean=v.mean(axis=(0, 1)),
                std=v.std(axis=(0, 1)),
                min=v.min(axis=(0, 1)),
                max=v.max(axis=(0, 1)),
            )
            for k, v in trajectories.items()
        }
        return ret_dict



