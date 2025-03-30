import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from abc import ABC, abstractmethod
from typing import Any, Dict
from multiprocessing import current_process

# from offlinerllib.buffer.base import Buffer
# from offlinerllib.utils.functional import discounted_cum_sum

from dataclasses import dataclass
import random
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


class Buffer(ABC):
    @abstractmethod
    def random_batch(self, batch_size: int) -> Dict[str, Any]:
        raise NotImplementedError

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


# converted_dataset = {
#     "observations": dataset["observations"].astype(np.float32),
#     "actions": dataset["actions"].astype(np.float32),
#     "rewards": dataset["rewards"][:, None].astype(np.float32),
#     "terminals": dataset["terminals"][:, None].astype(np.float32),
#     "next_observations": dataset["next_observations"].astype(np.float32)
# }

class SequenceDataset:
    def __init__(
            self,
            dataset,
            seq_len: int,
            discount: float = 1.0,
            return_scale: float = 1.0,
            max_len: int = 1001
    ) -> None:
        converted_dataset = dataset
        traj, traj_len = [], []
        self.seq_len = seq_len
        self.discount = discount
        self.return_scale = return_scale
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            # this_traj_len = i + 1 - traj_start
            if dataset["ends"][i]:
                episode_data = {k: v[traj_start:i + 1] for k, v in converted_dataset.items()}
                # episode_data["returns"] = discounted_cum_sum(episode_data["rewards"],
                #                                              discount=discount) * self.return_scale
                traj.append(episode_data)
                traj_len.append(i + 1 - traj_start)
                traj_start = i + 1
        self.traj_len = np.array(traj_len)
        self.size = self.traj_len.sum()
        self.traj_num = len(self.traj_len)
        self.sample_prob = self.traj_len / self.size

        # pad trajs to have the same mask len
        self.max_len = self.traj_len.max() + self.seq_len - 1  # this is for the convenience of sampling
        #self.max_len = max_len
        for i_traj in range(self.traj_num):
            this_len = self.traj_len[i_traj]
            for _key, _value in traj[i_traj].items():
                traj[i_traj][_key] = pad_along_axis(_value, pad_to=self.max_len)
            traj[i_traj]["masks"] = np.hstack([np.ones(this_len), np.zeros(self.max_len - this_len)])

        # register all entries

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

        self.observations_segmented = np.asarray([t["observations"] for t in traj])
        self.actions_segmented = np.asarray([t["actions"] for t in traj])
        self.rewards_segmented = np.asarray([t["rewards"] for t in traj])
        self.terminals_segmented = np.asarray([t["terminals"] for t in traj])
        #self.returns = np.asarray([t["returns"] for t in traj])
        self.next_observations_segmented = np.asarray([t["next_observations"] for t in traj])
        self.masks_segmented = np.asarray([t["masks"] for t in traj])
        self.timesteps = np.arange(self.max_len)

        self.observations = self.observations_segmented[keep_idx]
        self.actions = self.actions_segmented[keep_idx]
        self.rewards = self.rewards_segmented[keep_idx]
        self.terminals = self.terminals_segmented[keep_idx]
        self.next_observations = self.next_observations_segmented[keep_idx]
        self.masks = self.masks_segmented[keep_idx]

        if discount > 1.0:
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount
            self.use_avg = False

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

    # def __prepare_sample(self, traj_idx, start_idx):
    #     return {
    #         "observations": self.observations[traj_idx, start_idx:start_idx + self.seq_len],
    #         "actions": self.actions[traj_idx, start_idx:start_idx + self.seq_len],
    #         "rewards": self.rewards[traj_idx, start_idx:start_idx + self.seq_len],
    #         "terminals": self.terminals[traj_idx, start_idx:start_idx + self.seq_len],
    #         "next_observations": self.next_observations[traj_idx, start_idx:start_idx + self.seq_len],
    #         "returns": self.returns[traj_idx, start_idx:start_idx + self.seq_len],
    #         "masks": self.masks[traj_idx, start_idx:start_idx + self.seq_len],
    #         "timesteps": self.timesteps[start_idx:start_idx + self.seq_len]
    #     }
    def get_trajectory(self, traj_index: int) -> Dict[str, np.ndarray]:
        return {
            "states": self.observations[traj_index],
            "actions": self.actions[traj_index],
            "rewards": self.rewards[traj_index],
            "returns": self.values_segmented[traj_index],
        }

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        idx, start_idx = self.index_map[index]
        traj = self.get_trajectory(idx)
        return {
            k: v[start_idx: start_idx + self.seq_len] for k, v in traj.items()
        }


def return_reward_range(dataset, max_episode_steps=1000):
    returns, lengths = [], []
    nor_ep_ret, ep_ret, ep_len = 0.0, 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        # nor_ep_ret += float(r)
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns), max(lengths)

def modify_reward(dataset, max_episode_steps=1000):
    # if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
    min_ret, max_ret ,_= return_reward_range(dataset, max_episode_steps)
    normalized_rewards = dataset["rewards"]/(max_ret - min_ret)
    normalized_rewards = 100 * normalized_rewards

    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(normalized_rewards, dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)
    # elif "antmaze" in env_name:
    #     dataset["rewards"] -= 1.0

def discounted_cum_sum(rewards, discount):
    discounted_sum = np.zeros_like(rewards)
    running_sum = 0
    for i in reversed(range(len(rewards))):
        running_sum = rewards[i] + discount * running_sum
        discounted_sum[i] = running_sum
    return discounted_sum


def filter_top_returns(traj, traj_returns, top_percentage=0.2):
    traj_returns = [discounted_cum_sum(t['rewards'], discount=1.0).sum() for t in traj]
    sorted_indices = np.argsort(traj_returns)[::-1]
    cutoff = int(len(traj) * top_percentage)
    filtered_indices = sorted_indices[cutoff:]  # 删除回报最高的前 20%

    filtered_traj = [traj[i] for i in filtered_indices]

    random.shuffle(filtered_traj)
    return filtered_traj


class D4RLTrajectoryBuffer():
    def __init__(
        self, 
        dataset, 
        seq_len: int, 
        discount: float=1.0, 
        return_scale: float=1.0,
        max_len: int = 1000,
        sort_trajs = False,
        env = None,
        top_percentage = 0.01,
        top_BC = False,
        pct_traj = 0.2
    ) -> None:
        converted_dataset = dataset
        min_return, max_return, max_epi_len = return_reward_range(dataset)
        self.reward_scale = (1.0/(max_return - min_return))*max_epi_len
        # self.normalization_factor = max_return - min_return
        # self.max_epi_len = max_epi_len
        # self.max_return = max_return
        # min_returns, max_returns = modify_reward(dataset)
        self.min_return = min_return
        self.max_return = max_return
        traj, traj_len,returns = [], [],[]
        self.seq_len = seq_len
        self.discount = discount
        self.return_scale = return_scale
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                episode_data = {k: v[traj_start:i+1] for k, v in converted_dataset.items()}
                # episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) * self.return_scale
                traj.append(episode_data)
                traj_len.append(i+1-traj_start)
                traj_start = i+1
                returns.append(episode_data['rewards'].sum())

        self.traj_len = np.array(traj_len)
        print(f"Average traj len:{np.mean(traj_len)}")

        traj_returns = [env.get_normalized_score(t['rewards'].sum()) * 100.0 for t in traj]

        mean_return = np.mean(traj_returns)
        max_return = np.max(traj_returns)
        print(f"Mean normalized RT in dataset: {mean_return}")
        print(f"Max normalized RT in dataset: {max_return}")

        if top_BC:
            top_20_percent = int(pct_traj * len(returns))
            sorted_indices = sorted(range(len(returns)), key=lambda x: returns[x], reverse=True)
            top_indices = sorted_indices[:top_20_percent]
            traj = [traj[i] for i in top_indices]
            traj_len = [traj_len[i] for i in top_indices]
            #returns = [returns[i] for i in top_indices]
            self.traj_len = np.array(traj_len)

        if sort_trajs:
            # traj = filter_top_returns(traj, traj_returns, top_percentage=0.02)
            traj = filter_top_returns(traj, traj_returns, top_percentage=top_percentage)
            self.traj_len = np.array([len(t['rewards']) for t in traj])
            traj_returns = [env.get_normalized_score(t['rewards'].sum()) * 100.0 for t in traj]
            mean_return = np.mean(traj_returns)
            max_return = np.max(traj_returns)
            #print("traj_returns",traj_returns)
            print(f"Mean Return in dataset: {mean_return}")
            print(f"Max Return in dataset: {max_return}")

        self.size = self.traj_len.sum()
        self.traj_num = len(self.traj_len)
        self.sample_prob = self.traj_len / self.size
        print("Total training set samples:", self.size)
        # print("debug",self.traj_num, self.size)

        # pad trajs to have the same mask len
        #self.max_len = self.traj_len.max() + self.seq_len - 1  # this is for the convenience of sampling
        self.max_len = max_len
        for i_traj in range(self.traj_num):
            this_len = self.traj_len[i_traj]
            for _key, _value in traj[i_traj].items():
                traj[i_traj][_key] = pad_along_axis(_value, pad_to=self.max_len)
            traj[i_traj]["masks"] = np.hstack([np.ones(this_len), np.zeros(self.max_len-this_len)])

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


        # register all entries
        self.observations_segmented = np.asarray([t["observations"] for t in traj])
        self.actions_segmented = np.asarray([t["actions"] for t in traj])
        self.rewards_segmented = np.asarray([t["rewards"] for t in traj])
        self.terminals_segmented = np.asarray([t["terminals"] for t in traj])
        self.next_observations_segmented = np.asarray([t["next_observations"] for t in traj])
        self.masks_segmented = np.asarray([t["masks"] for t in traj])
        self.timesteps = np.arange(self.max_len)

        self.observations = self.observations_segmented[keep_idx]
        self.actions = self.actions_segmented[keep_idx]
        self.rewards = self.rewards_segmented[keep_idx]
        self.terminals = self.terminals_segmented[keep_idx]
        self.next_observations = self.next_observations_segmented[keep_idx]
        self.masks = self.masks_segmented[keep_idx]

        if discount > 1.0:
            self.discount = 1.0
            self.use_avg = True
        else:
            self.discount = discount
            self.use_avg = False

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
            "timesteps": self.timesteps,
            # "Qtrue": self.Qtarget_segmented[traj_index]
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
        
        