
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gym
import numpy as np
from research.env.make_env import get_env
import d4rl


def maze_normalize_reward(dataset):
    # dataset["rewards"] -= 1.0
    # return dataset, {}
    dataset["rewards"] = (dataset["rewards"] - 0.5) * 4
    return dataset, {}


def mujoco_normalize_reward(dataset):
    split_points = dataset["ends"].copy()
    split_points[-1] = False  # the last traj may be incomplete, so we discard them
    reward = dataset["rewards"]
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(reward, split_points):
        ep_ret += float(r)
        ep_len += 1
        if d:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    dataset["rewards"] /= max(returns) - min(returns)
    dataset["rewards"] *= 1000
    return dataset, {}


def d4rl_normalize_obs(dataset):
    all_obs = np.concatenate([dataset["observations"], dataset["next_observations"]], axis=0)
    # all_obs = dataset["observations"]
    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0) + 1e-3
    dataset["observations"] = (dataset["observations"] - obs_mean) / obs_std
    dataset["next_observations"] = (dataset["next_observations"] - obs_mean) / obs_std
    return dataset, {
        "obs_mean": obs_mean,
        "obs_std": obs_std
    }


def qlearning_dataset(env, dataset=None, terminate_on_end: bool = False, discard_last: bool = True, **kwargs):

    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    end_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(
            np.float32)  # Thus, the next_obs for the last timestep is totally false
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        end = 0
        episode_step += 1

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps)
        if final_timestep:
            if not done_bool:
                if not terminate_on_end:
                    if discard_last:
                        episode_step = 0
                        end_[-1] = True
                        continue
                else:
                    done_bool = True
        if final_timestep or done_bool:
            end = 1
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        end_.append(end)

    end_[-1] = True  # the last traj will be ended whatsoever
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        "ends": np.array(end_)
    }

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward

def get_datasets(
    env_name,
    seed: int = 0,
    use_reward: bool = True,
    seq_steps = 4,
    discount: int = 1.5,
    normalize_reward=False,
    normalize_obs=False,
    clip_to_eps: bool = True,
    eps: float = 1e-5,
    train_val_split: float = 0.95,
    ant_env: bool = False
):
    env = get_env(env_name, seed)
    dataset = qlearning_dataset(env)

    rewards = dataset['rewards']
    terminals = dataset['terminals']
    ends = dataset['ends']

    returns = []
    current_return = 0

    for i in range(len(rewards)):
        current_return += rewards[i]
        if ends[i] == 1:
            returns.append(current_return)
            current_return = 0

    num_samples = len(rewards)


    returns = np.array(returns)
    print(f'Number of samples collected: {num_samples}')
    print(
        f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    Max_return = np.max(returns)

    # if ant_env:
    #     reward_mod_dict = modify_reward(dataset, env_name)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    # dones_float = np.zeros_like(dataset["rewards"])
    # for i in range(len(dones_float) - 1):
    #     if (
    #             np.linalg.norm(
    #                 dataset["observations"][i + 1] - dataset["next_observations"][i]
    #             )
    #             > 1e-6
    #             or dataset["terminals"][i] == 1.0
    #     ):
    #         dones_float[i] = 1
    #     else:
    #         dones_float[i] = 0
    #
    # dones_float[-1] = 1

    converted_dataset = {
        "observations": dataset["observations"].astype(np.float32),
        "actions": dataset["actions"].astype(np.float32),
        "rewards": dataset["rewards"][:, None].astype(np.float32),
        "terminals": dataset["terminals"][:, None].astype(np.float32),
        "next_observations": dataset["next_observations"].astype(np.float32),
        # "ends":dataset["ends"][:, None].astype(np.float32),
        "ends": dataset["ends"],
        # "masks" : 1.0 - dataset["terminals"].astype(np.float32),
        # "dones_float" : dones_float.astype(np.float32),
        # "size" : len(dataset["observations"]),
    }

    train_d, val_d = train_validation_split(converted_dataset, train_val_split)


    return train_d, val_d, env



def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    ends = []

    for traj in trajs:
        for obs, act, rew, term, next_obs, done in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            terminals.append(term)
            next_observations.append(next_obs)
            ends.append(done)

    return (
        np.stack(observations),
        np.stack(actions),
        np.stack(rewards),
        np.stack(terminals),
        np.stack(next_observations),
        np.stack(ends),
    )

def split_into_trajectories(converted_dataset):

    trajs = [[]]
    observations = converted_dataset["observations"]
    actions = converted_dataset["actions"]
    rewards = converted_dataset["rewards"]
    terminals = converted_dataset["terminals"]
    next_observations = converted_dataset["next_observations"]
    ends = converted_dataset["ends"]
    # dones_float = converted_dataset["dones_float"]

    for i in range(len(observations)):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                terminals[i],
                next_observations[i],
                ends[i],
            )
        )
        # print("debug3", ends[i] == 1.0)
        if ends[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def train_validation_split(converted_dataset, train_fraction: float = 0.8):
    trajs = split_into_trajectories(converted_dataset)
    train_size = int(train_fraction * len(trajs))

    # np.random.shuffle(trajs)

    (
        train_observations,
        train_actions,
        train_rewards,
        train_terminals,
        train_next_observations,
        train_ends,
    ) = merge_trajectories(trajs[:train_size])

    (
        valid_observations,
        valid_actions,
        valid_rewards,
        valid_terminals,
        valid_next_observations,
        valid_ends,
    ) = merge_trajectories(trajs[train_size:])


    train_dataset = {
        "observations": train_observations,
        "actions": train_actions,
        "rewards": train_rewards,
        "terminals": train_terminals,
        "next_observations": train_next_observations,
        "ends": train_ends,
    }
    valid_dataset = {
        "observations": valid_observations,
        "actions": valid_actions,
        "rewards": valid_rewards,
        "terminals": valid_terminals,
        "next_observations": valid_next_observations,
        "ends": valid_ends,
    }

    return train_dataset, valid_dataset


