import torch
import gym
import tqdm
import wandb
import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, Sequence, Tuple

from research.tokenizers.base import TokenizerManager
from research.algo.sampler_v4 import *
from research.algo.utils import Trajectory
from research.algo.planning import cem_planner,pytorch_cov
from research.algo.utils import set_seed_everywhere

def compute_Q_true(episode_rewards, gamma):
    Q_true = []
    T = len(episode_rewards)
    for t in range(T):
        q_true_t = 0.0
        for k in range(t, T):
            q_true_t += episode_rewards[k] * (gamma ** (k - t))
        Q_true.append(q_true_t)
    return Q_true

def compute_V_diff(episode_Qs, Q_true):
    episode_Qs = np.array(episode_Qs)
    Q_true = np.array(Q_true)
    #diff = torch.tensor(episode_Qs) - torch.tensor(Q_true)
    diff = episode_Qs - Q_true
    V_diff = np.mean(diff)
    #V_diff = torch.mean(diff).item()
    return V_diff

def modify_reward_online(reward ,norm_record, ant_env = False ):
    if not ant_env:
        reward *= norm_record
    # elif "antmaze" in env_name:
    #     reward -= 1.0
    return reward

def augment_trajectories(
        model: Callable,
        tokenizer_manager: TokenizerManager,
        observation_shape,
        action_shape,
        env,
        eval_num = 10,
        argmin_alignment = False
) -> Dict[str, Any]:
    reward_scale = model.reward_scale
    nor_Max_return = model.Max_return
    log_data = {}
    device = next(model.parameters()).device # print("debug",device)cuda:0

    if "returns" in tokenizer_manager.tokenizers:
        for p in [0.6]:
            nor_Max_return = nor_Max_return * p
            bc_sampler = lambda o, t: sample_action_with_double_check(
                o, t, model, tokenizer_manager,observation_shape, action_shape, device,
                # N=300, percentage=p, eps=3.0
                percentage = p,
                eps=3.0,
                N = 300,
                argmin_alignment = argmin_alignment)

            results = evaluate_rollouts(bc_sampler, env,
                eval_num, observation_shape, action_shape, reward_scale, nor_Max_return)

            for k, v in results.items():
                log_data[f"eval_double_check/{k}"] = v

    # if "returns" in tokenizer_manager.tokenizers:
    #     for p in [1.0]:
    #         bc_sampler = lambda o, t: sample_action_with_bc(o,  t, model,
    #             tokenizer_manager, observation_shape, action_shape, device,
    #             percentage = p)
    #
    #         results = evaluate_rollouts(bc_sampler, env, eval_num,
    #             observation_shape, action_shape, reward_scale,nor_Max_return,)
    #
    #         for k, v in results.items():
    #             log_data[f"eval_bc/p={p}_{k}"] = v
    return log_data



def evaluate_rollouts(
        sample_actions,
        env: gym.Env,
        num_episodes: int,
        observation_space: int,
        action_space: int,
        reward_scale: float ,
        Max_return: float,
        disable_tqdm: bool = True,
        all_results: bool = False,
        max_len: int = 1000,
        gamma = 0.99,
) -> Dict[str, Any]:

    stats: Dict[str, Any] = defaultdict(list)
    value_diff = 0

    pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)
    traj_ls = []
    Qs = []
    RTs = []

    for i in pbar:
        current_return = Max_return * reward_scale

        episode_return = 0.0
        episode_rewards = []
        episode_Qs = []
        episode_length = 0
        observation, done = env.reset(), False
        trajectory_history = Trajectory.create_empty((observation_space,), (action_space,))

        mask = True
        for step in range(max_len):
            episode_length += 1
            # num_samples += mask.sum()
            action , Q = sample_actions(observation, trajectory_history)

            action = np.clip(action, -1, 1)

            new_observation, reward, done, info = env.step(action)
            nor_reward = modify_reward_online(reward, reward_scale)
            # episode_rewards.append(nor_reward)
            episode_Qs.append(Q)
            episode_return += reward

            trajectory_history = trajectory_history.append(observation, action, nor_reward, current_return)
            # (R_t - r_t) / gamma
            current_return = current_return - nor_reward
            #print("debug",current_return,nor_reward ,reward)
            mask = mask & (~done)

            observation = new_observation
            if not mask:
            #if not mask.any():
                break

        #print("one trajectory acquired by interaction", info) 'episode': {'return': 3.3054504348889715, 'length': 116, 'duration': 0.6932356357574463}}
        RTs.append(episode_return)
        traj_ls.append(episode_length)
        Qs.append(episode_Qs)

        if "episode" in info:
            stats["return"].append(info['episode']['return'])
            stats["length"].append(info['episode']['length'])
        else:
            stats["return"].append(trajectory_history.rewards.sum())
            stats["length"].append(len(trajectory_history.rewards))

    eval_scores = np.asarray(RTs)
    normalized = env.get_normalized_score(eval_scores.mean()) * 100.0
    print("normalized score: ", normalized)
    print("original score: ", eval_scores.mean())
    # print("num eval frame we got:", traj_ls[0])
    new_stats = {}
    # for k, v in stats.items():
    #     new_stats[k + "_mean"] = float(np.mean(v))
    #     new_stats[k + "_std"] = float(np.std(v))
    new_stats["normalized_RT"] = normalized
    # new_stats["V_diff"] = value_diff
    if all_results:
        new_stats.update(stats)
    stats = new_stats

    return stats






