from collections import OrderedDict
import numpy as np
from torch import nn
import torch
from research.algo.utils import PiecewiseSchedule
from research.models.critic import DoubleCritic
from copy import deepcopy
from collections import defaultdict
from typing import Any, Callable, Dict, Sequence, Tuple
from torch.utils.data import DataLoader
from operator import itemgetter
import os
from research.algo.planning import cem_planner, pytorch_cov
from research.models.rnd_model import RNDModel
import tqdm
from research.algo.utils import Trajectory
device = "cuda"

def normalize(value, mean, std):
    if std == 0:
        return 0
    return (value - mean) / std

class ExplorationAgent():
    def __init__(self, env, agent_params, logger, model, exploration_critic,
                 online_replay_buffer, online_trajectory_buffer,
                 tokenizer_manager,exploration_model ,normalize_rnd=True, rnd_gamma=0.99):
        # super(ExplorationAgent, self).__init__(env, agent_params)
        self.env = env
        self.logger = logger
        self.agent_params = agent_params
        # self.offline_replay_buffer = offline_replay_buffer
        self.online_replay_buffer = online_replay_buffer
        self.online_trajectory_buffer = online_trajectory_buffer
        self.N = agent_params.N
        self.top_k = agent_params.top_k
        self.planner = cem_planner(model, agent_params.state_dim, agent_params.action_dim,
                          tokenizer_manager=tokenizer_manager, horizon=1,
                          num_samples=self.N, num_elite=self.top_k, device=device)
        # agent_params.obs_shape = np.prod(env.observation_space.shape)
        # agent_params.action_shape = np.prod(env.action_space.shape)

        ################## pretrain rnd #####################
        # self.train_buffer = train_buffer
        # self.train_sampler = torch.utils.data.RandomSampler(self.train_buffer)
        # self.offline_train_loader = DataLoader(
        #     self.train_buffer, batch_size=1, pin_memory=agent_params.pin_memory,
        #     num_workers=2, sampler=self.train_sampler )
        ################## pretrain rnd #####################
        # self.critic_q_target = deepcopy(self.critic_q)
        # self.critic_q_optim = torch.optim.Adam(self.critic_q.parameters(), lr=agent_params.critic_lr)
        # self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=agent_params.critic_lr)


        self.exploration_critic = exploration_critic
        self.exploration_model = exploration_model
        # self.exploration_model = RNDModel(agent_params.obs_shape)

        self.bc_sampler = lambda o, t: self.action_sampler_with_cem(o, t,
        model, tokenizer_manager, agent_params.state_dim,
        agent_params.action_dim, device, critic=self.exploration_critic)

        self.exploit_weight_schedule = PiecewiseSchedule([(0, 1.0), (agent_params.rnd_pretrain_steps/2, 0.5)], outside_value=0.5)
        #self.exploit_weight_schedule = 0.0

        self.learning_starts = agent_params.learning_starts
        self.learning_freq = agent_params.learning_freq

        # self.actor = ArgMaxPolicy(self.exploration_critic)
        # self.eval_policy = ArgMaxPolicy(self.exploitation_critic)
        # self.exploit_rew_shift = agent_params['exploit_rew_shift']
        # self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = 0.2
        self.t = 0

        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma

    def pretrain_iql_wich_rnd(self,rnd_pretrain_steps):
        # for i_step in range(1, 200000+ 1):
        # epoch = 0
        # train_sampler = torch.utils.data.RandomSampler(self.online_trajectory_buffer)
        # offline_train_loader = DataLoader(
        #     self.online_trajectory_buffer, batch_size=1, pin_memory=self.agent_params.pin_memory,
        #     num_workers=2, sampler=train_sampler)
        # offline_buffer_iter = iter(offline_train_loader)

        for i_step in range(1, rnd_pretrain_steps + 1):

            rnd_metrics = {}
            batch = self.online_replay_buffer.sample(self.agent_params.iql_batch_size)
            batch = [b.to(device) for b in batch]
            next_obss = batch[3]
            with torch.no_grad():
                expl_bonus = self.exploration_model.forward_np(next_obss)

            expl_bonus = normalize(expl_bonus, expl_bonus.mean(), self.running_rnd_rew_std)
            self.running_rnd_rew_std = self.rnd_gamma * self.running_rnd_rew_std + (
                    1 - self.rnd_gamma) * expl_bonus.std()
            exploit_weight = self.exploit_weight_schedule.value(i_step)

            [states, actions, rewards, next_states, dones] = batch
            mixed_rewards = (1 - exploit_weight) * expl_bonus + exploit_weight * rewards.squeeze(1)
            critic_batch = [states, actions, mixed_rewards, next_states, dones]
            iql_dict = self.exploration_critic.train(critic_batch)
            rnd_dict = self.exploration_model.update(next_states)
            rnd_metrics.update(rnd_dict)
            rnd_metrics.update(iql_dict)

            if i_step % 1000 == 0:
                print("=========current rnd steps:", i_step)
                print("========exploit_weight:", exploit_weight)
                self.logger.log_scalars("rnd_pretrain", rnd_metrics, step=i_step)
        # if self.agent_params.rnd_save_path is not None:
        #     save_path = self.agent_params.rnd_save_path
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     torch.save(self.exploration_model.f_hat.state_dict(), os.path.join(save_path, "f_hat.pt"))
        #     torch.save(self.exploration_model.f.state_dict(), os.path.join(save_path, "f.pt"))

    @torch.inference_mode()
    def action_sampler_with_cem(self,
                                   observation: np.ndarray,
                                   traj, model, tokenizer_manager, observation_shape,
                                   action_shape, device,critic=None, cem_iterations=1 ):
        N = self.N
        percentage = 1.0
        percentages = np.random.uniform(0.5, 1.5, N)
        traj_len = model.max_len
        observations = np.zeros((traj_len, observation_shape))
        actions = np.zeros((traj_len, action_shape))
        return_max = tokenizer_manager.tokenizers["returns"].stats.max
        return_min = tokenizer_manager.tokenizers["returns"].stats.min
        # print("return_max_min",return_max,return_min)[4.378059] [-0.01247974]
        return_value = return_min + (return_max - return_min) * percentages
        ones_array = np.ones((len(return_value), traj_len, 1))
        returns = return_value[:, np.newaxis, np.newaxis] * ones_array
        # print("rewards", rewards.shape,rewards)(2, 4, 1)

        masks = np.zeros(traj_len)
        i = -1
        max_len = min(traj_len - 1, len(traj))
        assert max_len < traj_len
        for i in range(max_len):
            observations[i] = traj.observations[-max_len + i]
            actions[i] = traj.actions[-max_len + i]
            masks[i] = 1
        assert i == max_len - 1
        observations[i + 1] = observation
        obs_mask = np.copy(masks)
        obs_mask[i + 1] = 1

        trajectories = {
            "states": observations,
            "actions": actions,
        }
        return_mask = np.ones(traj_len)
        # rewward_mask = np.ones(traj_len)
        masks = {
            "states": obs_mask,
            "actions": masks,
            "returns": return_mask
            # "rewards": rewward_mask  # keep the same to maintain structure, even if not used directly
        }

        torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}
        # extract_action sequence make new torch sequence with 1024 copies
        torch_trajectories = {
            k: torch.tensor(v, device=device)[None].repeat(N, 1, 1)
            for k, v in trajectories.items()
        }
        torch_trajectories["returns"] = torch.tensor(returns, device=device)

        encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
        predicted, value_outs = model(encoded_trajectories, torch_masks)
        decode = tokenizer_manager.decode(predicted)

        obs_tiled = np.tile(observation, (N, 1))
        obs_tiled = torch.tensor(obs_tiled, device=device)
        action_tiled = decode["actions"][:, -1, :].clone()

        ###############add extrinsic critic##################
        # critic_out = critic.critic(obs_tiled, action_tiled)
        self.planner.mean = torch.mean(action_tiled, dim=0)
        self.planner.cov = pytorch_cov(action_tiled, rowvar=False)
        for m in range(cem_iterations):
            top_samples = self.planner._sample_top_trajectories_explore(obs_tiled, critic)

            self.planner.mean = torch.mean(top_samples, dim=0)
            self.planner.cov = pytorch_cov(top_samples, rowvar=False)
            if torch.linalg.matrix_rank(self.planner.cov) < self.planner.cov.shape[0]:
                self.planner.cov += self.planner.cov_reg

        top_samples= self.planner._sample_top_trajectories_explore(obs_tiled, critic)
        a = top_samples[0].cpu().numpy()
        return a


    def train_steps(self,
            env,
            num_episodes: int,
            disable_tqdm: bool = True,
            all_results: bool = False,
            max_len: int = 1000,
    ) -> Dict[str, Any]:

        stats: Dict[str, Any] = defaultdict(list)
        #max_steps = env._max_episode_steps
        observation_space = self.agent_params.state_dim
        action_space =self.agent_params.action_dim
        all_observations = np.zeros([num_episodes, max_len+1, observation_space], dtype=np.float32)
        all_actions = np.zeros([num_episodes, max_len, action_space], dtype=np.float32)
        all_rewards = np.zeros([num_episodes, max_len, 1], dtype=np.float32)
        all_terminals = np.zeros([num_episodes, max_len, 1], dtype=np.float32)
        # all_returns = np.zeros([num_episodes, max_len, 1], dtype=np.float32)
        # all_agent_advs = np.zeros([num_episodes, max_len, 1], dtype=np.float32)
        all_masks = np.zeros([num_episodes, max_len], dtype=np.float32)

        pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)
        for i in pbar:
            if(self.t > 2000000):
                break
            print("================== online episode:", i )
            observation, done = env.reset(), False
            trajectory_history = Trajectory.create_empty((observation_space,), (action_space,))

            # num_samples = 0
            # mask = np.ones([num_episodes, ], dtype=bool)
            mask = True
            all_observations[i, 0, :] = observation
            all_masks[i, 0] = True
            # while not done:
            for step in range(max_len):
                rnd_metrics = {}
                self.t += 1
                # num_samples += mask.sum()

                # perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts
                # if perform_random_action:
                #     action = self.env.action_space.sample()
                # else:
                #     action = self.bc_sampler(observation, trajectory_history)
                action = self.bc_sampler(observation, trajectory_history)
                action = np.clip(action, -1, 1)

                new_observation, reward, done, info = env.step(action)
                # next_state, reward, done, _ = env.step(action)
                trajectory_history = trajectory_history.append(observation, action, reward)
                mask = mask & (~done)

                # IndexError: index 1000 is out of bounds for axis 1 with size 1000
                all_observations[i, step + 1, :] = new_observation
                all_actions[i, step, :] = action
                all_rewards[i, step, :] = reward
                all_terminals[i, step, :] = done
                all_masks[i, step] = mask
                observation = new_observation.copy()

                real_done = False  # Episode can timeout which is different from done
                if done and step < max_len:
                    real_done = True

                self.online_replay_buffer.add_transition(observation, action, reward, new_observation, real_done)

                if (self.t > self.learning_starts):
                        # and self.t % self.learning_freq == 0):
                    # explore_weight = self.explore_weight_schedule.value(self.t)
                    # exploit_weight = self.exploit_weight_schedule
                    explore_weight = 0.6
                    exploit_weight = 0.4

                    batch = self.online_replay_buffer.sample(self.agent_params.iql_batch_size)
                    batch = [b.to(device) for b in batch]
                    next_obss = batch[3]
                    with torch.no_grad():
                        expl_bonus = self.exploration_model.forward_np(next_obss)
                    expl_bonus = normalize(expl_bonus, expl_bonus.mean(), self.running_rnd_rew_std)
                    self.running_rnd_rew_std = self.rnd_gamma * self.running_rnd_rew_std + (
                            1 - self.rnd_gamma) * expl_bonus.std()
                    # mixed_rewards = expl_bonus

                    # Update Critics And Exploration Model
                    [states, actions, rewards, next_states, dones] = batch
                    mixed_rewards = explore_weight * expl_bonus + exploit_weight * rewards.squeeze(1)
                    critic_batch = [states, actions, mixed_rewards, next_states, dones]
                    iql_dict = self.exploration_critic.train(critic_batch)

                    #if(self.t % self.learning_freq == 0):
                    rnd_dict = self.exploration_model.update(next_states)
                    rnd_metrics.update(rnd_dict)

                    rnd_metrics.update(iql_dict)
                    if self.t % self.agent_params.rnd_log_every == 0:
                        self.logger.info(f"Step {self.t}: \n{rnd_metrics}")
                        self.logger.log_scalars("rnd_rollout_loop", rnd_metrics, step=self.t)
                ###############update models#####################

                if not mask:
                    # if not mask.any():
                    break

            if "episode" in info:
                stats["return"].append(info['episode']['return'])
                stats["length"].append(info['episode']['length'])
            else:
                stats["return"].append(trajectory_history.rewards.sum())
                stats["length"].append(len(trajectory_history.rewards))
                # stats["achieved"].append(int(info["goal_achieved"]))

        new_data = {
            "observations": all_observations[:, :-1],
            "actions": all_actions,
            "next_observations": all_observations[:, 1:],
            "rewards": all_rewards,
            "terminals": all_terminals,
            "masks": all_masks,
            # "values":all_returns
        }
        self.online_trajectory_buffer.add_online_traj(new_data)

        new_stats = {}
        for k, v in stats.items():
            new_stats[k + "_mean"] = float(np.mean(v))
            new_stats[k + "_max"] = float(np.max(v))
            new_stats[k + "_std"] = float(np.std(v))
        self.logger.log_scalars("data_quality", new_stats, step=self.t)

        return new_stats







