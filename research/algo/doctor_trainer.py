import os
import random
import tqdm
import time
from tqdm import trange
from copy import deepcopy
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from research.algo.evaluation import *
from research.algo.rollout_v4 import *
from research.algo.sampler_v4 import *
from research.algo.utils import Trajectory
from research.algo.planning import cem_planner, pytorch_cov

EXP_ADV_MAX = 100.0

def modify_reward_online(reward, norm_record, ant_env = False):
    if not ant_env:
        reward *= norm_record
    # elif "antmaze" in args.task:
    #     reward -= 1.0
    return reward

def normalize(value, mean, std):
    if std == 0:
        return 0
    return (value - mean) / std

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    loss = torch.abs(tau - (u < 0).float()) * u ** 2
    return loss

def expectile_regression(pred, target, expectile):
    diff = target - pred
    return torch.where(diff > 0, expectile, 1-expectile) * (diff**2)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Doc_Trainer:
    def __init__(
            self,
            args, model, mask_functions, discrete_map, logger, eval_masks,
            offline_optimizer=None,
            offline_scheduler=None,
            eval_max=None,
            offline_buffer=None,
            val_loader=None,
            tokenizer_manager=None,
            env=None,
            rnd_model = None,
            normalize_rnd = True,
            rnd_gamma = 0.99,
            eval_num = 10,
            ema_decay=0.995,
            update_ema_every = 2,
            step_start_ema = 0,
            eta = 1.0
    ):
        self.args = args
        self.obs_shape = self.args.obs_shape
        self.action_shape = self.args.action_shape
        self.eval_num = eval_num
        self.mask_functions = mask_functions
        # self.ema = EMA(ema_decay)
        self.model = model
        # self.ema_model = deepcopy(self.model)
        # self.update_ema_every = update_ema_every
        # self.step_start_ema = step_start_ema
        # self.eta = eta

        self.offline_optimizer = offline_optimizer
        self.offline_scheduler = offline_scheduler

        self.discrete_map = discrete_map
        self._discount = 0.99
        self.logger = logger
        # self.eval_max = {}
        self.eval_masks = eval_masks
        self.offline_buffer = offline_buffer
        self.val_loader = val_loader
        self.tokenizer_manager = tokenizer_manager
        self.env = env
        self._iql_tau = args.iql_tau
        self._tau = args.tau #_tau = 5e-3 / 0.001
        self.device = "cuda"
        self.eval_max = eval_max
        # self.grad_norm = args.grad_norm
        self.beta = args.beta
        self.critic_q_optim = torch.optim.Adam(self.model.critic_q.parameters(), lr=args.critic_lr)
        self.critic_v_optim = torch.optim.Adam(self.model.critic_v.parameters(), lr=args.critic_lr)

        self.critic_q_lr_scheduler = CosineAnnealingLR(self.critic_q_optim, T_max=100, eta_min=0.)
        self.critic_v_lr_scheduler = CosineAnnealingLR(self.critic_v_optim, T_max=100, eta_min=0.)

        self.planner = cem_planner(model, args.obs_shape, args.action_shape,
                                   tokenizer_manager=tokenizer_manager, horizon=1,
                                   num_samples=100, num_elite=10, device=self.device)

        self.reward_scale = args.reward_scale

    def step_ema(self):
        if self._tt > self.step_start_ema and self._tt % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.model)

    def train_all_one_batch(
            self,
            model: MTM,
            optimizer: torch.optim.Optimizer,
            scheduler: Callable,
            tokenizer_manager: TokenizerManager,
            discrete_map: Dict[str, bool],
            batch: Dict[str, torch.Tensor],
            masks: Dict[str, torch.Tensor],
            loss_keys: Sequence[str] = None,
            discounted_rewards=True,
    ) -> Dict[str, Any]:
        log_dict = {}
        states, actions, rewards, next_states, terminals, mask, returns = itemgetter(
            "observations", "actions", "rewards", "next_observations", "terminals", "masks", "returns"
        )(batch)
        B = states.shape[0]
        T = states.shape[1]
        device = states.device

        doc_batch = {
            "states": states,
            "actions": actions,
            "returns": returns,
        }
        encoded_batch = tokenizer_manager.encode(doc_batch)
        predicted_trajectories, values_predicted = model(encoded_batch, masks)
        #predicted_trajectories_target, values_predicted_target = self.ema_model(encoded_batch, masks)

        rewards = modify_reward_online(rewards, self.reward_scale)

        v_out, [], q_out, q_out_old = values_predicted
        #v_out_target, [], q_out_target, q_out_old_target = values_predicted_target

        adv = q_out_old - v_out
        v_loss = asymmetric_l2_loss(adv, self._iql_tau)
        v_loss = (v_loss * mask).mean(0).mean()
        # v_loss = (v_loss * mask).mean()
        self.critic_v_optim.zero_grad()
        v_loss.backward()
        self.critic_v_optim.step()

        if discounted_rewards:
            # v_out_target
            v_next = v_out[:, -1].unsqueeze(-1)  # [B, 1]
            not_done = (1 - terminals[:, -1])  # [B, 1]
            if True:
                rewards[:, -1] = 0.
                mask_ = mask.sum(dim=1).detach().cpu()  # [B]
                discount = [i - 1 - torch.arange(i) for i in mask_]
                discount = torch.stack([torch.cat([i, torch.zeros(T - len(i))], dim=0) for i in discount],
                                       dim=0)  # [B, T]
                discount = (self._discount ** discount).unsqueeze(-1).to(device)  # [B, T, 1]
                k_rewards = torch.cumsum(rewards.flip(dims=[1]) * discount, dim=1).flip(dims=[1])  # [B, T, 1]

                discount = [torch.arange(i) for i in mask_]  #
                discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
                discount = (self._discount ** discount).unsqueeze(-1).to(device)
                k_rewards = k_rewards / discount

                discount = [i - 1 - torch.arange(i) for i in mask_]  # [B]
                discount = torch.stack([torch.cat([torch.zeros(T - len(i)), i], dim=0) for i in discount], dim=0)
                discount = (self._discount ** discount).to(device)  # [B, T]
                targets = (k_rewards + (not_done * discount * v_next).unsqueeze(-1)).detach()  # [B, T, 1]
        else:
            targets = (rewards[:, :-1] + (1.0 - terminals[:, 1:].float()) * self._discount * v_out[:, 1:].unsqueeze(
                -1)).detach()

        q_loss = sum(
            F.mse_loss(q[:, :-1][mask[:, :-1] > 0].unsqueeze(-1), targets[:, :-1][mask[:, :-1] > 0])
            for q in q_out) / len(q_out)

        self.critic_q_optim.zero_grad()
        q_loss.backward()
        self.critic_q_optim.step()

        soft_update(self.model.critic_q_target, self.model.critic_q, self._tau)

        log_dict["train/v_loss"] = v_loss.item()
        log_dict["train/q_loss"] = q_loss.item()
        log_dict["train/v_pred"] = v_out.mean().item()
        log_dict["train/q_pred"] = q_out[0].mean().item()

        _, losses_dict = model.forward_loss(encoded_batch, predicted_trajectories,
                                            masks, discrete_map, loss_keys=loss_keys)
        # print("debug",predicted_trajectories["actions"].shape,encoded_batch["actions"].shape)
        # action_losses = torch.sum((predicted_trajectories["actions"] - encoded_batch["actions"]) ** 2, dim=3).squeeze(2)
        action_losses = F.mse_loss(predicted_trajectories["actions"], encoded_batch["actions"], reduction='none')
        # print("debug",action_losses.shape)
        action_losses = action_losses.mean(dim=(2, 3))
        # exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        # losses_dict["actions"] = torch.mean(exp_adv * action_losses)
        losses_dict["actions"] = action_losses.mean()

        loss = torch.sum(torch.stack([losses_dict[key] for key in ["states", "actions", "returns"]]))
        log_dict["train/SL_loss"] = loss.item()
        log_dict["train/lr"] = scheduler.get_last_lr()[0]
        for k, v in losses_dict.items():
            log_dict[f"train/loss_{k}"] = v.item()

        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # self.step_ema()

        with torch.no_grad():
            mse_loss = 0
            predictions = tokenizer_manager.decode(predicted_trajectories)
            for k, v in predictions.items():
                _mse = F.mse_loss(v.to(torch.float32), doc_batch[k].to(torch.float32)).item()
                log_dict[f"train/mse_{k}"] = _mse
                mse_loss += _mse

        return log_dict


    def pretrain(self):

        if self.args.model_save_path is not None:
            save_path = os.path.join(self.args.model_save_path, self.args.task, "seed" + str(self.args.seed))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        best_ret = -10000
        best_nor_ret = -1000
        best_iter = -1

        step = 0
        self._tt = 0
        epoch = 0
        self.offline_train_sampler = torch.utils.data.RandomSampler(self.offline_buffer)

        train_loader = DataLoader(
            self.offline_buffer, batch_size=self.args.train_batch_size, pin_memory=self.args.pin_memory,
            num_workers=self.args.num_workers, sampler=self.offline_train_sampler)


        buffer_iter = iter(train_loader)
        for step in trange(1, self.args.num_pretrain_steps + 1, desc="pretrain main loop"):
            self._tt += 1
            B = time.time()
            train_metrics = {}
            train_metrics["epochs"] = epoch
            start_time = time.time()

            ##################TRAIN##################
            try:
                batch = next(buffer_iter)
            except StopIteration:
                buffer_iter = iter(train_loader)
                batch = next(buffer_iter)
                epoch += 1

            masks = random.choice(self.mask_functions)()
            #print("debug",masks)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            self.model.train()
            _log_dict = self.train_all_one_batch(
                self.model,
                self.offline_optimizer,
                self.offline_scheduler,
                self.tokenizer_manager,
                self.discrete_map,
                batch,
                masks,
                loss_keys=self.args.loss_keys,
            )
            train_metrics.update(_log_dict)
            train_metrics["training_time"] = time.time() - start_time

            if step % self.args.print_every == 0:
                train_loss = train_metrics["train/SL_loss"]
                self.logger.info(f"Step: {step}, Train Loss: {train_loss}")

            if step % self.args.log_every == 0:
                self.logger.info(f"Step {step}: \n{train_metrics}")
                self.logger.log_scalars("pretraining_loop", train_metrics, step=step)

            ##################EVAL##################
            # if step % self.args.eval_every == 0 and step != 0 and step > self.args.tm_ppretrain_steps:
            if step % self.args.eval_every == 0 and step != 0:
                start_time = time.time()
                self.model.eval()
                val_batch = next(iter(self.val_loader))
                val_batch = {k: v.to(self.device, non_blocking=True) for k, v in val_batch.items()}

                # ================ 1. eval interaction===================== #
                eval_metrics = augment_trajectories(
                    # self.offline_buffer,
                    self.model,
                    self.tokenizer_manager,
                    self.obs_shape,
                    self.action_shape,
                    self.env,
                    eval_num=self.eval_num
                )
                # ================ 2. eval supervised===================== #
                _val_dict = evaluate(
                    self.model,
                    self.tokenizer_manager,
                    self.discrete_map,
                    val_batch,
                    self.eval_masks,
                    loss_keys=["states", "actions", "returns"]
                )
                eval_metrics.update(_val_dict)

                # ======= 3. for everything with eval prefix keep the max====== #
                max_log = {}
                for k, v in eval_metrics.items():
                    # print("debug",k)
                    if k.startswith("eval_double_check/normalized_RT"):
                        self.eval_max[k] = max(self.eval_max[k], v)
                        max_log[f"max_{k}"] = self.eval_max[k]
                # print("max_log",max_log)
                eval_metrics.update(max_log)

                eval_metrics["time/eval_time"] = time.time() - start_time
                val_loss = eval_metrics["val/total_loss"]
                eval_return = eval_metrics["eval_double_check/normalized_RT"]
                self.logger.info(f"Step: {step}, Val Loss: {val_loss}")
                self.logger.info(f"Step: {step}, Eval return: {eval_return}")
                self.logger.log_scalars("testing", eval_metrics, step=step)

                if eval_return > best_ret and self.args.model_save_path is not None:
                    best_ret = eval_return
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self.offline_optimizer.state_dict(),
                            "step": self._tt,
                            "eval_max": dict(self.eval_max),
                        },
                        os.path.join(save_path, f"pretrain_doctor_model_{step}.pt")
                    )
                    self.alignment_path = (self.args.model_save_path +
                                           self.args.task + "/" + "seed0" + "/" + f"pretrain_doctor_model_{step}.pt")



    def interaction_steps(self,
            num_episodes: int,
            num_epoch: int,
            disable_tqdm: bool = True,
            max_len: int = 1000,
            ant_maze = False,
    ) -> Dict[str, Any]:
        self._it = 0

        if self.doctor_interaction:
            self.bc_sampler = lambda o, t: interaction_sampler_with_value_diff(o, t,
                                                                    self.model, self.tokenizer_manager,
                                                                    self.args.obs_shape,self.args.action_shape, self.device,
                                                                   )
        else:
            self.bc_sampler = lambda o, t: interaction_sampler_with_bc(o, t,
                                                                    self.model, self.tokenizer_manager,
                                                                    self.args.obs_shape,
                                                                    self.args.action_shape, self.device,
                                                                 )

        stats: Dict[str, Any] = defaultdict(list)
        observation_space = self.args.obs_shape
        action_space =self.args.action_shape
        # env.seed(s)
        # self.env.seed(seed)
        RTs = []
        pbar = tqdm.tqdm(range(num_episodes), disable=disable_tqdm, ncols=85)
        for i in pbar:
            current_return = 0
            episode_return = 0.0

            all_observations = np.zeros([1, max_len + 1, observation_space], dtype=np.float32)
            all_actions = np.zeros([1, max_len, action_space], dtype=np.float32)
            all_rewards = np.zeros([1, max_len, 1], dtype=np.float32)
            all_terminals = np.zeros([1, max_len, 1], dtype=np.float32)
            all_masks = np.zeros([1, max_len], dtype=np.float32)

            observation, done = self.env.reset(), False
            trajectory_history = Trajectory.create_empty((observation_space,), (action_space,))
            mask = True
            all_observations[0, 0, :] = observation
            all_masks[0, 0] = True

            for step in range(max_len):
                self._it += 1
                # num_samples += mask.sum()
                action = self.bc_sampler(observation, trajectory_history)
                action = np.clip(action, -1, 1)

                new_observation, reward, done, info = self.env.step(action)
                if ant_maze:
                    reward -= 1.0
                episode_return += reward

                trajectory_history = trajectory_history.append(observation, action, reward, current_return )
                mask = mask & (~done)

                all_observations[0, step + 1, :] = new_observation
                all_actions[0, step, :] = action
                all_rewards[0, step, :] = reward
                all_terminals[0, step, :] = done
                all_masks[0, step] = mask
                observation = new_observation.copy()
                if not mask:
                    # if not mask.any():
                    break

            RTs.append(episode_return)

            if "episode" in info:
                stats["return"].append(info['episode']['return'])
                stats["length"].append(info['episode']['length'])
            else:
                stats["return"].append(trajectory_history.rewards.sum())
                stats["length"].append(len(trajectory_history.rewards))

            new_data = {
                "observations": all_observations[:, :-1],
                "actions": all_actions,
                "next_observations": all_observations[:, 1:],
                "rewards": all_rewards,
                "terminals": all_terminals,
                "masks": all_masks,
            }
            self.online_buffer.add_online_traj(new_data)

        new_stats = {}
        eval_scores = np.asarray(RTs)
        normalized = self.env.get_normalized_score(eval_scores.mean()) * 100.0
        print("normalized score: ", normalized)
        # new_stats["epi_return"] = episode_return
        new_stats["normalized"] = normalized
        new_stats["num_samples"] = self.total_samples
        self.logger.log_scalars("episode_return", new_stats, step = num_epoch)
        return new_stats







