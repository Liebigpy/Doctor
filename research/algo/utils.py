import os
from typing import Optional,Dict,Tuple
import numpy as np
from typing import Sequence
import torch
import random
import gym
from dataclasses import dataclass


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def expectile_regression(pred, target, expectile):
    diff = target - pred
    return torch.where(diff > 0, expectile, 1-expectile) * (diff**2)

@dataclass(frozen=True)
class Trajectory:
    """Immutable container for a Trajectory.

    Each has shape (T, X).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray

    def __post_init__(self):
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.observations.shape[0] == self.rewards.shape[0]

    def __len__(self) -> int:
        return self.observations.shape[0]

    @staticmethod
    def create_empty(
        observation_shape: Sequence[int], action_shape: Sequence[int]
    ) -> "Trajectory":
        """Create an empty trajectory."""
        return Trajectory(
            observations=np.zeros((0,) + observation_shape),
            actions=np.zeros((0,) + action_shape),
            rewards=np.zeros((0, 1)),
            returns=np.zeros((0, 1))
        )

    def append(
        self, observation: np.ndarray, action: np.ndarray, reward: float, value: float,
    ) -> "Trajectory":
        """Append a new observation, action, and reward to the trajectory."""
        assert observation.shape == self.observations.shape[1:]
        assert action.shape == self.actions.shape[1:]
        return Trajectory(
            observations=np.concatenate((self.observations, observation[None])),
            actions=np.concatenate((self.actions, action[None])),
            rewards=np.concatenate((self.rewards, np.array([reward])[None])),
            returns=np.concatenate((self.returns, np.array([value])[None]))
        )


def get_ckpt_path_from_folder(folder) -> Optional[str]:
    steps = []
    names = []
    paths_ = os.listdir(folder)
    for name in [os.path.join(folder, n) for n in paths_ if "pt" and "doctor" in n]:
        step = os.path.basename(name).split("_")[-1].split(".")[0]
        steps.append(step)
        names.append(name)
    # print("steps,names############",steps,names)
    if len(steps) == 0:
        return None
    else:
        ckpt_path = names[np.argmax(steps)]
        return ckpt_path


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def return_reward_range(dataset, max_episode_steps):
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



# @staticmethod
def configure_optimizers(
        model, learning_rate: float, weight_decay: float, betas: Tuple[float, float]
):
    decay = set()
    no_decay = set()
    allowlist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
    blocklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # print("pn",pn)
            if "critic_q" in fpn or "critic_v" in fpn or "critic_q_target" in fpn:
                continue
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, allowlist_weight_modules):
                # weights of allowed modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blocklist_weight_modules):
                # weights of blocked modules will NOT be weight decayed
                no_decay.add(fpn)

    for pn, _ in model.named_parameters():
        # print("pn2", pn)
        if "critic_q" in pn or "critic_v" in pn or "critic_q_target" in pn:
            continue
        if "dict" in pn and "bias" in pn:
            no_decay.add(pn)
        if "per_dim_encoding" in pn or "mask_token_dict" in pn:
            no_decay.add(pn)


    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if "critic_q" not in pn and "critic_v" not in pn and "critic_q_target" not in pn}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
            len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
            len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    m_optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return m_optimizer

