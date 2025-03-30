import enum
from enum import Enum, unique
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

BASIC_MODE = True

@unique
class MaskType(Enum):
    RANDOM = enum.auto()
    ID = enum.auto()
    FD = enum.auto()
    GOAL = enum.auto()
    GOAL_N = enum.auto()
    FULL_RANDOM = enum.auto()
    BC = enum.auto()
    RCBC = enum.auto()
    BC_RANDOM = enum.auto()
    AUTO_MASK = enum.auto()

def create_mask(mask_type, data_shapes, cfg, has_rew, has_img, has_ret, device = "cuda"):
    if mask_type == "RANDOM":
        return create_random_masks(data_shapes, cfg.mask_ratios, cfg.seq_len, device)
    elif mask_type == "FULL_RANDOM":
        return create_full_random_masks(data_shapes, cfg.mask_ratios, cfg.seq_len, device)
    elif mask_type == "AUTO_MASK":
        return create_random_autoregressize_mask(data_shapes, cfg.mask_ratios, cfg.seq_len, device)
    elif mask_type == "RCBC":
        return create_rcbc_mask(cfg.seq_len, device)
    elif mask_type == "GOAL":
        return maybe_add_rew_to_mask(cfg.seq_len, device, create_goal_reaching_masks, has_rew, has_img, has_ret)
    elif mask_type == "GOAL_N":
        return maybe_add_rew_to_mask(cfg.seq_len, device, create_goal_n_reaching_masks, has_rew, has_img, has_ret)
    elif mask_type == "ID":
        return maybe_add_rew_to_mask(cfg.seq_len, device, create_inverse_dynamics_mask, has_rew, has_img, has_ret)
    elif mask_type == "FD":
        return maybe_add_rew_to_mask(cfg.seq_len, device, create_forward_dynamics_mask, has_rew, has_img, has_ret)
    elif mask_type == "BC":
        return maybe_add_rew_to_mask(cfg.seq_len, device, create_bc_mask, has_rew, has_img, has_ret)
    elif mask_type == "BC_RANDOM":
        return maybe_add_rew_to_mask(cfg.seq_len, device, lambda l, d: create_random_bc_masks(l, d, data_shapes, p=0.5), has_rew, has_img, has_ret)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")



# tensor([1., 0., 1., 0., 1., 0., 1., 0., 1., 0.])
def create_random_mask(
    traj_length: int,
    mask_ratios: Union[Tuple[float, ...], float],
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    # random_mask = np.concatenate(
    #     [
    #         np.ones(6),
    #         np.zeros(traj_length - 6),
    #     ]
    # )
    # return torch.tensor(random_mask, device=device)

    if isinstance(mask_ratios, Sequence):
        if rnd_state is None:
            mask_ratio = np.random.choice(mask_ratios)
        else:
            mask_ratio = rnd_state.choice(mask_ratios)
    else:
        mask_ratio = mask_ratios

    masked = int(traj_length * mask_ratio)
    random_mask = np.concatenate(
        [
            np.ones(masked),
            np.zeros(traj_length - masked),
        ]
    )
    if rnd_state is None:
        np.random.shuffle(random_mask)
    else:
        rnd_state.shuffle(random_mask)

    # same mask for now
    random_mask = torch.tensor(random_mask, device=device)
    return random_mask


# tensor([[1, 1, 1, 1, 1],
#         [1, 1, 0, 1, 1],
#         [1, 1, 1, 0, 1],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1]])
#（10，5）
def create_full_random_mask(
    data_shape: Tuple[int, int],
    traj_length: int,
    mask_ratios: Union[Tuple[float, ...], float],
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    L = traj_length
    P, _ = data_shape

    if isinstance(mask_ratios, Sequence):
        if rnd_state is None:
            mask_ratio = np.random.choice(mask_ratios)
            #print("debug100ration",mask_ratio)
        else:
            mask_ratio = rnd_state.choice(mask_ratios)
    else:
        mask_ratio = mask_ratios

    masked = int(L * P * float(mask_ratio))
    random_mask = np.concatenate(
        [
            np.ones(masked),
            np.zeros(L * P - masked),
        ]
    )
    if rnd_state is None:
        np.random.shuffle(random_mask)
    else:
        rnd_state.shuffle(random_mask)

    random_mask = torch.tensor(random_mask, device=device)
    return random_mask.reshape(L, P)



def create_goal_reaching_masks(
    traj_length: int,
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    state_mask = np.zeros(traj_length)

    if BASIC_MODE:
        state_mask[0] = 1
        state_mask[-1] = 1
        if rnd_state is None:
            end_state = np.random.randint(0, traj_length)
        else:
            end_state = rnd_state.randint(0, traj_length)
        state_mask[end_state] = 1
    else:
        state_mask[0:3] = 1
        state_mask[-3:] = 1
        if rnd_state is None:
            end_state = np.random.randint(3, traj_length - 2)
        else:
            end_state = rnd_state.randint(3, traj_length - 2)
        state_mask[end_state] = 1

    action_mask = np.zeros(traj_length)
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def create_goal_n_reaching_masks(
    traj_length: int,
    device: str,
    rnd_state: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    state_mask = np.zeros(traj_length)
    action_mask = np.zeros(traj_length)

    if traj_length > 1:
        if rnd_state is None:
            end_state = np.random.randint(1, traj_length)
        else:
            end_state = rnd_state.randint(1, traj_length)

        state_mask[:end_state] = 1
        action_mask[: (end_state - 1)] = 1

        if BASIC_MODE:
            state_mask[-1] = 1
        else:
            if rnd_state is None:
                end_state = np.random.randint(1, 4)
            else:
                end_state = np.random.randint(1, 4)
            state_mask[-end_state:] = 1

    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }

# {
#     'states': tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
#     'actions': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# }
def create_inverse_dynamics_mask(
    traj_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    state_mask = np.ones(traj_length)
    action_mask = np.zeros(traj_length)
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


# {
#     'states': tensor([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]),
#     'actions': tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]),
#     'rewards': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
#     'returns': tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
# }

def create_forward_dynamics_mask(
    traj_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    state_mask = np.zeros(traj_length)
    index = np.random.randint(0, traj_length - 1)
    state_mask[:index] = 1

    action_mask = np.ones(traj_length)
    reward_mask = np.zeros(traj_length)
    return_mask = np.zeros(traj_length)
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
        "rewards": torch.from_numpy(reward_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(device),
    }

# {
#     'states': tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]),
#     'actions': tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]),
#     'rewards': tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]),
#     'returns': tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
# }
def create_random_masks(
    data_shapes, mask_ratios, traj_length, device
) -> Dict[str, np.ndarray]:
    masks = {}
    for k in data_shapes.keys():
        # create a random mask, different mask for each modality
        random_mask = create_random_mask(traj_length, mask_ratios, device)
        masks[k] = random_mask
    return masks

# {
#     'states': tensor([[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]),
#     'actions': tensor([[1.], [1.], [1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.]]),
#     'rewards': tensor([[1.], [1.], [1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.]]),
#     'returns': tensor([[1.], [1.], [1.], [1.], [1.], [0.], [0.], [0.], [0.], [0.]])
# }
def create_full_random_masks(
    data_shapes, mask_ratios, traj_length, device
) -> Dict[str, np.ndarray]:
    masks = {}
    # hardcode mask ratio. make it follow cosin funciton
    mask_ratios = np.linspace(0.15, 0.9, 30)
    mask_ratios = np.cos(mask_ratios * np.pi) / 2 + 0.5  # following mask git
    mask_ratios = mask_ratios.tolist()

    for k, v in data_shapes.items():
        # create a random mask, different mask for each modality
        random_mask = create_full_random_mask(v, traj_length, mask_ratios, device)
        masks[k] = random_mask
    return masks


def maybe_add_rew_to_mask(traj_length, device, mask_fn, add_rew, add_img, add_ret):
    masks = mask_fn(traj_length, device)
    if add_rew and "rewards" not in masks:
        masks["rewards"] = masks["actions"].clone()
        if len(masks["rewards"].shape) == 2:
            masks["rewards"] = masks["rewards"][..., 0:1]
    if add_ret and "returns" not in masks:
        masks["returns"] = masks["actions"].clone()
        if len(masks["returns"].shape) == 2:
            masks["returns"] = masks["returns"][..., 0:1]
    if add_img:
        masks["images"] = masks["states"].clone()
    return masks

# {
#     'states': tensor([1., 1., 1., 1., 1., 1., 1., 1., 0., 0.]),
#     'actions': tensor([1., 1., 1., 1., 1., 1., 1., 0., 0., 0.])
# }
def create_bc_mask(
    traj_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    state_mask = np.ones(traj_length)
    action_mask = np.ones(traj_length)
    index = np.random.randint(0, traj_length)
    action_mask[index:] = 0
    state_mask[index + 1 :] = 0
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


# {
#     'states': tensor([1., 1., 1., 0., 0.]),
#     'returns': tensor([1., 1., 1., 1., 1.]),
#     'actions': tensor([1., 1., 0., 0., 0.])
# }

def create_rcbc_mask(
    traj_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    state_mask = np.ones(traj_length)
    action_mask = np.ones(traj_length)
    index = np.random.randint(0, traj_length)
    action_mask[index:] = 0
    state_mask[index + 1 :] = 0
    return_mask = np.ones(traj_length)
    return {
        "states": torch.from_numpy(state_mask).to(device),
        "returns": torch.from_numpy(return_mask).to(
            device
        ),  # returns copies state mask
        "actions": torch.from_numpy(action_mask).to(device),
    }

# {
#     "states": tensor([[1., 1.],
#                       [1., 1.],
#                       [0., 0.],
#                       [0., 0.]], device='cpu'),
#     "returns": tensor([[1.],
#                        [0.],
#                        [0.],
#                        [0.]], device='cpu'),
#     "actions": tensor([[1.],
#                        [0.],
#                        [0.],
#                        [0.]], device='cpu')
# }


def create_random_autoregressize_mask(
    data_shapes, mask_ratios, traj_length, device, p_weights=(0.2, 0.1, 0.7)
) -> Dict[str, np.ndarray]:

    mode_order = ["states", "returns", "actions"]
    random_mode = np.random.choice(mode_order, p=p_weights)

    random_position = np.random.randint(0, traj_length)
    #random_position = np.random.randint(5, traj_length)

    masks = {}

    for k, v in data_shapes.items():
        # create a random mask, different mask for each modality
        masks[k] = create_full_random_mask(v, traj_length, mask_ratios, device)

    end_plus_one = False
    for k in mode_order:
        if k == random_mode:
            end_plus_one = True

        # mask out future
        if k in masks:
            if end_plus_one:
                masks[k][random_position:, :] = 0
            else:
                masks[k][random_position + 1 :, :] = 0

    # print(random_mode, random_position)
    return masks


def create_random_bc_masks(
    traj_length, device, data_shapes, p=0.5
) -> Dict[str, np.ndarray]:
    state_mask = np.ones((traj_length, data_shapes["states"][0]))
    action_mask = np.ones((traj_length, data_shapes["actions"][0]))
    index = np.random.randint(0, traj_length)
    action_mask[index:] = 0
    state_mask[index + 1 :] = 0

    action_mask[:index] = action_mask[:index] * np.random.choice(
        a=[1.0, 0.0], size=action_mask[:index].shape, p=[1 - p, p]
    )
    state_mask[: index + 1] = state_mask[: index + 1] * np.random.choice(
        a=[1.0, 0.0], size=state_mask[: index + 1].shape, p=[1 - p, p]
    )

    return {
        "states": torch.from_numpy(state_mask).to(device),
        "actions": torch.from_numpy(action_mask).to(device),
    }


def main():
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 2, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [1], 4, "cpu"
    )
    print(m)
    print()
    print()
    print()
    print()
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)
    m = create_random_autoregressize_mask(
        {"states": (3, 100), "returns": (1, 100), "actions": (2, 100)}, [0.35], 4, "cpu"
    )
    print(m)



#     {'states': tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [1.]], dtype=torch.float64), 'actions': tensor([[1., 1.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[1., 1., 1.],
#         [0., 0., 0.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[1., 1.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[1., 1., 1.],
#         [0., 0., 0.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[0., 0.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [1.]], dtype=torch.float64), 'actions': tensor([[1., 1.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[1., 1.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [1.]], dtype=torch.float64), 'actions': tensor([[1., 1.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [0., 0., 0.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [1.],
#         [1.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[1., 1.],
#         [1., 1.],
#         [1., 1.],
#         [0., 0.]], dtype=torch.float64)}




# {'states': tensor([[0., 1., 0.],
#         [1., 1., 0.],
#         [0., 0., 1.],
#         [0., 0., 0.]], dtype=torch.float64), 'returns': tensor([[0.],
#         [1.],
#         [0.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[0., 0.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[1., 0., 0.],
#         [1., 1., 1.],
#         [0., 0., 0.],
#         [0., 0., 0.]], dtype=torch.float64), 'returns': tensor([[1.],
#         [0.],
#         [0.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[1., 0.],
#         [1., 0.],
#         [0., 0.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[0., 0., 0.],
#         [1., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]], dtype=torch.float64), 'returns': tensor([[0.],
#         [0.],
#         [0.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[0., 0.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.]], dtype=torch.float64)}
# {'states': tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]], dtype=torch.float64), 'returns': tensor([[0.],
#         [0.],
#         [0.],
#         [0.]], dtype=torch.float64), 'actions': tensor([[0., 0.],
#         [0., 0.],
#         [0., 0.],
#         [0., 0.]], dtype=torch.float64)}

if __name__ == "__main__":
    main()
