
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn.functional as F
import torch.nn.parallel
from operator import itemgetter
from research.models.mtm_model import MTM
from research.tokenizers.base import Tokenizer, TokenizerManager


# {
#     'states': tensor([1., 1., 1., 0.]),
#     'actions': tensor([0., 0., 1., 0.]),
#     'returns': tensor([0., 0., 0., 0.])
# }
def eval_fd(
    model: MTM,
    env,
    eval_batch,
    tokenizer_manager,
    ratio: int = 1,
) -> Dict[str, Any]:
    """Evaluate the model on the forward dynamics task.
    Args:
        env (gym.Env): env
        eval_batch (Dict[str, torch.Tensor]): eval_batch
        tokenizer_manager (TokenizerManager): tokenizer_manager
    """
    seq_len = eval_batch["actions"].shape[1]
    batch_size = eval_batch["actions"].shape[0]
    device = eval_batch["states"].device
    assert seq_len >= 2, "Forward dynamics eval only works for seq_len=2"

    # Given initial state and all actions. Predict future states.
    obs_mask1 = torch.ones(seq_len, device=device)
    obs_mask1[-1] = 0
    actions_mask1 = torch.zeros(seq_len, device=device)
    actions_mask1[-2] = 1
    returns_mask = torch.zeros(seq_len, device=device)
    # rewards_mask = torch.ones(seq_len, device=device)
    masks = {
        "states": obs_mask1,
        "actions": actions_mask1,
        "returns": returns_mask,
        # "rewards": rewards_mask,
    }

    mtm_batch = {
        "states": eval_batch["states"],
        "actions": eval_batch["actions"],
        "returns": eval_batch["returns"],
        # "rewards": eval_batch["rewards"],
    }


    encoded_batch = tokenizer_manager.encode(mtm_batch)
    predictions, _ = model.mask_git_forward(
        encoded_batch,
        masks,
        ratio=ratio,
    )
    predicted_next_state = tokenizer_manager.decode(predictions)["states"]

    states = eval_batch["states"]
    next_state = states[:, -1]
    state_error = (next_state - predicted_next_state[:, -1, :]) ** 2
    eval_dict = {}
    eval_dict[f"eval/fd_error"] = torch.mean(state_error).item()
    return eval_dict




seq_len = 5



@torch.inference_mode()
def evaluate(
    model: MTM,
    tokenizer_manager: TokenizerManager,
    discrete_map: Dict[str, bool],
    val_batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    loss_keys = None
) -> Dict[str, Any]:

    mtm_batch = {
        "states": val_batch["states"],
        "actions": val_batch["actions"],
        "returns": val_batch["returns"],
        # "rewards": val_batch["rewards"],
    }
    # m_returns = masks["returns"].clone()
    # m_returns[m_returns == 1] = 0
    # masks["returns"] = m_returns

    encoded_batch = tokenizer_manager.encode(mtm_batch)
    predicted_trajectories, value_out = model(encoded_batch, masks)

    #model_without_ddp = model.module if hasattr(model, "module") else model
    (
        loss,
        losses_dict,
        # masked_losses,
        # masked_c_losses,
    ) = model.forward_loss(
        encoded_batch,
        predicted_trajectories,
        masks,
        discrete_map,
        loss_keys = loss_keys
        # norm=model_without_ddp.norm,
        # reduce_use_sum=model_without_ddp.config.reduce_use_sum,
        # loss_keys=model_without_ddp.config.loss_keys,
    )

    log_dict = {"val/total_loss": loss.item()}
    for k, v in losses_dict.items():
        log_dict[f"val/full_loss_{k}"] = v.item()
    # for k, v in masked_losses.items():
    #     log_dict[f"val/masked_loss_{k}"] = v.item()
    # for k, v in masked_c_losses.items():
    #     log_dict[f"val/masked_c_loss_{k}"] = v.item()

    mse_loss = 0
    predictions = tokenizer_manager.decode(predicted_trajectories)
    for k, v in predictions.items():
        _mse = F.mse_loss(v.to(torch.float32), mtm_batch[k].to(torch.float32)).item()
        log_dict[f"val/mse_{k}"] = _mse
        mse_loss += _mse
    log_dict["val/mse_sum"] = mse_loss

    return log_dict
