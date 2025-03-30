
import numpy as np
import torch
from research.algo.planning import cem_planner,pytorch_cov


@torch.inference_mode()
def sample_action_with_bc(
        observation: np.ndarray,
        traj,
        model,
        tokenizer_manager,
        observation_shape,
        action_shape,
        device,
        percentage = 1.0,
):
    traj_len = model.max_len
    observations = np.zeros((traj_len, observation_shape))
    actions = np.zeros((traj_len, action_shape))

    masks = np.zeros(traj_len)
    i = -1
    max_len = min(traj_len - 1, len(traj))
    # print("max_len",max_len) 3
    assert max_len < traj_len
    for i in range(max_len):
        observations[i] = traj.observations[-max_len + i]
        actions[i] = traj.actions[-max_len + i]
        masks[i] = 1

    assert i == max_len - 1

    # fill in the rest with the current observation
    observations[i + 1] = observation
    obs_mask = np.copy(masks)
    obs_mask[i + 1] = 1

    if True:
        return_max = tokenizer_manager.tokenizers["returns"].stats.max
        return_min = tokenizer_manager.tokenizers["returns"].stats.min
        # print("return_max_min",return_max,return_min)[4.378059] [-0.01247974]
        target_return = return_min + (return_max - return_min) * percentage
        return_to_go = float(target_return)
        returns = return_to_go * np.ones((traj_len, 1))
        returns_mask = np.ones(traj_len)

    trajectories = {
        "states": observations,
        "actions": actions,
        "returns": returns,
    }
    masks = {
        "states": obs_mask,
        "actions": masks,
        "returns": returns_mask,
    }

    # convert to tensors and add
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}

    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted,v = model(encoded_trajectories, torch_masks)
    [v_out, n_v_out, q_out, q_out_old] = v

    decode = tokenizer_manager.decode(predicted)

    #print("debug",decode["actions"].shape,q_out_old.shape)
    a = decode["actions"][0][i + 1].cpu().numpy()
    v = q_out_old[0][i + 1].cpu().numpy()
    # print("debug", v, v.shape)

    return a, v


@torch.inference_mode()
def sample_action_with_double_check(
    observation: np.ndarray,
    traj,
    model,
    tokenizer_manager,
    observation_shape,
    action_shape,
    device,
    percentage = 1.0,
    eps = 1.0,
    N = 30,
    argmin_alignment = False
):
    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min
    MAX_return = return_min + (return_max - return_min)
    #percentages = np.random.uniform(0.5, 1.5, N)
    #percentages = np.random.uniform(percentage - eps, percentage + eps, N)
    percentages = np.random.uniform(percentage, percentage + eps, N)

    traj_len = model.max_len
    observations = np.zeros((traj_len, observation_shape))
    actions = np.zeros((traj_len, action_shape))
    alignment_returns = np.ones((traj_len, 1))

    masks = np.zeros(traj_len)
    i = -1
    # print("debug",traj,len(traj))
    max_len = min(traj_len - 1, len(traj))
    assert max_len < traj_len
    for i in range(max_len):
        observations[i] = traj.observations[-max_len + i]
        actions[i] = traj.actions[-max_len + i]
        alignment_returns[i] = traj.returns[-max_len + i]
        alignment_returns[i + 1] = (traj.returns[-max_len + i] - traj.rewards[-max_len + i]) / 0.99
        #print("debug", alignment_returns[i + 1])
        masks[i] = 1

    assert i == max_len - 1
    observations[i + 1] = observation
    # print("debug",i,max_len,traj,observations)

    obs_mask = np.copy(masks)
    obs_mask[i + 1] = 1
    pos = i+1

    trajectories = {
        "states": observations,
        "actions": actions,
    }
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None].repeat(N, 1, 1)
        for k, v in trajectories.items()
    }

    if True:
        # print("return_max_min",return_max,return_min)[4.378059] [-0.01247974]
        target_return = MAX_return * percentages
        ones_array = np.ones((len(target_return), traj_len, 1))
        returns = target_return[:, np.newaxis, np.newaxis] * ones_array
        #print("debug168",returns[0],returns.shape,returns[0].shape)
        return_mask = np.ones(traj_len)

    torch_trajectories["returns"] = torch.tensor(returns, device=device)
    masks = {
        "states": obs_mask,
        "actions": masks,
        "returns": return_mask
    }
    # print("debug440", masks, i)
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}

    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    predicted, value_outs = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)

    [v_out, n_v_out, q_out, q_out_old] = value_outs
    value_return = q_out_old[:, pos]

    if argmin_alignment:
        alignment_returns = torch.tensor(alignment_returns, device=device)
        value_diff = abs(q_out_old[:, pos] - alignment_returns[pos])# torch.Size([120])
        best_action_idx = np.argmin(value_diff.cpu().numpy())
    else:
        best_action_idx = np.argmax(value_return.cpu().numpy())

    new_decode = {k: v[best_action_idx] for k, v in decode.items()}
    # print("debug", new_decode["actions"].shape,best_action_idx)
    a = new_decode["actions"][i + 1].cpu().numpy()
    v = value_return[best_action_idx].cpu().numpy()
    # print("debug",v,v.shape,value_return.shape)
    return a, v



# ################# interaction samplers #########################

@torch.inference_mode()
def interaction_sampler_with_value_diff(
        observation: np.ndarray,
        traj,
        model,
        tokenizer_manager,
        observation_shape,
        action_shape,
        device,
        N=300,
        c = 0.5
):
    percentages = np.random.uniform(0.7, 3.0, N)
    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min
    Max_return = return_min + (return_max - return_min)

    traj_len = model.max_len
    observations = np.zeros((traj_len, observation_shape))
    actions = np.zeros((traj_len, action_shape))

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
    pos = i + 1

    trajectories = {
        "states": observations,
        "actions": actions,
    }
    # extract_action sequence make new torch sequence with 1024 copies
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None].repeat(N, 1, 1)
        for k, v in trajectories.items()
    }

    if True:
        # print("return_max_min",return_max,return_min)[4.378059] [-0.01247974]
        return_value = Max_return * percentages
        ones_array = np.ones((len(return_value), traj_len, 1))
        returns = return_value[:, np.newaxis, np.newaxis] * ones_array
        return_mask = np.ones(traj_len)

    torch_trajectories["returns"] = torch.tensor(returns, device=device)

    masks = {
        "states": obs_mask,
        "actions": masks,
        "returns": return_mask
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}

    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)
    model.compute_value_diff = True
    predicted, value_outs = model(encoded_trajectories, torch_masks)
    model.compute_value_diff = False
    decode = tokenizer_manager.decode(predicted)

    [v_out, n_v_out, q_out, q_out_old] = value_outs
    if np.random.uniform() > 0.5:
        q_out_flat = q_out[0][:, pos].view(-1)
    else:
        q_out_flat = q_out[1][:, pos].view(-1)
    q_old_flat = q_out_old[:, pos].view(-1)
    top20_values, top20_indices = torch.topk(q_out_flat, 20, largest=True)
    # print("debug",top20_indices)
    diff = torch.abs(q_out_flat[top20_indices] - q_old_flat[top20_indices])
    min_diff, min_diff_idx = torch.min(diff, dim=0)
    best_action_idx = top20_indices[min_diff_idx]
    # print("debug", best_action_idx,min_diff_idx)

    new_decode = {k: v[best_action_idx] for k, v in decode.items()}

    #  ε ~ N(0, delta)，clip to [-c, c]
    delta = diff[min_diff_idx].item()
    # print("delta",delta)
    epsilon = torch.randn_like(new_decode["actions"]) * delta
    epsilon = torch.clamp(epsilon, -c, c)
    new_decode["actions"] += epsilon
    new_decode["actions"] = torch.clamp(new_decode["actions"], -1, 1)

    # new_decode["actions"] += (torch.randn_like(new_decode["actions"]) * 0.03)
    # new_decode["actions"] = torch.clamp(new_decode["actions"], -1, 1)

    a = new_decode["actions"][i + 1].cpu().numpy()

    return a

@torch.inference_mode()
def interaction_sampler_with_bc(
        observation: np.ndarray,
        traj,
        model,
        tokenizer_manager,
        observation_shape,
        action_shape,
        device,
        percentage=1.0,
):
    traj_len = model.max_len
    return_max = tokenizer_manager.tokenizers["returns"].stats.max
    return_min = tokenizer_manager.tokenizers["returns"].stats.min
    Max_return = return_min + (return_max - return_min)

    observations = np.zeros((traj_len, observation_shape))
    actions = np.zeros((traj_len, action_shape))

    masks = np.zeros(traj_len)

    i = -1
    max_len = min(traj_len - 1, len(traj))
    # print("max_len",max_len)3
    assert max_len < traj_len
    for i in range(max_len):
        observations[i] = traj.observations[-max_len + i]
        actions[i] = traj.actions[-max_len + i]
        masks[i] = 1

    assert i == max_len - 1

    # fill in the rest with the current observation
    observations[i + 1] = observation
    obs_mask = np.copy(masks)
    obs_mask[i + 1] = 1

    if True:
        return_value = Max_return * percentage
        return_to_go = float(return_value)
        returns = return_to_go * np.ones((traj_len, 1))
        returns_mask = np.ones(traj_len)

    # pass through tokenizer
    trajectories = {
        "states": observations,
        "actions": actions,
        "returns": returns,
    }
    masks = {
        "states": obs_mask,
        "actions": masks,
        "returns": returns_mask}
    # convert to tensors and add
    torch_trajectories = {
        k: torch.tensor(v, device=device)[None] for k, v in trajectories.items()
    }
    torch_masks = {k: torch.tensor(v, device=device) for k, v in masks.items()}

    encoded_trajectories = tokenizer_manager.encode(torch_trajectories)

    predicted, v = model(encoded_trajectories, torch_masks)
    decode = tokenizer_manager.decode(predicted)
    # extract_action
    a = decode["actions"][0][i + 1].cpu().numpy()
    return a







