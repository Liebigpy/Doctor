from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union,Callable
import torch
import torch.nn as nn
from copy import deepcopy
from research.models.mtm_model import MTM



class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], -1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class Doctor_Transformer(MTM):
    def __init__(
            self,
            data_shapes: Dict[str, Tuple[int, ...]],
            traj_length: int,
            config: None
    ) -> None:
        super().__init__(
            data_shapes =data_shapes,
            traj_length = traj_length,
            config = config,
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.n_enc_layer,
            norm=nn.LayerNorm(self.n_embd),
        )

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.n_dec_layer,
            norm=nn.LayerNorm(self.n_embd),
        )

        self.critic_q = TwinQ(self.n_embd, self.n_embd)
        self.critic_v = ValueFunction(self.n_embd)
        self.critic_q_target = deepcopy(self.critic_q)
        self.critic_q_target.eval()

        self.compute_value_diff = False

        self.reward_scale = 0.0
        self.Max_return = 0.0

        self.output_head_dict = nn.ModuleDict()
        for key, shape in data_shapes.items():
            self.output_head_dict[key] = nn.Sequential(
                nn.LayerNorm(self.n_embd),
                nn.Linear(self.n_embd, self.n_embd),
                nn.GELU(),
                nn.Linear(self.n_embd, shape[-1]),
            )


    @staticmethod
    def forward_loss(
            targets: Dict[str, torch.Tensor],
            preds: Dict[str, torch.Tensor],
            masks: Dict[str, torch.Tensor],
            discrete_map: Dict[str, bool],
            norm="l2",
            reduce_use_sum=False,
            loss_keys: Optional[List[str]] = None,
    ):
        losses = {}
        loss_keys = loss_keys
        for key in loss_keys:
            target = targets[key]
            pred = preds[key]
            mask = masks[key]
            #print("debug81", key, target, pred, mask)
            if len(mask.shape) == 1:
                # only along time dimension: repeat across the given dimension
                mask = mask[:, None].repeat(1, target.shape[2])
            elif len(mask.shape) == 2:
                pass

            batch_size, T, P, _ = target.size()
            raw_loss = nn.MSELoss(reduction="none")(pred, target)

            if reduce_use_sum:
                loss = raw_loss.sum(dim=(2, 3)).mean()
            else:
                # print("1")
                loss = raw_loss.mean(dim=(2, 3)).mean()

            losses[key] = loss
        loss = torch.sum(torch.stack([losses[key] for key in loss_keys]))
        return loss, losses


    def forward_encoder(self, trajectories, masks):
        features = []
        ids_restore = {}
        keep_len = {}
        # process obs

        keys = list(trajectories.keys())  # get the keys in a list to maintain order
        for k in keys:
            traj = trajectories[k]
            mask = masks[k]
            x, ids_restore[k], keep_len[k] = self._index(traj, mask)
            features.append(x)

        x = torch.cat(features, dim=1)
        x = self.encoder(x)

        idx = 0
        encoded_trajectories = {}
        for k in keys:
            v = keep_len[k]
            encoded_trajectories[k] = x[:, idx : idx + v]
            idx += v

        return encoded_trajectories, ids_restore, keep_len

    def forward_decoder(
        self,
        trajectories: Dict[str, torch.Tensor],
        ids_restore: Dict[str, torch.Tensor],
        keep_lengths: Dict[str, torch.Tensor],
    ):
        encoded_trajectories_with_mask = {}
        keys = list(trajectories.keys())
        reconstruction_keys = ['states', 'actions', 'returns']
        for k in reconstruction_keys:
            traj = trajectories[k]
            batch_size = traj.shape[0]
            assert len(ids_restore[k].shape) == 1
            num_mask_tokens = ids_restore[k].shape[0] - keep_lengths[k]
            mask_tokens = self.mask_token_dict[k].repeat(batch_size, num_mask_tokens, 1)
            x_ = torch.cat([traj, mask_tokens], dim=1)
            assert (
                ids_restore[k].shape[0] == x_.shape[1]
            ), f"{ids_restore[k]}, {x_.shape}"

            # reorganize the indicies to be in their original positions
            x_ = torch.gather(
                x_,
                1,
                ids_restore[k][None, :, None].repeat(batch_size, 1, traj.shape[-1]),
            )
            encoded_trajectories_with_mask[k] = x_

        decoder_embedded_trajectories = self._decoder_trajectory_encoding(
            encoded_trajectories_with_mask
        )
        concat_trajectories = torch.cat(
            [decoder_embedded_trajectories[k] for k in keys], dim=1
        )

        x = self.decoder(concat_trajectories)

        critic_trajectories = {}
        extracted_trajectories = {}
        pos = 0
        for k in keys:
            b, t_p, f = decoder_embedded_trajectories[k].shape #torch.Size([1, 4, 512])
            output_head = self.output_head_dict[k]
            traj_segment = x[:, pos : pos + t_p, :]
            critic_trajectories[k] = x[:, pos: pos + t_p, :].clone()

            p = self.data_shapes[k][0]
            extracted_trajectories[k] = output_head(traj_segment.reshape(b, -1, p, f))
            pos += t_p

        states_in = critic_trajectories['states'].detach()
        actions_in = critic_trajectories['actions'].detach()
        v_out = self.critic_v(states_in)
        q_out = self.critic_q.both(states_in, actions_in)
        if self.compute_value_diff:
            with torch.no_grad():
                q_out_old = self.critic_q_ema(states_in, actions_in)
        else:
            with torch.no_grad():
                q_out_old = self.critic_q_target(states_in, actions_in)

        return extracted_trajectories, [v_out, [], q_out, q_out_old]

    def forward(self, trajectories, masks):
        """
        Args:
            trajectories: (batch_size, T, tokens_per_time, feature_dim)
            masks: (T,) or (T, tokens_per_time), or (batch_size, T, tokens_per_time)
        """
        batched_masks = self.process_masks(trajectories, masks)
        embedded_trajectories = self.trajectory_encoding(trajectories)

        encoded_trajectories, ids_restore, keep_length = self.forward_encoder(
            embedded_trajectories, batched_masks
        )
        decoded_trajectories, value_list = self.forward_decoder(encoded_trajectories, ids_restore, keep_length)

        return decoded_trajectories, value_list



