from abc import ABC, abstractclassmethod
from typing import Dict, Type, TypeVar

import torch
from torch.utils.data import Dataset

T = TypeVar("T", bound="Tokenizer")


class Tokenizer(torch.nn.Module, ABC):
    @abstractclassmethod
    def create(cls: Type[T], key: str, train_dataset, **kwargs) -> T:
        """Create a new instance of the model."""

    @property
    def discrete(self) -> bool:
        """Whether the tokenizer is discrete or continuous."""

    def encode(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode a trajectory.

        Args:
            trajectories (torch.Tensor): shape=(batch_size, L, ...))

        Returns:
            tokenized_trajectories (torch.Tensor): shape=(batch_size, L, tokens_per_dim, tokens_feature_size)
        """
        ...

    def decode(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Decode a trajectory.

        Args:
            tokenized_trajectories (torch.Tensor): shape=(batch_size, L, tokens_per_dim, tokens_feature_size)

        Returns:
            trajectories (torch.Tensor): shape=(batch_size, L, ...))
        """
        ...


class TokenizerManager(torch.nn.Module):
    def __init__(self, tokenizers: Dict[str, Tokenizer]):
        super().__init__()
        self.tokenizers = torch.nn.ModuleDict(tokenizers)
        # self.record = record
        # self.normalization_factor = args.normalization_factor
        # self.max_epi_len = args.max_epi_len
        # self.reward_scale = reward_scale
        # self.return_discount = return_discount
        # self.ant_maze = ant_maze

    def encode(self, trajectories: Dict[str, torch.Tensor], use_reward = False) -> Dict[str, torch.Tensor]:
        """Encode all of the trajectories.

        Args:
            trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, ...)). This could be states, actions, images, etc

        Returns:
            tokenized_trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, tokens_per_dim, tokens_feature_size))
        """
        out_trajectories = {}
        for key, value in trajectories.items():
            if key in self.tokenizers.keys():
                out_trajectories[key] = self.tokenizers[key].encode(value)
                assert len(out_trajectories[key].shape) == 4
        # if use_reward:
        #     out_trajectories["rewards"] = ((trajectories["rewards"] / self.normalization_factor) * self.max_epi_len).to(torch.float32)
        #print(trajectories.keys())
        # if self.return_discount:
        #     out_trajectories["returns"] = trajectories["returns"].unsqueeze(2).to(torch.float32) * self.reward_scale
        # if self.ant_maze:
        #     out_trajectories["returns"] = trajectories["returns"].unsqueeze(2).to(torch.float32)
        #     if use_reward:
        #         out_trajectories["rewards"] = trajectories["rewards"].unsqueeze(2).to(torch.float32) - 1.0
        return out_trajectories

    def decode(
        self, tokenzied_trajectories: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Decode all of the trajectories.

        Args:
            tokenized_trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, tokens_per_dim, tokens_feature_size))

        Returns:
            trajectories (Dict[str, torch.Tensor]): Each trajectory has shape=(batch_size, L, ...)).
        """
        out_trajectories = {}
        for key, value in tokenzied_trajectories.items():
            out_trajectories[key] = self.tokenizers[key].decode(value)
        # if self.ant_maze:
        #     #out_trajectories["rewards"] = tokenzied_trajectories["rewards"].squeeze(2) + 1.0
        #     out_trajectories["returns"] = tokenzied_trajectories["returns"].squeeze(2)
        # if self.return_discount:
        #     out_trajectories["returns"] = tokenzied_trajectories["returns"].squeeze(2) / self.reward_scale
        return out_trajectories
