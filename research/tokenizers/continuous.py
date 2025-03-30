import torch
from numpy.typing import ArrayLike
from research.tokenizers.base import Tokenizer
from research.buffer.d4rl_buffer import DataStatistics


class ContinuousTokenizer(Tokenizer):
    def __init__(
        self,
        data_mean: ArrayLike,
        data_std: ArrayLike,
        stats: DataStatistics,
        normalize: bool = True,
    ):
        super().__init__()
        self._data_mean = torch.nn.Parameter(
            torch.tensor(data_mean, dtype=torch.float32), requires_grad=False
        )
        self._data_std = torch.nn.Parameter(
            torch.tensor(data_std, dtype=torch.float32), requires_grad=False
        )
        self.stats = stats
        self.normalize = normalize

    @classmethod
    def create(
        cls, key: str, train_dataset, normalize: bool = True
    ) -> "ContinuousTokenizer":
        data = []
        stats = train_dataset.trajectory_statistics()[key]
        data_mean = stats.mean
        data_std = stats.std
        #print(data_mean,data_std) [96.39118] [304.96222]
        #print( data_mean, data_std,stats.max,stats.min)
        data_std[data_std < 0.1] = 1  # do not normalize if std is too small
        return cls(data_mean, data_std, stats, normalize=normalize)

    @property
    def discrete(self) -> bool:
        return False

    def encode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 3

        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)
            # normalize trajectory
            trajectory = (trajectory - mean) / std
        #(batch_size, sequence_length, 1, feature_dimension)
        return trajectory.unsqueeze(2).to(torch.float32)

    def decode(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        assert trajectory.dim() == 4
        assert trajectory.size(2) == 1
        if self.normalize:
            mean = self._data_mean.to(trajectory.device)
            std = self._data_std.to(trajectory.device)

            # denormalize trajectory
            return trajectory.squeeze(2) * std + mean
        else:
            return trajectory.squeeze(2)
