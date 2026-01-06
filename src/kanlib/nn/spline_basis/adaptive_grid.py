from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from ._validation import validate_basis_inputs, validate_grid, validate_spline_range


@dataclass
class GridUpdate:
    grid: torch.Tensor
    spline_range: torch.Tensor


class AdaptiveGrid(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.grid: torch.Tensor
        self.spline_range: torch.Tensor

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.grid_update_from_samples = validate_basis_inputs(
            cls.grid_update_from_samples
        )

    @property
    @abstractmethod
    def num_features(self) -> int: ...

    @abstractmethod
    def forward(
        self, x: torch.Tensor, grid: Optional[torch.Tensor]
    ) -> torch.Tensor: ...

    @abstractmethod
    def grid_update_from_samples(self, x: torch.Tensor) -> GridUpdate: ...

    def update_grid(self, grid_update: GridUpdate) -> None:
        validate_grid(grid_update.grid, self.num_features)
        validate_spline_range(grid_update.spline_range, self.num_features)
        self.grid = grid_update.grid
        self.spline_range = grid_update.spline_range
