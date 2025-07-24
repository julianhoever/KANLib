from abc import ABC, abstractmethod
from typing import Protocol

import torch


class _InitializeGrid(Protocol):
    def __call__(
        self, grid_size: int, grid_range: tuple[float, float]
    ) -> torch.Tensor: ...


class SplineBasis(torch.nn.Module, ABC):
    def __init__(
        self,
        grid_size: int,
        grid_range: tuple[float, float],
        initialize_grid: _InitializeGrid,
    ) -> None:
        super().__init__()
        if grid_size < 1:
            raise ValueError("`grid_size` must be at least 1.")

        self.grid_size = grid_size
        self.grid_range = grid_range
        self.initialize_grid = initialize_grid
        self.grid: torch.Tensor
        self.register_buffer(
            "grid", self.initialize_grid(grid_size=grid_size, grid_range=grid_range)
        )

    @property
    @abstractmethod
    def num_basis_functions(self) -> int: ...
