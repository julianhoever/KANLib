from abc import ABC, abstractmethod
from typing import Protocol

import torch


class _InitializeGrid(Protocol):
    def __call__(
        self,
        num_features: int,
        grid_size: int,
        grid_range: torch.Tensor,
    ) -> torch.Tensor: ...


class SplineBasis(torch.nn.Module, ABC):
    def __init__(
        self,
        num_features: int,
        grid_size: int,
        grid_range: tuple[float, float] | torch.Tensor,
        initialize_grid: _InitializeGrid,
    ) -> None:
        super().__init__()
        if grid_size < 1:
            raise ValueError("`grid_size` must be at least 1.")

        if isinstance(grid_range, torch.Tensor):
            if grid_range.shape != (num_features, 2):
                raise ValueError("`grid_range` must be of shape `(num_features, 2)`")
        else:
            grid_range = _convert_grid_range_to_tensor(grid_range, num_features)

        self.num_features = num_features
        self.grid_size = grid_size
        self.grid: torch.Tensor
        self.register_buffer(
            "grid",
            initialize_grid(
                num_features=num_features, grid_size=grid_size, grid_range=grid_range
            ),
        )

    @property
    def grid_range(self) -> torch.Tensor:
        return torch.cat(
            [
                self.grid.min(dim=-1).values.unsqueeze(dim=-1),
                self.grid.max(dim=-1).values.unsqueeze(dim=-1),
            ],
            dim=-1,
        )

    @property
    @abstractmethod
    def num_basis_functions(self) -> int: ...

    @abstractmethod
    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.num_features:
            raise ValueError("`x` must be of shape `(*, num_features)`.")
        return self._perform_forward(x)


def _convert_grid_range_to_tensor(
    grid_range: tuple[float, float], num_features: int
) -> torch.Tensor:
    return torch.tensor(grid_range, dtype=torch.get_default_dtype()).repeat(
        num_features, 1
    )
