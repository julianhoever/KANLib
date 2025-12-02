from abc import ABC, abstractmethod
from collections.abc import Callable

import torch


class SplineBasis(torch.nn.Module, ABC):
    def __init__(
        self,
        grid_size: int,
        spline_range: torch.Tensor,
        initialize_grid: Callable[[], torch.Tensor],
    ) -> None:
        super().__init__()
        if spline_range.ndim != 2 or spline_range.shape[-1] != 2:
            raise ValueError("`spline_range` must be of shape `(num_features, 2)`")

        self.grid_size = grid_size
        self._initial_spline_range = spline_range

        self.grid: torch.Tensor
        self.register_buffer("grid", initialize_grid())

        if self.grid.ndim != 2 or self.grid.shape[0] != spline_range.shape[0]:
            raise ValueError(
                "`grid` must be of shape `(num_features, num_grid_points)`"
            )

    @property
    def spline_range(self) -> torch.Tensor:
        return self._initial_spline_range

    @property
    def num_features(self) -> int:
        return self.grid.shape[0]

    @property
    @abstractmethod
    def num_basis_functions(self) -> int: ...

    @abstractmethod
    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_features != 1 and x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but got {x.shape[-1]}."
            )
        return self._perform_forward(x)
