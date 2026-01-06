from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from ._validation import validate_basis_inputs, validate_grid, validate_spline_range


class SplineBasis(torch.nn.Module, ABC):
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.forward = validate_basis_inputs(cls.forward)

    def __init__(
        self,
        grid_size: int,
        spline_range: torch.Tensor,
        initialize_grid: Callable[[], torch.Tensor],
    ) -> None:
        super().__init__()
        validate_spline_range(spline_range)

        self.grid_size = grid_size
        self.spline_range = spline_range

        self.grid: torch.Tensor
        self.register_buffer("grid", initialize_grid())

        validate_grid(grid=self.grid, num_features=spline_range.shape[0])

    @property
    def num_features(self) -> int:
        return self.grid.shape[0]

    @property
    @abstractmethod
    def num_basis_functions(self) -> int: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
