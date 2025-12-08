from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional, Protocol, Self

import torch


class _InputValidatedFunc[S, R](Protocol):
    def __call__(_: Self, self: S, x: torch.Tensor, *args: Any, **kwargs: Any) -> R: ...  # pyright: ignore[reportSelfClsParameterName]


def _validate_basis_inputs[S: Any, R: Any](
    func: _InputValidatedFunc[S, R],
) -> _InputValidatedFunc[S, R]:
    def wrapper(self: S, x: torch.Tensor, *args: Any, **kwargs: Any) -> R:
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but got {x.shape[-1]}."
            )
        return func(self, x, *args, **kwargs)

    return wrapper


def _validate_grid(grid: torch.Tensor, num_features: int) -> None:
    if grid.ndim != 2 or grid.shape[0] != num_features:
        raise ValueError("`grid` must be of shape `(num_features, num_grid_points)`")


class SplineBasis(torch.nn.Module, ABC):
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.forward = _validate_basis_inputs(cls.forward)

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

        _validate_grid(grid=self.grid, num_features=spline_range.shape[0])

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
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class AdaptiveGrid(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.grid: torch.Tensor

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.updated_grid_from_samples = _validate_basis_inputs(
            cls.updated_grid_from_samples
        )

    @property
    @abstractmethod
    def num_features(self) -> int: ...

    @abstractmethod
    def forward(
        self, x: torch.Tensor, grid: Optional[torch.Tensor]
    ) -> torch.Tensor: ...

    @abstractmethod
    def updated_grid_from_samples(self, x: torch.Tensor) -> torch.Tensor: ...

    def set_grid(self, grid: torch.Tensor) -> None:
        _validate_grid(grid, self.num_features)
        self.grid = grid
