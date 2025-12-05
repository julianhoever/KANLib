from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, Self

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
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class AdaptiveGrid(ABC):
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.update_grid = _validate_basis_inputs(cls.update_grid)

    @abstractmethod
    def update_grid(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> None: ...
