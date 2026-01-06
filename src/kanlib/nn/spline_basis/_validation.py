from typing import Any, Optional, Protocol, Self

import torch


class _InputValidatedFunc[S, R](Protocol):
    def __call__(_: Self, self: S, x: torch.Tensor, *args: Any, **kwargs: Any) -> R: ...  # pyright: ignore[reportSelfClsParameterName]


def validate_basis_inputs[S: Any, R: Any](
    func: _InputValidatedFunc[S, R],
) -> _InputValidatedFunc[S, R]:
    def wrapper(self: S, x: torch.Tensor, *args: Any, **kwargs: Any) -> R:
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, but got {x.shape[-1]}."
            )
        return func(self, x, *args, **kwargs)

    return wrapper


def validate_grid(grid: torch.Tensor, num_features: int) -> None:
    if grid.ndim != 2 or grid.shape[0] != num_features:
        raise ValueError("`grid` must be of shape `(num_features, num_grid_points)`")


def validate_spline_range(
    spline_range: torch.Tensor, num_features: Optional[int] = None
) -> None:
    if (
        spline_range.ndim != 2
        or spline_range.shape[-1] != 2
        or (num_features is not None and spline_range.shape[0] != num_features)
    ):
        raise ValueError("`spline_range` must be of shape `(num_features, 2)`")
