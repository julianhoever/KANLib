from functools import partial

import torch

from kanlib.nn.spline_basis import SplineBasis


class BSplineBasis(SplineBasis):
    def __init__(
        self, grid_size: int, spline_range: torch.Tensor, spline_order: int
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            spline_range=spline_range,
            initialize_grid=partial(
                _initialize_grid,
                grid_size=grid_size,
                spline_range=spline_range,
                spline_order=spline_order,
            ),
        )
        self.spline_order = spline_order

        if self.grid_size < 1:
            raise ValueError("`grid_size` must be at least 1")

    @property
    def spline_range(self) -> torch.Tensor:
        _, grid_points = self.grid.shape
        trimmed_grid = self.grid[:, self.spline_order : grid_points - self.spline_order]
        return trimmed_grid[:, [0, -1]]

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size + self.spline_order

    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _compute_bspline_basis(
            x=x, grid=self.grid, spline_order=self.spline_order
        )


def _initialize_grid(
    grid_size: int, spline_range: torch.Tensor, spline_order: int
) -> torch.Tensor:
    smin, smax = spline_range.unsqueeze(dim=-2).unbind(dim=-1)
    scale = (smax - smin) / grid_size
    grid = torch.arange(
        start=-spline_order,
        end=grid_size + spline_order + 1,
        dtype=torch.get_default_dtype(),
    ).repeat(spline_range.shape[0], 1)
    return grid * scale + smin


def _compute_bspline_basis(
    x: torch.Tensor, grid: torch.Tensor, spline_order: int
) -> torch.Tensor:
    """
    Implementation of B-spline basis functions is based on:
    https://github.com/Blealtan/efficient-kan/blob/7b6ce1c87f18c8bc90c208f6b494042344216b11/src/efficient_kan/kan.py#L78-L111
    """
    g = grid
    x = x.unsqueeze(dim=-1)

    b = ((x >= g[..., :-1]) & (x < g[..., 1:])).type(x.dtype)

    for k in range(1, spline_order + 1):
        b1 = (
            (x - g[..., : -(k + 1)]) / (g[..., k:-1] - g[..., : -(k + 1)]) * b[..., :-1]
        )
        b2 = (g[..., k + 1 :] - x) / (g[..., k + 1 :] - g[..., 1:-k]) * b[..., 1:]
        b = b1 + b2

    return b
