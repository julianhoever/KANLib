from functools import partial
from typing import Optional

import torch

from kanlib.nn.spline_basis import AdaptiveGrid, GridUpdate, SplineBasis

_APPROXIMATED_BSPLINE_ORDER = 3


class GaussianRbfBasis(SplineBasis, AdaptiveGrid):
    def __init__(self, grid_size: int, spline_range: torch.Tensor) -> None:
        super().__init__(
            grid_size=grid_size,
            spline_range=spline_range,
            initialize_grid=partial(
                _initialize_grid, grid_size=grid_size, spline_range=spline_range
            ),
        )

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size + _APPROXIMATED_BSPLINE_ORDER

    def forward(
        self, x: torch.Tensor, grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return _compute_grbf_basis(x=x, grid=self.grid if grid is None else grid)

    def grid_update_from_samples(self, x: torch.Tensor) -> GridUpdate:
        x_transposed = x.view(-1, self.num_features).T
        smin = x_transposed.min(dim=-1, keepdim=True).values
        smax = x_transposed.max(dim=-1, keepdim=True).values
        spline_range = torch.hstack([smin, smax])
        return GridUpdate(
            grid=_initialize_grid(grid_size=self.grid_size, spline_range=spline_range),
            spline_range=spline_range,
        )


def _initialize_grid(grid_size: int, spline_range: torch.Tensor) -> torch.Tensor:
    smin, smax = spline_range.unsqueeze(dim=-2).unbind(dim=-1)
    scale = (smax - smin) / grid_size
    grid = torch.linspace(
        start=-_APPROXIMATED_BSPLINE_ORDER / 2,
        end=grid_size + _APPROXIMATED_BSPLINE_ORDER / 2,
        steps=grid_size + _APPROXIMATED_BSPLINE_ORDER,
        dtype=torch.get_default_dtype(),
        device=spline_range.device,
    ).repeat(spline_range.shape[0], 1)
    return grid * scale + smin


def _compute_grbf_basis(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    grid_min = grid.min(dim=-1, keepdim=True).values
    grid_max = grid.max(dim=-1, keepdim=True).values
    grid_len = grid.size(dim=-1)

    epsilon = (grid_len - 1) / (grid_max - grid_min)
    distance = x.unsqueeze(dim=-1) - grid

    return torch.exp(-1.0 * (epsilon * distance) ** 2)
