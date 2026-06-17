from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch

from kanlib.nn.spline_basis import AdaptiveGrid, GridUpdate, SplineBasis

_APPROXIMATED_BSPLINE_ORDER = 3


@dataclass
class _GridStats:
    len: torch.Tensor
    min: torch.Tensor
    max: torch.Tensor


class GaussianRbfBasis(SplineBasis, AdaptiveGrid):
    def __init__(
        self,
        grid_size: int,
        spline_range: torch.Tensor,
        adaptive_grid_margin: float = 0.01,
    ) -> None:
        super().__init__(
            grid_size=grid_size,
            spline_range=spline_range,
            initialize_grid=partial(
                _initialize_grid, grid_size=grid_size, spline_range=spline_range
            ),
        )
        self.margin = adaptive_grid_margin

        grid_stats = _compute_grid_stats(self.grid)
        self.register_buffer("_grid_len", grid_stats.len)
        self.register_buffer("_grid_min", grid_stats.min)
        self.register_buffer("_grid_max", grid_stats.max)

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size + _APPROXIMATED_BSPLINE_ORDER

    def forward(
        self, x: torch.Tensor, grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self._compute_grbf_basis(x, grid)

    @torch.no_grad()
    def grid_update_from_samples(self, x: torch.Tensor) -> GridUpdate:
        x_transposed = x.view(-1, self.num_features).T
        smin = x_transposed.min(dim=-1, keepdim=True).values - self.margin
        smax = x_transposed.max(dim=-1, keepdim=True).values + self.margin
        spline_range = torch.hstack([smin, smax])
        return GridUpdate(
            grid=_initialize_grid(grid_size=self.grid_size, spline_range=spline_range),
            spline_range=spline_range,
        )

    def _set_grid(self, grid: torch.Tensor) -> None:
        grid_stats = _compute_grid_stats(grid)
        self._grid_len = grid_stats.len
        self._grid_min = grid_stats.min
        self._grid_max = grid_stats.max
        super()._set_grid(grid)

    def _compute_grbf_basis(
        self, x: torch.Tensor, grid: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if grid is None:
            epsilon = (self._grid_len - 1) / (self._grid_max - self._grid_min)
            distance = x.unsqueeze(dim=-1) - self.grid
        else:
            stats = _compute_grid_stats(grid)
            epsilon = (stats.len - 1) / (stats.max - stats.min)
            distance = x.unsqueeze(dim=-1) - grid

        return torch.exp(-1.0 * (epsilon * distance) ** 2)


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


def _compute_grid_stats(grid: torch.Tensor) -> _GridStats:
    return _GridStats(
        len=torch.tensor(grid.size(dim=-1)),
        min=grid.min(dim=-1, keepdim=True).values,
        max=grid.max(dim=-1, keepdim=True).values,
    )
