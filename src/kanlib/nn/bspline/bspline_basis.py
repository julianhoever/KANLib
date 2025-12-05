from functools import partial
from typing import Any

import torch

from kanlib.nn.spline_basis import AdaptiveGrid, SplineBasis


class BSplineBasis(SplineBasis, AdaptiveGrid):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _compute_bspline_basis(
            x=x, grid=self.grid, spline_order=self.spline_order
        )

    @torch.no_grad
    def update_grid(
        self, x: torch.Tensor, margin: float = 0.0, uniform_fraction: float = 0.02
    ) -> None:
        """
        Implementation is based on:
        https://github.com/Blealtan/efficient-kan/blob/7b6ce1c87f18c8bc90c208f6b494042344216b11/src/efficient_kan/kan.py#L169-L215
        """

        def arange(*args: Any) -> torch.Tensor:
            values = torch.arange(
                *args, dtype=torch.get_default_dtype(), device=self.grid.device
            )
            return values.unsqueeze(dim=-1)

        x_flat = x.view(-1, self.num_features)
        x_sorted = torch.sort(x_flat, dim=0)[0]

        grid_sampling_mask = torch.linspace(
            start=0,
            end=len(x_sorted) - 1,
            steps=self.grid_size + 1,
            dtype=torch.int64,
            device=x.device,
        )
        grid_adaptive = x_sorted[grid_sampling_mask]

        min_value = x_sorted[0] - margin
        max_value = x_sorted[-1] + margin
        step_size = (max_value - min_value) / self.grid_size

        grid_uniform = arange(self.grid_size + 1) * step_size + min_value

        grid = uniform_fraction * grid_uniform + (1 - uniform_fraction) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1] - step_size * arange(self.spline_order, 0, -1),
                grid,
                grid[-1:] + step_size * arange(1, self.spline_order + 1),
            ],
            dim=0,
        )

        self.grid = grid.T


def _initialize_grid(
    grid_size: int, spline_range: torch.Tensor, spline_order: int
) -> torch.Tensor:
    smin, smax = spline_range.unsqueeze(dim=-2).unbind(dim=-1)
    scale = (smax - smin) / grid_size
    grid = torch.arange(
        start=-spline_order,
        end=grid_size + spline_order + 1,
        dtype=torch.get_default_dtype(),
        device=spline_range.device,
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
