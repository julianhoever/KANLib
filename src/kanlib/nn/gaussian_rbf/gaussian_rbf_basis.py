from functools import partial

import torch

from kanlib.nn.spline_basis import SplineBasis

_APPROXIMATED_BSPLINE_ORDER = 3


class GaussianRbfBasis(SplineBasis):
    def __init__(self, grid_size: int, spline_range: torch.Tensor) -> None:
        super().__init__(
            grid_size=grid_size,
            spline_range=spline_range,
            initialize_grid=partial(
                _initialize_grid, grid_size=grid_size, spline_range=spline_range
            ),
        )
        self.epsilon: torch.Tensor
        self.register_buffer(
            name="epsilon",
            tensor=(self.num_basis_functions - 1)
            / (self.spline_range[:, [1]] - self.spline_range[:, [0]]),
        )

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size + _APPROXIMATED_BSPLINE_ORDER

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _compute_grbf_basis(x=x, grid=self.grid, epsilon=self.epsilon)


def _initialize_grid(grid_size: int, spline_range: torch.Tensor) -> torch.Tensor:
    smin, smax = spline_range.unsqueeze(dim=-2).unbind(dim=-1)
    scale = (smax - smin) / grid_size
    grid = torch.linspace(
        start=-_APPROXIMATED_BSPLINE_ORDER / 2,
        end=grid_size + _APPROXIMATED_BSPLINE_ORDER / 2,
        steps=grid_size + _APPROXIMATED_BSPLINE_ORDER,
        dtype=torch.get_default_dtype(),
    ).repeat(spline_range.shape[0], 1)
    return grid * scale + smin


def _compute_grbf_basis(
    x: torch.Tensor, grid: torch.Tensor, epsilon: torch.Tensor
) -> torch.Tensor:
    distance = x.unsqueeze(dim=-1) - grid
    return torch.exp(-1.0 * (epsilon * distance) ** 2)
