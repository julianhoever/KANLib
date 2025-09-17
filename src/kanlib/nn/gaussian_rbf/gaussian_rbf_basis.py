from functools import partial

import torch

from kanlib.nn.spline_basis import SplineBasis


class GaussianRbfBasis(SplineBasis):
    _APPROXIMATED_BSPLINE_ORDER = 3

    def __init__(self, grid_size: int, grid_range: tuple[float, float]) -> None:
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            initialize_grid=partial(
                _initialize_grid, spline_order=self._APPROXIMATED_BSPLINE_ORDER
            ),
        )
        self.epsilon = (self.grid_size - 1) / (self.grid_range[1] - self.grid_range[0])

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size + self._APPROXIMATED_BSPLINE_ORDER

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distance = x.unsqueeze(dim=-1) - self.grid
        return torch.exp(-1.0 * (self.epsilon * distance) ** 2)


def _initialize_grid(
    grid_size: int, grid_range: tuple[float, float], spline_order: int
) -> torch.Tensor:
    min_value, max_value = grid_range
    scale = (max_value - min_value) / grid_size
    grid = torch.linspace(
        start=-spline_order / 2,
        end=grid_size - spline_order / 2,
        steps=grid_size + spline_order,
        dtype=torch.get_default_dtype(),
    )
    return grid * scale + min_value
