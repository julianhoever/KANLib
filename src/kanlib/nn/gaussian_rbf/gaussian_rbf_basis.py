from functools import partial

import torch

from kanlib.nn.spline_basis import SplineBasis

_APPROXIMATED_BSPLINE_ORDER = 3


class GaussianRbfBasis(SplineBasis):
    def __init__(
        self, num_features: int, grid_size: int, grid_range: tuple[float, float]
    ) -> None:
        super().__init__(
            num_features=num_features,
            grid_size=grid_size,
            initialize_grid=partial(_initialize_grid, grid_range=grid_range),
        )
        self.epsilon = float(
            self.num_basis_functions / (self.grid.max() - self.grid.min())
        )

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size + _APPROXIMATED_BSPLINE_ORDER

    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor:
        distance = x.unsqueeze(dim=-1) - self.grid
        return torch.exp(-1.0 * (self.epsilon * distance) ** 2)


def _initialize_grid(
    num_features: int, grid_size: int, grid_range: tuple[float, float]
) -> torch.Tensor:
    min_value, max_value = grid_range
    scale = (max_value - min_value) / grid_size
    grid = torch.linspace(
        start=-_APPROXIMATED_BSPLINE_ORDER / 2,
        end=grid_size + _APPROXIMATED_BSPLINE_ORDER / 2,
        steps=grid_size + _APPROXIMATED_BSPLINE_ORDER,
        dtype=torch.get_default_dtype(),
    ).repeat(num_features, 1)
    return grid * scale + min_value
