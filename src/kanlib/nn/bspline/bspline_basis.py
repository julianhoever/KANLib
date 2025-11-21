from functools import partial

import torch

from kanlib.nn.spline_basis import SplineBasis


class BSplineBasis(SplineBasis):
    def __init__(
        self,
        num_features: int,
        spline_order: int,
        grid_size: int,
        grid_range: tuple[float, float],
    ) -> None:
        super().__init__(
            num_features=num_features,
            grid_size=grid_size,
            grid_range=grid_range,
            initialize_grid=partial(_initialize_grid, spline_order=spline_order),
        )
        self.spline_order = spline_order

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size + self.spline_order

    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of B-spline basis functions is based on:
        https://github.com/Blealtan/efficient-kan/blob/7b6ce1c87f18c8bc90c208f6b494042344216b11/src/efficient_kan/kan.py#L78-L111
        """
        g = self.grid
        x = x.unsqueeze(dim=-1)

        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).type(x.dtype)

        for k in range(1, self.spline_order + 1):
            b1 = (x - g[:, : -(k + 1)]) / (g[:, k:-1] - g[:, : -(k + 1)]) * b[..., :-1]
            b2 = (g[:, k + 1 :] - x) / (g[:, k + 1 :] - g[:, 1:-k]) * b[..., 1:]
            b = b1 + b2

        return b


def _initialize_grid(
    num_features: int,
    grid_size: int,
    grid_range: tuple[float, float],
    spline_order: int,
) -> torch.Tensor:
    min_value, max_value = grid_range
    scale = (max_value - min_value) / grid_size
    grid = torch.arange(
        start=-spline_order,
        end=grid_size + spline_order + 1,
        dtype=torch.get_default_dtype(),
    ).repeat(num_features, 1)
    return grid * scale + min_value
