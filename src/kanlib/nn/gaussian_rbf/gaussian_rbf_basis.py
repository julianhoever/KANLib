import torch

from kanlib.nn.spline_basis import SplineBasis


class GaussianRbfBasis(SplineBasis):
    def __init__(self, grid_size: int, grid_range: tuple[float, float]) -> None:
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            initialize_grid=_initialize_grid,
        )
        self.epsilon = (self.grid_size - 1) / (self.grid_range[1] - self.grid_range[0])

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distance = x.unsqueeze(dim=-1) - self.grid
        return torch.exp(-1.0 * (self.epsilon * distance) ** 2)


def _initialize_grid(grid_size: int, grid_range: tuple[float, float]) -> torch.Tensor:
    return torch.linspace(*grid_range, grid_size)
