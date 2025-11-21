import pytest
import torch

from kanlib.nn.base_modules.linear import LinearBase
from kanlib.nn.spline_basis import SplineBasis


class TestGridRefinement:
    class DummyBasis(SplineBasis):
        def __init__(
            self, num_features: int, grid_size: int, grid_range: tuple[float, float]
        ) -> None:
            super().__init__(
                num_features=num_features,
                grid_size=grid_size,
                grid_range=grid_range,
                initialize_grid=self._initialize_grid,
            )

        @property
        def num_basis_functions(self) -> int:
            return self.grid_size

        def _perform_forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(*x.shape, self.num_basis_functions)

        @staticmethod
        def _initialize_grid(
            num_features: int, grid_size: int, grid_range: tuple[float, float]
        ) -> torch.Tensor:
            return (
                torch.linspace(*grid_range, grid_size)
                .unsqueeze(dim=0)
                .repeat(num_features, 1)
            )

    @pytest.fixture
    def linear(self) -> LinearBase:
        return LinearBase(
            in_features=2,
            out_features=3,
            grid_size=4,
            grid_range=(-1.0, 1.0),
            basis_factory=self.DummyBasis,
            use_output_bias=True,
            use_residual_branch=True,
            use_layer_norm=False,
            use_spline_weight=True,
            init_coeff_std=0.1,
        )

    def test_refine_to_larger_grid_size(self, linear: LinearBase) -> None:
        linear.refine_grid(new_grid_size=8)
        assert linear.basis.grid_size == 8
        assert linear.basis.num_basis_functions == 8
        assert linear.coefficients.shape == (3, 2, 8)

    def test_forward_after_refinement(self, linear: LinearBase) -> None:
        inputs = torch.ones(3, 2)
        linear.refine_grid(new_grid_size=8)
        outputs = linear(inputs)
        assert outputs.shape == (3, 3)
