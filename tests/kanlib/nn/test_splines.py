import pytest
import torch

from kanlib.nn.bspline.bspline_basis import BSplineBasis
from kanlib.spline import compute_refined_coefficients


class TestComputeRefinedCoefficients:
    @pytest.fixture(
        params=[
            torch.arange(1, 8, dtype=torch.get_default_dtype()),
            torch.arange(1, 8, dtype=torch.get_default_dtype()).repeat(3, 1),
            torch.arange(1, 8, dtype=torch.get_default_dtype()).repeat(2, 3, 1),
            torch.arange(1, 8, dtype=torch.get_default_dtype()).repeat(1, 2, 3, 1),
        ],
        ids=["1d-coeff", "2d-coeff", "3d-coeff", "4d-coeff"],
    )
    def coeff_coarse(self, request: pytest.FixtureRequest) -> torch.Tensor:
        return request.param

    @pytest.fixture(params=[5, 7, 9])
    def fine_grid_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def create_basis(
        self, grid_size: int, grid_range: tuple[float, float]
    ) -> BSplineBasis:
        return BSplineBasis(spline_order=3, grid_size=grid_size, grid_range=grid_range)

    def test_grid_size_equals_target_grid_size(
        self, coeff_coarse: torch.Tensor, fine_grid_size: int
    ) -> None:
        basis_coarse = self.create_basis(grid_size=4, grid_range=(-1.0, 1.0))
        basis_fine = self.create_basis(grid_size=fine_grid_size, grid_range=(-1.0, 1.0))

        coeff_fine = compute_refined_coefficients(
            basis_coarse=basis_coarse, basis_fine=basis_fine, coeff_coarse=coeff_coarse
        )

        assert coeff_fine.shape == (
            *coeff_coarse.shape[:-1],
            basis_fine.num_basis_functions,
        )

    @pytest.mark.parametrize("grid_size", [1, 4])
    def test_raises_error_on_invalid_fine_grid_size(self, grid_size: int) -> None:
        basis_coarse = self.create_basis(grid_size=4, grid_range=(-1.0, 1.0))
        basis_fine = self.create_basis(grid_size=grid_size, grid_range=(-1.0, 1.0))
        with pytest.raises(ValueError):
            _ = compute_refined_coefficients(
                basis_coarse=basis_coarse,
                basis_fine=basis_fine,
                coeff_coarse=torch.ones(basis_coarse.num_basis_functions),
            )
