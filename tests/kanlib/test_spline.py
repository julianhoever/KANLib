from functools import partial

import pytest
import torch

from kanlib.nn.bspline.bspline_basis import BSplineBasis
from kanlib.spline import compute_refined_coefficients


class TestComputeRefinedCoefficients:
    @pytest.fixture(
        params=[
            torch.arange(1, 8, dtype=torch.get_default_dtype()).repeat(3, 1),
            torch.arange(1, 8, dtype=torch.get_default_dtype()).repeat(2, 3, 1),
            torch.arange(1, 8, dtype=torch.get_default_dtype()).repeat(1, 2, 3, 1),
        ],
        ids=["2d-coeff", "3d-coeff", "4d-coeff"],
    )
    def coeff_coarse_three_feat(self, request: pytest.FixtureRequest) -> torch.Tensor:
        return request.param

    @pytest.fixture
    def coarse_grid_size(self) -> int:
        return 4

    @pytest.fixture(params=[5, 7, 9])
    def fine_grid_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def create_basis(self, num_features: int, grid_size: int) -> BSplineBasis:
        return BSplineBasis(
            num_features=num_features,
            spline_order=3,
            grid_size=grid_size,
            grid_range=(-1.0, 1.0),
        )

    def test_grid_size_equals_target_grid_size(
        self,
        coeff_coarse_three_feat: torch.Tensor,
        coarse_grid_size: int,
        fine_grid_size: int,
    ) -> None:
        create_basis = partial(self.create_basis, num_features=3)
        basis_coarse = create_basis(grid_size=coarse_grid_size)
        basis_fine = create_basis(grid_size=fine_grid_size)

        coeff_fine = compute_refined_coefficients(
            basis_coarse=basis_coarse,
            basis_fine=basis_fine,
            coeff_coarse=coeff_coarse_three_feat,
        )

        assert coeff_fine.shape == (
            *coeff_coarse_three_feat.shape[:-1],
            basis_fine.num_basis_functions,
        )

    @pytest.mark.parametrize("grid_size", [1, 4])
    def test_raises_error_on_invalid_fine_grid_size(self, grid_size: int) -> None:
        basis_coarse = self.create_basis(num_features=1, grid_size=4)
        basis_fine = self.create_basis(num_features=1, grid_size=grid_size)
        with pytest.raises(ValueError):
            _ = compute_refined_coefficients(
                basis_coarse=basis_coarse,
                basis_fine=basis_fine,
                coeff_coarse=torch.ones(basis_coarse.num_basis_functions, 1),
            )
