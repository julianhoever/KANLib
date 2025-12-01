from functools import partial

import pytest
import torch

from kanlib.nn.bspline.bspline_basis import BSplineBasis
from kanlib.refinement import compute_refined_coefficients


class TestComputeRefinedCoefficients:
    @pytest.fixture
    def spline_order(self) -> int:
        return 3

    @pytest.fixture(params=[1, 2, 3], ids=["1-feat", "2-feat", "3-feat"])
    def num_features(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def coarse_grid_size(self) -> int:
        return 5

    @pytest.fixture(params=[2, 3, 4], ids=["2d-coeff", "3d-coeff", "4d-coeff"])
    def coeff_coarse(
        self,
        request: pytest.FixtureRequest,
        coarse_grid_size: int,
        spline_order: int,
        num_features: int,
    ) -> torch.Tensor:
        ndim = request.param
        return torch.arange(
            1, coarse_grid_size + spline_order + 1, dtype=torch.get_default_dtype()
        ).repeat(*range(1, ndim - 2), num_features, 1)

    @pytest.fixture(params=[10, 15, 20])
    def fine_grid_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def create_basis(
        self,
        num_features: int,
        spline_order: int,
        grid_size: int,
    ) -> BSplineBasis:
        return BSplineBasis(
            spline_order=spline_order,
            grid_size=grid_size,
            spline_range=torch.tensor([[-1.0, 1.0]] * num_features),
        )

    @pytest.mark.parametrize("grid_size", [1, 4])
    def test_raises_error_on_invalid_fine_grid_size(self, grid_size: int) -> None:
        create_basis = partial(self.create_basis, num_features=1, spline_order=3)
        basis_coarse = create_basis(grid_size=4)
        basis_fine = create_basis(grid_size=grid_size)
        with pytest.raises(ValueError):
            _ = compute_refined_coefficients(
                basis_coarse=basis_coarse,
                basis_fine=basis_fine,
                coeff_coarse=torch.ones(basis_coarse.num_basis_functions, 1),
            )

    def test_grid_size_equals_target_grid_size(
        self,
        coeff_coarse: torch.Tensor,
        coarse_grid_size: int,
        fine_grid_size: int,
        num_features: int,
        spline_order: int,
    ) -> None:
        create_basis = partial(
            self.create_basis, num_features=num_features, spline_order=spline_order
        )
        basis_coarse = create_basis(grid_size=coarse_grid_size)
        basis_fine = create_basis(grid_size=fine_grid_size)

        coeff_fine = compute_refined_coefficients(
            basis_coarse=basis_coarse,
            basis_fine=basis_fine,
            coeff_coarse=coeff_coarse,
        )

        assert coeff_fine.shape == (
            *coeff_coarse.shape[:-1],
            basis_fine.num_basis_functions,
        )
