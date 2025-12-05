import pytest
import torch

from kanlib.compute_coefficients import compute_coefficients
from kanlib.nn.bspline.bspline_basis import BSplineBasis


@pytest.fixture(params=[1, 2, 3], ids=["1-feat", "2-feat", "3-feat"])
def num_features(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def original_basis(num_features: int) -> BSplineBasis:
    return BSplineBasis(
        grid_size=5,
        spline_range=torch.tensor([[-1, 1]] * num_features),
        spline_order=3,
    )


@pytest.fixture
def target_basis(num_features: int) -> BSplineBasis:
    return BSplineBasis(
        grid_size=10,
        spline_range=torch.tensor([[-1, 1]] * num_features),
        spline_order=3,
    )


@pytest.fixture(params=[2, 3, 4], ids=["2d-coeff", "3d-coeff", "4d-coeff"])
def original_coefficients(
    request: pytest.FixtureRequest, original_basis: BSplineBasis
) -> torch.Tensor:
    ndim = request.param
    return torch.empty(
        *[1] * (ndim - 2),
        original_basis.num_features,
        original_basis.num_basis_functions,
    )


@pytest.mark.parametrize(
    "orig_coeff_shape, orig_values_shape, targ_values_shape",
    [
        ((1, 1), (1, 1, 1), (1, 1)),
        ((1,), (1,), (1,)),
        (((1, 1), (1, 1, 1), (1, 1))),
        ((1, 1), (1, 1), (1,)),
        ((2, 3), (2, 3), (3, 2)),
    ],
    ids=[
        "basis values have different number of dimensions",
        "basis values have less than 2 dimensions",
        "shape of original/target basis values does not match",
        "coefficients have less than 2 dimensions",
        "last two dimensions of original basis values and coefficients does not match",
    ],
)
def test_raises_error_on_invalid_input(
    orig_coeff_shape: tuple[int, ...],
    orig_values_shape: tuple[int, ...],
    targ_values_shape: tuple[int, ...],
) -> None:
    orig_coeff, orig_values, targ_values = map(
        torch.empty, (orig_coeff_shape, orig_values_shape, targ_values_shape)
    )

    with pytest.raises(ValueError):
        _ = compute_coefficients(
            original_coefficients=orig_coeff,
            original_basis_values=orig_values,
            target_basis_values=targ_values,
        )


@pytest.mark.parametrize("input_dim", [1, 2, 3])
def test_computes_coefficients_of_correct_shape(
    input_dim: int,
    original_basis: BSplineBasis,
    target_basis: BSplineBasis,
    original_coefficients: torch.Tensor,
) -> None:
    inputs = torch.empty(*range(1, input_dim), original_basis.num_features)

    original_basis_values = original_basis(inputs)
    target_basis_values = target_basis(inputs)

    target_coefficients = compute_coefficients(
        original_coefficients=original_coefficients,
        original_basis_values=original_basis_values,
        target_basis_values=target_basis_values,
    )

    assert target_coefficients.shape == (
        *original_coefficients.shape[:-1],
        target_basis.num_basis_functions,
    )
