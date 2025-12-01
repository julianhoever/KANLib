import pytest
import torch
from torch.testing import assert_close

from kanlib.nn.gaussian_rbf.gaussian_rbf_basis import GaussianRbfBasis


@pytest.fixture
def rbf() -> GaussianRbfBasis:
    return GaussianRbfBasis(grid_size=3, spline_range=torch.tensor([[-1.0, 1.0]]))


def test_grid_has_two_dimensions(rbf: GaussianRbfBasis) -> None:
    assert rbf.grid.ndim == 2


def test_epsilon_has_two_dimensions(rbf: GaussianRbfBasis) -> None:
    assert rbf.epsilon.ndim == 2


def test_correct_number_of_grid(rbf: GaussianRbfBasis) -> None:
    assert rbf.grid.shape[-1] == rbf.num_basis_functions


def test_grid_oversample_is_equal(rbf: GaussianRbfBasis) -> None:
    lower_grid_bound_distance = torch.abs(rbf.spline_range[:, 0] - rbf.grid[:, 0])
    upper_grid_bound_distance = torch.abs(rbf.spline_range[:, 1] - rbf.grid[:, -1])
    assert_close(lower_grid_bound_distance, upper_grid_bound_distance)


@pytest.mark.parametrize(
    "x",
    [
        torch.linspace(-2.0, 2.0, 2).view(2),
        torch.linspace(-2.0, 2.0, 8).view(4, 2),
        torch.linspace(-2.0, 2.0, 24).view(4, 3, 2),
    ],
    ids=["single", "batch", "batch_with_channels"],
)
def test_forward_returns_correct_shape(
    rbf: GaussianRbfBasis, x: torch.Tensor
) -> None:
    assert rbf(x).shape == (*x.shape, rbf.num_basis_functions)

@pytest.mark.parametrize("input_shape", [(1,), (3,), (2, 3), (2, 3, 1), (2, 3, 4, 5)])
def test_forward_works_on_every_input_for_single_feature(
    input_shape: tuple[int, ...],
) -> None:
    basis = GaussianRbfBasis(
        grid_size=10, spline_range=torch.tensor([[-1.0, 1.0]])
    )
    inputs = torch.empty(*input_shape)
    outputs = basis(inputs)
    assert outputs.shape == (*inputs.shape, basis.num_basis_functions)
