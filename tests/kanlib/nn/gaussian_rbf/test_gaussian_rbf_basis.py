import pytest
import torch
from torch.testing import assert_close

from kanlib.nn.gaussian_rbf.gaussian_rbf_basis import GaussianRbfBasis


@pytest.fixture
def rbf() -> GaussianRbfBasis:
    return GaussianRbfBasis(grid_size=3, grid_range=(-1.0, 1.0))


def test_grid_has_single_dimension(rbf: GaussianRbfBasis) -> None:
    assert rbf.grid.dim() == 1


def test_correct_number_of_grid(rbf: GaussianRbfBasis) -> None:
    assert len(rbf.grid) == rbf.num_basis_functions


def test_epsilon_is_single_scalar(rbf: GaussianRbfBasis) -> None:
    assert isinstance(rbf.epsilon, float)


def test_grid_oversample_is_equal(rbf: GaussianRbfBasis) -> None:
    lower_grid_bound_distance = torch.abs(rbf.grid_range[0] - rbf.grid[0])
    upper_grid_bound_distance = torch.abs(rbf.grid_range[1] - rbf.grid[-1])
    assert_close(lower_grid_bound_distance, upper_grid_bound_distance)


@pytest.mark.parametrize(
    "x",
    [
        torch.linspace(-2.0, 2.0, 16).view(16),
        torch.linspace(-2.0, 2.0, 16).view(2, 8),
        torch.linspace(-2.0, 2.0, 16).view(4, 2, 2),
    ],
    ids=["single", "batch", "batch_with_channels"],
)
def test_forward_pass_returns_correct_shape(
    rbf: GaussianRbfBasis, x: torch.Tensor
) -> None:
    assert rbf(x).shape == (*x.shape, rbf.num_basis_functions)
