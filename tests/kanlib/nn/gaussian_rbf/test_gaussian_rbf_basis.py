import pytest
import torch
from torch.testing import assert_close

from kanlib.nn.gaussian_rbf.gaussian_rbf_basis import GaussianRbfBasis


@pytest.fixture
def rbf() -> GaussianRbfBasis:
    return GaussianRbfBasis(grid_size=3, grid_range=(-2.0, 2.0))


def test_grid_has_single_dimension(rbf: GaussianRbfBasis) -> None:
    assert rbf.grid.dim() == 1


def test_grid_maintain_bounds(rbf: GaussianRbfBasis) -> None:
    assert rbf.grid.min() == -2.0
    assert rbf.grid.max() == 2.0


def test_correct_number_of_grid(rbf: GaussianRbfBasis) -> None:
    assert len(rbf.grid) == 3


def test_grid_is_evenly_spaced(rbf: GaussianRbfBasis) -> None:
    min_value, max_value = rbf.grid_range
    expected_spacing = (max_value - min_value) / (rbf.grid_size - 1)
    for i in range(0, len(rbf.grid)):
        assert_close(rbf.grid[i], torch.tensor(min_value + i * expected_spacing))


def test_epsilon_is_single_scalar(rbf: GaussianRbfBasis) -> None:
    assert isinstance(rbf.epsilon, float)


def test_epsilon_is_correctly_calculated(rbf: GaussianRbfBasis) -> None:
    assert_close(torch.tensor(rbf.epsilon), torch.tensor(0.5))


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
    assert rbf(x).shape == (*x.shape, rbf.grid_size)
