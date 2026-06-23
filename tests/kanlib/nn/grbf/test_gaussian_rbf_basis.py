import pytest
import torch
from torch.testing import assert_close

from kanlib.nn.grbf.gaussian_rbf_basis import GaussianRbfBasis, _compute_epsilon


@pytest.fixture
def rbf() -> GaussianRbfBasis:
    return GaussianRbfBasis(
        grid_size=3, spline_range=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    )


def test_grid_has_two_dimensions(rbf: GaussianRbfBasis) -> None:
    assert rbf.grid.ndim == 2


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
def test_forward_returns_correct_shape(rbf: GaussianRbfBasis, x: torch.Tensor) -> None:
    assert rbf(x).shape == (*x.shape, rbf.num_basis_functions)


def test_forward_with_explicit_grid_matches_cached_grid(rbf: GaussianRbfBasis) -> None:
    x = torch.linspace(-2.0, 2.0, 16).view(-1, 2)
    assert_close(rbf(x), rbf(x, grid=rbf.grid))


def test_new_epsilon_after_update(rbf: GaussianRbfBasis) -> None:
    old_epsilon = rbf.epsilon.clone()
    x = torch.linspace(-3.0, 3.0, 40).view(-1, 2)
    rbf.update_grid(rbf.grid_update_from_samples(x))
    assert_close(rbf.epsilon, _compute_epsilon(rbf.grid))
    assert not torch.allclose(rbf.epsilon, old_epsilon)
