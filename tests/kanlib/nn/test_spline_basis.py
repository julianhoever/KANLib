import pytest
import torch

from kanlib.nn.spline_basis import SplineBasis


class SplineBasisImpl(SplineBasis):
    def __init__(self, num_features: int = 1, grid_size: int = 1) -> None:
        super().__init__(
            num_features=num_features,
            grid_size=grid_size,
            initialize_grid=lambda num_features, grid_size: torch.ones(1),
        )

    @property
    def num_basis_functions(self) -> int:
        raise NotImplementedError()

    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@pytest.mark.parametrize("grid_size", [0, -1, -2])
def test_raises_error_on_grid_size_smaller_one(grid_size: int) -> None:
    with pytest.raises(ValueError):
        _ = SplineBasisImpl(grid_size=grid_size)


@pytest.mark.parametrize("num_features", [1, 2, 4, 5])
def test_forward_raises_error_on_invalid_num_features(
    subtests: pytest.Subtests, num_features: int
) -> None:
    spline_basis = SplineBasisImpl(num_features=3)

    for ndim in [1, 2, 3]:
        with subtests.test():
            inputs = torch.empty(*([1] * (ndim - 1)), num_features)

            with pytest.raises(ValueError):
                _ = spline_basis(inputs)


@pytest.mark.parametrize("num_features", [1, 2, 3])
def test_forward_does_not_raises_error_on_valid_num_features(
    subtests: pytest.Subtests, num_features: int
) -> None:
    spline_basis = SplineBasisImpl(num_features=num_features)

    for ndim in [1, 2, 3]:
        with subtests.test():
            inputs = torch.empty(*([1] * (ndim - 1)), num_features)

            with pytest.raises(NotImplementedError):
                _ = spline_basis(inputs)
