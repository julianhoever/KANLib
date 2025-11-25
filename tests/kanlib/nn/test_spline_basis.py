import pytest
import torch

from kanlib.nn.spline_basis import SplineBasis


class SplineBasisImpl(SplineBasis):
    def __init__(
        self,
        num_features: int = 1,
        grid_size: int = 1,
        grid_range: tuple[float, float] | torch.Tensor = (-1, 1),
    ) -> None:
        super().__init__(
            num_features=num_features,
            grid_size=grid_size,
            grid_range=grid_range,
            initialize_grid=lambda num_features, grid_size, grid_range: grid_range,
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


@pytest.mark.parametrize("grid_range_tensor_shape", [(3,), (3, 1), (3, 2, 1)])
def test_raises_error_on_invalid_grid_range_tensor(
    grid_range_tensor_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        _ = SplineBasisImpl(
            num_features=3, grid_range=torch.empty(*grid_range_tensor_shape)
        )


def test_grid_range_tuple_leads_to_correct_grid_range() -> None:
    grid_range = (-2.0, 2.0)
    spline_basis = SplineBasisImpl(num_features=3, grid_range=grid_range)
    assert spline_basis.grid_range.shape == (3, 2)
    assert (spline_basis.grid_range == torch.tensor(grid_range).repeat(3, 1)).all()


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
