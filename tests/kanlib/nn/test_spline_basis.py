from collections.abc import Callable
from typing import Optional

import pytest
import torch

from kanlib.nn.spline_basis import SplineBasis


class SplineBasisImpl(SplineBasis):
    def __init__(
        self,
        spline_range: torch.Tensor,
        initialize_grid: Optional[Callable[[], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            grid_size=1,
            spline_range=spline_range,
            initialize_grid=(lambda: spline_range)
            if initialize_grid is None
            else initialize_grid,
        )

    @property
    def num_basis_functions(self) -> int:
        raise NotImplementedError()

    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@pytest.mark.parametrize("spline_range_shape", [(2,), (3, 1), (4, 3, 2)])
def test_raises_error_on_invalid_spline_range_shape(
    spline_range_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        _ = SplineBasisImpl(spline_range=torch.empty(*spline_range_shape))


@pytest.mark.parametrize(
    "grid_shape",
    argvalues=[(10,), (2, 10), (1, 2, 10)],
    ids=[
        "grid with missing feature dim",
        "grid with wrong number of features",
        "grid with too many dimensions",
    ],
)
def test_raises_error_if_grid_is_initialized_incorrectly(
    grid_shape: tuple[int, ...],
) -> None:
    spline_range_shape = (3, 2)
    with pytest.raises(ValueError):
        _ = SplineBasisImpl(
            spline_range=torch.empty(*spline_range_shape),
            initialize_grid=lambda: torch.empty(*grid_shape),
        )


@pytest.mark.parametrize("num_features", [1, 2, 3])
def test_num_features_determined_correctly(num_features: int) -> None:
    spline_basis = SplineBasisImpl(spline_range=torch.empty(num_features, 2))
    assert spline_basis.num_features == num_features


@pytest.mark.parametrize("input_features", [2, 5])
def test_forward_raises_error_if_number_of_features_not_match(
    input_features: int,
) -> None:
    spline_basis = SplineBasisImpl(spline_range=torch.empty(3, 2))
    inputs = torch.ones(input_features)
    with pytest.raises(ValueError):
        _ = spline_basis(inputs)


@pytest.mark.parametrize(
    "input_shape",
    argvalues=[(5,), (3, 5), (3, 4, 5)],
    ids=["1d-input", "2d-input", "3d-input"],
)
def test_forward_not_raises_error_on_valid_number_of_features(
    input_shape: tuple[int, ...],
) -> None:
    spline_basis = SplineBasisImpl(spline_range=torch.empty(5, 2))
    inputs = torch.empty(*input_shape)
    with pytest.raises(NotImplementedError):
        _ = spline_basis(inputs)


@pytest.mark.parametrize(
    "input_shape",
    argvalues=[(5,), (5, 4), (5, 4, 3)],
    ids=["1d-input", "2d-input", "3d-input"],
)
def test_forward_not_raises_error_on_unfeatured_input_for_single_feature(
    input_shape: tuple[int, ...],
) -> None:
    spline_basis = SplineBasisImpl(spline_range=torch.empty(1, 2))
    inputs = torch.empty(*input_shape)
    with pytest.raises(NotImplementedError):
        _ = spline_basis(inputs)
