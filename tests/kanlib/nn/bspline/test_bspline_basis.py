import pytest
import torch
from scipy.interpolate import BSpline
from torch.testing import assert_close

from kanlib.nn.bspline.bspline_basis import BSplineBasis


def reference_bspline_basis(
    x: torch.Tensor, spline_order: int, grid: torch.Tensor, base_idx: int
) -> torch.Tensor:
    y = BSpline.basis_element(
        t=grid[base_idx : base_idx + spline_order + 2],
        extrapolate=False,
    )(x)
    y = torch.tensor(y, dtype=torch.get_default_dtype())
    y = torch.nan_to_num(y)
    return y


@pytest.fixture(
    params=[
        dict(num_features=1, spline_order=0, grid_size=1, grid_range=(-1, 1)),
        dict(num_features=1, spline_order=1, grid_size=1, grid_range=(-1, 1)),
        dict(num_features=2, spline_order=3, grid_size=10, grid_range=(-1, 1)),
        dict(num_features=2, spline_order=10, grid_size=3, grid_range=(-1, 1)),
        dict(num_features=3, spline_order=3, grid_size=10, grid_range=(-2, 2)),
        dict(num_features=3, spline_order=3, grid_size=10, grid_range=(2, 4)),
    ],
    ids=[
        "num_features=1, spline_order=0, grid_size=1, grid_range=(-1, 1)",
        "num_features=1, spline_order=1, grid_size=1, grid_range=(-1, 1)",
        "num_features=2, spline_order=3, grid_size=10, grid_range=(-1, 1)",
        "num_features=2, spline_order=10, grid_size=3, grid_range=(-1, 1)",
        "num_features=3, spline_order=3, grid_size=10, grid_range=(-2, 2)",
        "num_features=3, spline_order=3, grid_size=10, grid_range=(2, 4)",
    ],
)
def basis(request: pytest.FixtureRequest) -> BSplineBasis:
    return BSplineBasis(**request.param)


@pytest.mark.parametrize("grid_size", [0, -1])
def test_raises_error_on_invalid_grid_size(grid_size: int) -> None:
    with pytest.raises(ValueError):
        _ = BSplineBasis(
            num_features=1, spline_order=0, grid_size=grid_size, grid_range=(-1.0, 1.0)
        )


def test_number_of_basis_functions(basis: BSplineBasis) -> None:
    assert basis.num_basis_functions == basis.grid_size + basis.spline_order


def test_number_of_grid_points(basis: BSplineBasis) -> None:
    assert basis.grid.shape == (
        basis.num_features,
        basis.grid_size + 1 + 2 * basis.spline_order,
    )


def test_grid_is_monotonic_increasing(basis: BSplineBasis) -> None:
    assert (basis.grid[:, 1:] > basis.grid[:, :-1]).all()


def test_forward_returns_correct_number_of_basis_functions(basis: BSplineBasis) -> None:
    inputs = torch.ones(3, basis.num_features)
    outputs = basis(inputs)
    assert outputs.shape == (*inputs.shape, basis.num_basis_functions)


@pytest.mark.parametrize("batched", [True, False])
def test_can_handle_batched_and_unbatched_inputs(
    basis: BSplineBasis, batched: bool
) -> None:
    inputs = torch.ones((3, basis.num_features) if batched else (basis.num_features,))
    outputs = basis(inputs)
    assert outputs.shape == (*inputs.shape, basis.num_basis_functions)


def test_bspline_basis_is_equivalent_to_scipy(basis: BSplineBasis) -> None:
    if basis.spline_order == 0:
        pytest.skip(
            "Skipping comparison for B-splines of order 0 as scipy seems to output"
            "a wrong value in this case. (The last value of the output differs.)"
        )

    gmin, gmax = basis.grid_range.unbind(dim=-1)

    inputs = torch.cat(
        [
            torch.linspace(gmin[feat_idx], gmax[feat_idx], 1000).unsqueeze(-1)
            for feat_idx in range(basis.num_features)
        ],
        dim=-1,
    )

    outputs = basis(inputs)

    for feat_idx in range(basis.num_features):
        for base_idx in range(basis.num_basis_functions):
            reference_outputs = reference_bspline_basis(
                x=inputs[:, feat_idx],
                spline_order=basis.spline_order,
                grid=basis.grid[feat_idx],
                base_idx=base_idx,
            )
            actual_outputs = outputs[:, feat_idx, base_idx]
            assert_close(actual_outputs, reference_outputs)
