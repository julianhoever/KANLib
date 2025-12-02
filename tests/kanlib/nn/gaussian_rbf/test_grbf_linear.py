import pytest
import torch

from kanlib.nn.gaussian_rbf.linear import Linear


@pytest.fixture
def grid_size() -> int:
    return 3


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture(params=[1, 2], ids=["in_1d", "in_2d"])
def in_features(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[1, 2], ids=["out_1d", "out_2d"])
def out_features(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[False, True], ids=["without_norm", "normalized"])
def normalize_spline_inputs(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[False, True], ids=["without_residual", "with_residual"])
def use_residual_branch(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(
    params=[False, True],
    ids=["without_spline_weight", "with_spline_weight"],
)
def use_spline_weight(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[False, True], ids=["unbatched", "batched"])
def inputs(
    request: pytest.FixtureRequest, in_features: int, batch_size: int
) -> torch.Tensor:
    batched = request.param
    return torch.ones(
        (batch_size, in_features) if batched else (in_features,), dtype=torch.float32
    )


@pytest.fixture
def linear(
    in_features: int,
    out_features: int,
    grid_size: int,
    normalize_spline_inputs: bool,
    use_residual_branch: bool,
    use_spline_weight: bool,
) -> Linear:
    return Linear(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        normalize_spline_inputs=normalize_spline_inputs,
        use_residual_branch=use_residual_branch,
        use_spline_weight=use_spline_weight,
    )


def test_forward_pass_returns_correct_shape(
    linear: Linear, inputs: torch.Tensor
) -> None:
    output = linear(inputs)
    batched = inputs.dim() == 2
    target_shape = (
        (inputs.shape[0], linear.out_features) if batched else (linear.out_features,)
    )
    assert output.shape == target_shape
