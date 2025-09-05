import pytest
import torch

from kanlib.nn.bspline.linear import Linear


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
def use_layer_norm(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[False, True], ids=["without_base", "with_base"])
def use_base_branch(request: pytest.FixtureRequest) -> bool:
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
    use_layer_norm: bool,
    use_base_branch: bool,
    use_spline_weight: bool,
) -> Linear:
    return Linear(
        in_features=in_features,
        out_features=out_features,
        spline_order=3,
        grid_size=grid_size,
        use_layer_norm=use_layer_norm,
        use_base_branch=use_base_branch,
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
