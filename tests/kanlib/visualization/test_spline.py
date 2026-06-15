import numpy as np
import pytest
import torch
from torch.testing import assert_close

from kanlib.nn.bspline.convolution import Conv1d, Conv2d
from kanlib.nn.bspline.linear import Linear
from kanlib.nn.gaussian_rbf.linear import Linear as GaussianRbfLinear
from kanlib.visualization import SplineExtractor, spline_curve


def test_spline_curve_returns_1d_numpy_arrays() -> None:
    layer = Linear(in_features=2, out_features=3, grid_size=4, spline_order=3)

    x, y = spline_curve(layer, (1, 0), num_points=128)

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.shape == (128,)
    assert y.shape == (128,)


def test_spline_curve_uses_1000_points_by_default() -> None:
    layer = Linear(in_features=1, out_features=1, grid_size=4, spline_order=3)

    x, y = spline_curve(layer, (0, 0))

    assert x.shape == (1000,)
    assert y.shape == (1000,)


def test_spline_curve_x_spans_the_input_features_spline_range() -> None:
    spline_range = torch.tensor([[-2.0, 2.0], [3.0, 7.0]])
    layer = Linear(
        in_features=2,
        out_features=1,
        grid_size=4,
        spline_order=3,
        spline_range=spline_range,
    )

    x, _ = spline_curve(layer, (0, 1), num_points=50)

    assert x[0] == pytest.approx(3.0)
    assert x[-1] == pytest.approx(7.0)


def test_spline_curve_matches_forward_for_linear_single_edge() -> None:
    torch.manual_seed(0)
    layer = Linear(
        in_features=1,
        out_features=1,
        grid_size=5,
        spline_order=3,
        use_residual_branch=False,
        use_output_bias=False,
    )

    x, y = spline_curve(layer, (0, 0), num_points=40)

    inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=-1)
    expected = layer(inputs).squeeze(dim=-1).detach().numpy()
    assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_spline_curve_reflects_weight_spline() -> None:
    torch.manual_seed(1)
    layer = Linear(
        in_features=2,
        out_features=2,
        grid_size=4,
        spline_order=3,
        use_spline_weight=True,
    )

    weight_spline = layer.weight_spline
    assert weight_spline is not None

    with torch.no_grad():
        weight_spline.fill_(1.0)
    _, y1 = spline_curve(layer, (1, 0), num_points=30)

    with torch.no_grad():
        weight_spline.fill_(2.0)
    _, y2 = spline_curve(layer, (1, 0), num_points=30)

    assert_close(y2, 2.0 * y1, rtol=1e-5, atol=1e-6)


def test_spline_curve_matches_forward_for_conv1d_single_edge() -> None:
    torch.manual_seed(0)
    conv = Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        grid_size=5,
        spline_order=3,
        use_residual_branch=False,
        use_output_bias=False,
    )

    x, y = spline_curve(conv, (0, 0, 0), num_points=30)

    inputs = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 1)
    expected = conv(inputs).reshape(-1).detach().numpy()
    assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_spline_curve_matches_forward_for_conv2d_single_edge() -> None:
    torch.manual_seed(0)
    conv = Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        grid_size=5,
        spline_order=3,
        use_residual_branch=False,
        use_output_bias=False,
    )

    x, y = spline_curve(conv, (0, 0, 0, 0), num_points=20)

    inputs = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 1, 1)
    expected = conv(inputs).reshape(-1).detach().numpy()
    assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_spline_curve_selects_grouped_input_channel_for_conv() -> None:
    spline_range = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    conv = Conv1d(
        in_channels=4,
        out_channels=4,
        kernel_size=1,
        groups=2,
        grid_size=4,
        spline_order=3,
        spline_range=spline_range,
    )

    x, _ = spline_curve(conv, (2, 1, 0), num_points=20)
    assert x[0] == pytest.approx(3.0)
    assert x[-1] == pytest.approx(4.0)

    x, _ = spline_curve(conv, (0, 0, 0), num_points=20)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)


def test_spline_curve_uses_provided_spline_extractor() -> None:
    class FixedFeatureExtractor(SplineExtractor):
        @property
        def input_feature_index(self) -> int:
            return 0

    spline_range = torch.tensor([[0.0, 1.0], [5.0, 6.0]])
    layer = Linear(
        in_features=2,
        out_features=2,
        grid_size=4,
        spline_order=3,
        spline_range=spline_range,
    )

    x_default, _ = spline_curve(layer, (0, 1), num_points=20)
    assert x_default[0] == pytest.approx(5.0)

    x_override, _ = spline_curve(
        layer, (0, 1), num_points=20, spline_extractor=FixedFeatureExtractor
    )
    assert x_override[0] == pytest.approx(0.0)
    assert x_override[-1] == pytest.approx(1.0)


def test_spline_curve_matches_forward_for_gaussian_rbf_linear() -> None:
    torch.manual_seed(0)
    layer = GaussianRbfLinear(
        in_features=1,
        out_features=1,
        grid_size=6,
        use_residual_branch=False,
        use_output_bias=False,
    )

    x, y = spline_curve(layer, (0, 0), num_points=30)

    inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=-1)
    expected = layer(inputs).squeeze(dim=-1).detach().numpy()
    assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_spline_curve_raises_for_unsupported_layer() -> None:
    with pytest.raises(TypeError):
        spline_curve(torch.nn.Identity(), (0, 0))  # type: ignore


@pytest.mark.parametrize("spline_index", [(0,), (0, 0, 0)])
def test_spline_curve_raises_for_wrong_length_spline_index(
    spline_index: tuple[int, ...],
) -> None:
    layer = Linear(in_features=2, out_features=3, grid_size=4, spline_order=3)

    with pytest.raises(ValueError, match="must have length 2"):
        spline_curve(layer, spline_index)


def test_spline_curve_raises_for_negative_spline_index() -> None:
    layer = Linear(in_features=2, out_features=3, grid_size=4, spline_order=3)

    with pytest.raises(ValueError, match="negative"):
        spline_curve(layer, (-1, 0))


def test_spline_curve_raises_for_out_of_bounds_spline_index() -> None:
    layer = Linear(in_features=2, out_features=3, grid_size=4, spline_order=3)

    with pytest.raises(ValueError, match="out of bounds"):
        spline_curve(layer, (3, 0))


def test_spline_curve_checks_bounds_on_all_index_dimensions() -> None:
    conv = Conv1d(
        in_channels=1, out_channels=1, kernel_size=3, grid_size=4, spline_order=3
    )

    with pytest.raises(ValueError, match="out of bounds"):
        spline_curve(conv, (0, 0, 3))


def test_spline_curve_accepts_maximum_valid_spline_index() -> None:
    layer = Linear(in_features=2, out_features=3, grid_size=4, spline_order=3)

    x, y = spline_curve(layer, (2, 1), num_points=10)

    assert x.shape == (10,)
    assert y.shape == (10,)
