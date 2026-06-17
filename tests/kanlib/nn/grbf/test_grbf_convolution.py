from math import floor

import pytest
import torch
from kanlib.nn.grbf.convolution import Conv1d, Conv2d


@pytest.fixture(params=[2, 3])
def kernel_size(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[1, 2])
def stride(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[0, 2])
def padding(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[1, 2])
def dilation(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[1, 2])
def groups(request: pytest.FixtureRequest) -> int:
    return request.param


def test_conv1d_output_has_correct_shape(
    kernel_size: int, stride: int, padding: int, dilation: int, groups: int
) -> None:
    in_channels, out_channels = 4, 2
    batch, seq_len = 32, 100

    conv = Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=3,
    )

    inputs = torch.ones(batch, in_channels, seq_len)
    outputs = conv(inputs)

    output_seq_len = floor(
        (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )

    assert outputs.shape == (batch, out_channels, output_seq_len)


def test_conv2d_output_has_correct_shape(
    kernel_size: int, stride: int, padding: int, dilation: int, groups: int
) -> None:
    in_channels, out_channels = 4, 2
    batch, h_in, w_in = 32, 100, 150

    conv = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=3,
    )

    inputs = torch.ones(batch, in_channels, h_in, w_in)
    outputs = conv(inputs)

    h_out = floor((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    w_out = floor((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    assert outputs.shape == (batch, out_channels, h_out, w_out)
