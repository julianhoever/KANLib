from functools import partial
from typing import Literal, Protocol, cast

import torch
from torch.nn.functional import conv1d, conv2d, silu

from kanlib.nn.kan_base_layer import BasisFactory, KANBaseLayer, default_param_specs

type PaddingStr = Literal["same", "valid"]
type SupportedConvDim = Literal[1, 2]


class _ConvBase(KANBaseLayer):
    def __init__(
        self,
        conv_dim: SupportedConvDim,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...] | PaddingStr,
        dilation: int | tuple[int, ...],
        groups: int,
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor,
        basis_factory: BasisFactory,
        use_output_bias: bool,
        use_residual_branch: bool,
        use_spline_weight: bool,
        init_coeff_std: float,
    ) -> None:
        kernel_size = _to_tuple(kernel_size, conv_dim)

        super().__init__(
            param_shape=(out_channels, in_channels // groups, *kernel_size),
            in_feature_dim=1,
            out_feature_dim=0,
            param_specs=default_param_specs(
                use_spline_weight=use_spline_weight,
                use_residual_branch=use_residual_branch,
                use_output_bias=use_output_bias,
                init_coeff_std=init_coeff_std,
            ),
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=basis_factory,
        )

        if groups <= 0:
            raise ValueError("Number of groups must be a positive integer.")
        if in_channels % groups != 0:
            raise ValueError("`in_channels` must be divisible by `groups`.")
        if out_channels % groups != 0:
            raise ValueError("`out_channels` must be divisible by `groups`.")
        if padding == "same" and stride != 1:
            raise ValueError("Same padding is not supported for stride != 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self._conv = partial(
            _determine_convolution(conv_dim),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        self._conv_dim = conv_dim
        self._input_channel_dim = -(self._conv_dim + 1)

    def residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.weight_residual is not None
        return self._conv(input=silu(x), weight=self.weight_residual)

    def spline_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.movedim(self._input_channel_dim, -1)

        basis: torch.Tensor = (
            self.basis(x).flatten(start_dim=-2).movedim(-1, self._input_channel_dim)
        )

        coeff = self.weighted_coefficients.movedim(-1, self.in_feature_dim + 1).flatten(
            start_dim=self.in_feature_dim, end_dim=self.in_feature_dim + 1
        )

        return self._conv(input=basis, weight=coeff)


def _to_tuple(x: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if isinstance(x, tuple):
        return x
    return tuple([x] * ndim)


class _ConvolutionFunc(Protocol):
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...] | PaddingStr,
        dilation: int | tuple[int, ...],
        groups: int,
    ) -> torch.Tensor: ...


def _determine_convolution(conv_dim: SupportedConvDim) -> _ConvolutionFunc:
    match conv_dim:
        case 1:
            conv = conv1d
        case 2:
            conv = conv2d
    return cast(_ConvolutionFunc, conv)


class Conv1dBase(_ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int],  # default: 1
        padding: int | tuple[int] | PaddingStr,  # default: 0
        dilation: int | tuple[int],  # default: 1
        groups: int,  # default: 1
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor,
        basis_factory: BasisFactory,
        use_output_bias: bool,
        use_residual_branch: bool,
        use_spline_weight: bool,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            conv_dim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=basis_factory,
            use_output_bias=use_output_bias,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )


class Conv2dBase(_ConvBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],  # default: 1
        padding: int | tuple[int, int] | PaddingStr,  # default: 0
        dilation: int | tuple[int, int],  # default: 1
        groups: int,  # default: 1
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor,
        basis_factory: BasisFactory,
        use_output_bias: bool,
        use_residual_branch: bool,
        use_spline_weight: bool,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            conv_dim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=basis_factory,
            use_output_bias=use_output_bias,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )
