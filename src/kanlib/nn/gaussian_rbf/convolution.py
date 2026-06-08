from functools import partial

import torch

from kanlib.nn.base_modules.convolution import Conv1dBase, Conv2dBase, PaddingStr

from .gaussian_rbf_basis import GaussianRbfBasis


class Conv1d(Conv1dBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        grid_size: int,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | PaddingStr = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        spline_range: tuple[float, float] | torch.Tensor = (-1.0, 1.0),
        use_output_bias: bool = False,
        use_residual_branch: bool = True,
        use_spline_weight: bool = True,
        init_coeff_std: float = 0.1,
        adaptive_grid_margin: float = 0.01,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=partial(
                GaussianRbfBasis, adaptive_grid_margin=adaptive_grid_margin
            ),
            use_output_bias=use_output_bias,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )


class Conv2d(Conv2dBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        grid_size: int,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | PaddingStr = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        spline_range: tuple[float, float] | torch.Tensor = (-1.0, 1.0),
        use_output_bias: bool = False,
        use_residual_branch: bool = True,
        use_spline_weight: bool = True,
        init_coeff_std: float = 0.1,
        adaptive_grid_margin: float = 0.01,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=partial(
                GaussianRbfBasis, adaptive_grid_margin=adaptive_grid_margin
            ),
            use_output_bias=use_output_bias,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )
