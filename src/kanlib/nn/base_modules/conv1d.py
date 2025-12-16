from functools import partial
from typing import Literal

import torch
from torch.nn.init import normal_ as init_normal
from torch.nn.init import ones_ as init_ones
from torch.nn.init import xavier_uniform_ as init_xavier_uniform
from torch.nn.init import zeros_ as init_zeros

from kanlib.nn.kan_base_layer import (
    BasisFactory,
    KANBaseLayer,
    ModuleParamSpecs,
    ParamSpec,
)

type PaddingStr = Literal["same", "valid"]


class Conv1d(KANBaseLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,  # default: 1
        padding: int | PaddingStr,  # default: 0
        dilation: int,  # default: 1
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
            param_shape=(out_channels, in_channels // groups, kernel_size),
            in_feature_dim=1,
            out_feature_dim=0,
            param_specs=ModuleParamSpecs(
                coefficients=ParamSpec(
                    partial(init_normal, mean=0, std=init_coeff_std)
                ),
                weight_spline=ParamSpec(init_ones) if use_spline_weight else None,
                weight_residual=(
                    ParamSpec(init_xavier_uniform) if use_residual_branch else None
                ),
                bias_output=ParamSpec(init_zeros) if use_output_bias else None,
            ),
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=basis_factory,
            spline_input_norm=None,
        )
        assert in_channels % groups == 0
        assert out_channels % groups == 0

    def residual_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def spline_forward(self, x: torch.Tensor) -> torch.Tensor: ...

def _validate_conv_params(in_channels: int, out_channels: int, groups: int, padding: int | PaddingStr) -> None:
    if groups <= 0:
        raise ValueError("Number of groups must be a positive integer")
    if 