from functools import partial
from typing import Literal

import torch
from torch.nn.functional import conv1d, silu
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

    def residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.weight_residual is not None
        return conv1d(
            input=silu(x),
            weight=self.weight_residual,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def spline_forward(self, x: torch.Tensor) -> torch.Tensor:
        basis: torch.Tensor = self.basis(x)

        basis = _flatten_input_channels_and_basis(basis)
        coeff = _flatten_input_channels_and_basis(self.weighted_coefficients)

        return conv1d(
            input=basis,
            weight=coeff,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * self.basis.num_basis_functions,  # TODO: WRONG!!!
        )


def _flatten_input_channels_and_basis(x: torch.Tensor) -> torch.Tensor:
    return torch.movedim(x, source=-1, destination=-2).flatten(start_dim=-3, end_dim=-2)
