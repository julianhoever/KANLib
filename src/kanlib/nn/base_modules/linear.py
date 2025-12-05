from functools import partial
from typing import Any, Optional

import torch
from torch.nn.functional import linear, silu
from torch.nn.init import normal_ as init_normal
from torch.nn.init import ones_ as init_ones
from torch.nn.init import xavier_uniform_ as init_xavier_uniform
from torch.nn.init import zeros_ as init_zeros

from kanlib.nn.kan_module import BasisFactory, KANModule, ModuleParamSpecs, ParamSpec


class LinearBase(KANModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor,
        basis_factory: BasisFactory,
        use_output_bias: bool,
        use_spline_input_norm: bool,
        adaptive_grid_kwargs: Optional[dict[str, Any]],
        use_residual_branch: bool,
        use_spline_weight: bool,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            param_shape=(out_features, in_features),
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
            spline_input_norm=torch.nn.LayerNorm(
                normalized_shape=in_features, elementwise_affine=False, bias=False
            )
            if use_spline_input_norm
            else None,
            adaptive_grid_kwargs=adaptive_grid_kwargs,
        )
        self.in_features = in_features
        self.out_features = out_features

    def residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.weight_residual is not None
        return linear(silu(x), self.weight_residual)

    def spline_forward(self, x: torch.Tensor) -> torch.Tensor:
        basis: torch.Tensor = self.basis(x).flatten(start_dim=-2)
        spline = linear(basis, self.weighted_coefficients.flatten(start_dim=-2))
        return spline
