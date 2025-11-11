from functools import partial

import torch
from torch.nn.functional import linear, silu
from torch.nn.init import normal_ as init_normal
from torch.nn.init import ones_ as init_ones
from torch.nn.init import xavier_uniform_ as init_xavier_uniform
from torch.nn.init import zeros_ as init_zeros

from kanlib.nn.kan_module import BasisFactory, KANModule, ParamSpec


class LinearBase(KANModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        grid_range: tuple[float, float],
        basis_factory: BasisFactory,
        use_output_bias: bool,
        use_layer_norm: bool,
        use_residual_branch: bool,
        use_spline_weight: bool,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            param_shape=(out_features, in_features),
            coefficients=ParamSpec(partial(init_normal, mean=0, std=init_coeff_std)),
            weight_spline=ParamSpec(init_ones) if use_spline_weight else None,
            weight_residual=(
                ParamSpec(init_xavier_uniform) if use_residual_branch else None
            ),
            bias_output=ParamSpec(init_zeros) if use_output_bias else None,
            grid_size=grid_size,
            grid_range=grid_range,
            basis_factory=basis_factory,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.layer_norm = torch.nn.LayerNorm(in_features) if use_layer_norm else None

    def residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.weight_residual is not None
        return linear(silu(x), self.weight_residual)

    def spline_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        basis: torch.Tensor = self.basis(x).flatten(start_dim=-2)
        spline = linear(basis, self.weighted_coefficients.flatten(start_dim=-2))
        return spline
