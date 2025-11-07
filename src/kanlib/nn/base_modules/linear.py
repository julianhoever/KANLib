from functools import partial
from typing import Protocol

import torch
from torch.nn.functional import linear, silu
from torch.nn.init import normal_ as init_normal
from torch.nn.init import ones_ as init_ones
from torch.nn.init import xavier_uniform_ as init_xavier_uniform

from kanlib.nn.kan_module import KANModule, ParamSpec
from kanlib.nn.spline_basis import SplineBasis


class _BasisFactory(Protocol):
    def __call__(
        self, grid_size: int, grid_range: tuple[float, float]
    ) -> SplineBasis: ...


class LinearBase(KANModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        grid_range: tuple[float, float],
        basis_factory: _BasisFactory,
        use_base_branch: bool,
        use_layer_norm: bool,
        use_spline_weight: bool,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            base_shape=(out_features, in_features),
            coefficients=ParamSpec(partial(init_normal, mean=0, std=init_coeff_std)),
            weight_spline=ParamSpec(init_ones) if use_spline_weight else None,
            weight_base=ParamSpec(init_xavier_uniform) if use_base_branch else None,
            grid_size=grid_size,
            grid_range=grid_range,
            basis_factory=basis_factory,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.layer_norm = torch.nn.LayerNorm(in_features) if use_layer_norm else None

    def base_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.weight_base is not None
        return linear(silu(x), self.weight_base)

    def spline_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        basis: torch.Tensor = self.basis(x).flatten(start_dim=-2)
        spline = linear(basis, self.weighted_coefficients.flatten(start_dim=-2))
        return spline
