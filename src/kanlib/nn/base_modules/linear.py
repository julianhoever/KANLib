import torch
from torch.nn.functional import linear, silu

from kanlib.nn.kan_base_layer import (
    BasisSpec,
    KANBaseLayer,
    LayerSpec,
    default_param_specs,
)


class LinearBase(KANBaseLayer):
    def __init__(
        self,
        basis_spec: BasisSpec,
        in_features: int,
        out_features: int,
        use_output_bias: bool,
        use_residual_branch: bool,
        use_spline_weight: bool,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            layer_spec=LayerSpec(
                input_features=in_features,
                param_shape=(out_features, in_features),
                in_feat_dim=1,
                out_feat_dim=0,
            ),
            param_specs=default_param_specs(
                use_spline_weight=use_spline_weight,
                use_residual_branch=use_residual_branch,
                use_output_bias=use_output_bias,
                init_coeff_std=init_coeff_std,
            ),
            basis_spec=basis_spec,
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
