from functools import partial

import torch

from kanlib.nn.base_modules.linear import LinearBase
from kanlib.nn.kan_base_layer import BasisSpec

from .gaussian_rbf_basis import GaussianRbfBasis


class Linear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor = (-1.0, 1.0),
        use_output_bias: bool = False,
        use_residual_branch: bool = True,
        use_spline_weight: bool = True,
        init_coeff_std: float = 0.1,
        adaptive_grid_margin: float = 0.01,
    ) -> None:
        super().__init__(
            basis_spec=BasisSpec(
                basis_factory=partial(
                    GaussianRbfBasis, adaptive_grid_margin=adaptive_grid_margin
                ),
                grid_size=grid_size,
                spline_range=spline_range,
            ),
            in_features=in_features,
            out_features=out_features,
            use_output_bias=use_output_bias,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )
