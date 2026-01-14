from functools import partial

import torch

from kanlib.nn.base_modules.linear import LinearBase

from .gaussian_rbf_basis import GaussianRbfBasis


class Linear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor = (-2.0, 2.0),
        use_output_bias: bool = False,
        use_spline_input_norm: bool = True,
        use_residual_branch: bool = True,
        use_spline_weight: bool = True,
        init_coeff_std: float = 0.1,
        adaptive_grid_margin: float = 0.01,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=partial(
                GaussianRbfBasis, adaptive_grid_margin=adaptive_grid_margin
            ),
            use_output_bias=use_output_bias,
            use_spline_input_norm=use_spline_input_norm,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )
