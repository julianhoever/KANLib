from functools import partial

import torch

from kanlib.nn.base_modules.linear import LinearBase

from .bspline_basis import BSplineBasis


class Linear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int,
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor = (-1.0, 1.0),
        use_output_bias: bool = False,
        use_spline_input_norm: bool = False,
        use_residual_branch: bool = True,
        use_spline_weight: bool = True,
        init_coeff_std: float = 0.1,
        adaptive_grid_margin: float = 0.01,
        adaptive_grid_uniform_fraction: float = 0.02,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            spline_range=spline_range,
            basis_factory=partial(
                BSplineBasis,
                spline_order=spline_order,
                adaptive_grid_margin=adaptive_grid_margin,
                adaptive_grid_uniform_fraction=adaptive_grid_uniform_fraction,
            ),
            use_output_bias=use_output_bias,
            use_spline_input_norm=use_spline_input_norm,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )
