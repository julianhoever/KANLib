from kanlib.nn.base_modules.linear import LinearBase

from .gaussian_rbf_basis import GaussianRbfBasis


class Linear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        use_output_bias: bool = True,
        use_layer_norm: bool = True,
        use_residual_branch: bool = True,
        use_spline_weight: bool = True,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            grid_range=grid_range,
            basis_factory=GaussianRbfBasis,
            use_output_bias=use_output_bias,
            use_layer_norm=use_layer_norm,
            use_residual_branch=use_residual_branch,
            use_spline_weight=use_spline_weight,
            init_coeff_std=init_coeff_std,
        )
