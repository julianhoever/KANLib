from kanlib.nn.base_modules.linear import LinearBase

from .gaussian_rbf_basis import GaussianRbfBasis


class Linear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        use_layer_norm: bool = True,
        use_base_branch: bool = True,
        use_coefficient_weight: bool = True,
        init_coeff_std: float = 0.1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            grid_range=grid_range,
            basis_factory=GaussianRbfBasis,
            use_layer_norm=use_layer_norm,
            use_base_branch=use_base_branch,
            use_spline_weight=use_coefficient_weight,
            init_coeff_std=init_coeff_std,
        )
