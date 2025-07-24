from functools import partial

from kanlib.nn.linear_base import LinearBase

from .bspline_basis import BSplineBasis


class Linear(LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spline_order: int,
        grid_size: int,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        use_layer_norm: bool = False,
        use_base_branch: bool = True,
        use_coefficient_weight: bool = True,
        init_coefficient_std: float = 0.1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size,
            grid_range=grid_range,
            basis_factory=partial(BSplineBasis, spline_order=spline_order),
            use_layer_norm=use_layer_norm,
            use_base_branch=use_base_branch,
            use_coefficient_weight=use_coefficient_weight,
            init_coefficient_std=init_coefficient_std,
        )
