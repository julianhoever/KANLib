from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Optional, Protocol

import torch
from torch.nn.init import normal_ as init_normal
from torch.nn.init import ones_ as init_ones
from torch.nn.init import xavier_uniform_ as init_xavier_uniform
from torch.nn.init import zeros_ as init_zeros

from kanlib.compute_coefficients import compute_coefficients

from .spline_basis import AdaptiveGrid, SplineBasis


@dataclass
class LayerSpec:
    input_features: int
    param_shape: tuple[int, ...]
    in_feat_dim: int
    out_feat_dim: int


@dataclass
class KANParamSpec:
    @dataclass
    class ParamSpec:
        initializer: Callable[[torch.Tensor], torch.Tensor]
        requires_grad: bool = True

    coefficients: ParamSpec
    weight_spline: Optional[ParamSpec]
    weight_residual: Optional[ParamSpec]
    bias_output: Optional[ParamSpec]


@dataclass
class BasisSpec:
    class _BasisFactory(Protocol):
        def __call__(
            self, grid_size: int, spline_range: torch.Tensor
        ) -> SplineBasis: ...

    basis_factory: _BasisFactory
    grid_size: int
    spline_range: tuple[float, float] | torch.Tensor


class KANBaseLayer(torch.nn.Module, ABC):
    def __init__(
        self, layer_spec: LayerSpec, param_specs: KANParamSpec, basis_spec: BasisSpec
    ) -> None:
        super().__init__()
        self.in_feature_dim = layer_spec.in_feat_dim

        self.basis = basis_spec.basis_factory(
            grid_size=basis_spec.grid_size,
            spline_range=_spline_range_to_tensor(
                basis_spec.spline_range, num_features=layer_spec.input_features
            ),
        )
        self.basis_factory = basis_spec.basis_factory
        self.param_specs = param_specs

        self.coefficients: torch.nn.Parameter
        self.weight_spline: torch.nn.Parameter | None
        self.weight_residual: torch.nn.Parameter | None
        self.bias_output: torch.nn.Parameter | None

        self._add_parameter(
            "coefficients", (*layer_spec.param_shape, self.basis.num_basis_functions)
        )
        self._add_parameter("weight_spline", layer_spec.param_shape)
        self._add_parameter("weight_residual", layer_spec.param_shape)
        self._add_parameter(
            "bias_output", (layer_spec.param_shape[layer_spec.out_feat_dim],)
        )

    @property
    def weighted_coefficients(self) -> torch.Tensor:
        if self.weight_spline is not None:
            return self.coefficients * self.weight_spline.unsqueeze(dim=-1)
        return self.coefficients

    @abstractmethod
    def residual_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def spline_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.spline_forward(x)

        if self.weight_residual is not None:
            output += self.residual_forward(x)

        if self.bias_output is not None:
            output += self.bias_output

        return output

    @torch.no_grad()
    def refine_grid(self, grid_size: int) -> None:
        basis_fine = self.basis_factory(
            grid_size=grid_size, spline_range=self.basis.spline_range
        ).to(self.basis.grid.device)

        coeff_fine = compute_coefficients(
            original_coefficients=self.coefficients.movedim(self.in_feature_dim, -2),
            original_basis_values=self.basis(basis_fine.grid.t()),
            target_basis_values=basis_fine(basis_fine.grid.t()),
        )

        self.basis = basis_fine
        self.coefficients.data = coeff_fine.movedim(-2, self.in_feature_dim)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor) -> None:
        if not isinstance(self.basis, AdaptiveGrid):
            raise ValueError(
                f"{type(self.basis).__name__} does not support adaptive grid."
            )

        grid_update = self.basis.grid_update_from_samples(x)

        try:
            new_coeff = compute_coefficients(
                original_coefficients=self.coefficients.movedim(
                    self.in_feature_dim, -2
                ),
                original_basis_values=self.basis(x),
                target_basis_values=self.basis(x, grid=grid_update.grid),
            )
        except RuntimeError:
            pass
        else:
            self.basis.update_grid(grid_update)
            self.coefficients.data = new_coeff.movedim(-2, self.in_feature_dim)

    def _add_parameter(self, name: str, shape: tuple[int, ...]) -> None:
        spec = getattr(self.param_specs, name)
        if spec is not None:
            param_data = spec.initializer(torch.empty(shape))
            self.register_parameter(
                name=name,
                param=torch.nn.Parameter(
                    data=param_data, requires_grad=spec.requires_grad
                ),
            )
        else:
            setattr(self, name, None)


def _spline_range_to_tensor(
    spline_range: tuple[float, float] | torch.Tensor, num_features: int
) -> torch.Tensor:
    if isinstance(spline_range, torch.Tensor):
        return spline_range
    return torch.tensor(spline_range, dtype=torch.get_default_dtype()).repeat(
        num_features, 1
    )


def default_param_specs(
    use_spline_weight: bool,
    use_residual_branch: bool,
    use_output_bias: bool,
    init_coeff_std: float,
) -> KANParamSpec:
    return KANParamSpec(
        coefficients=KANParamSpec.ParamSpec(
            partial(init_normal, mean=0, std=init_coeff_std)
        ),
        weight_spline=KANParamSpec.ParamSpec(init_ones) if use_spline_weight else None,
        weight_residual=(
            KANParamSpec.ParamSpec(init_xavier_uniform) if use_residual_branch else None
        ),
        bias_output=KANParamSpec.ParamSpec(init_zeros) if use_output_bias else None,
    )
