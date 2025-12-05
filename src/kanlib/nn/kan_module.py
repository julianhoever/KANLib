from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol, cast

import torch

from kanlib.compute_coefficients import compute_coefficients

from .spline_basis import SplineBasis


@dataclass
class ParamSpec:
    initializer: Callable[[torch.Tensor], torch.Tensor]
    requires_grad: bool = True


@dataclass
class ModuleParamSpecs:
    coefficients: ParamSpec
    weight_spline: Optional[ParamSpec]
    weight_residual: Optional[ParamSpec]
    bias_output: Optional[ParamSpec]


class BasisFactory(Protocol):
    def __call__(self, grid_size: int, spline_range: torch.Tensor) -> SplineBasis: ...


class KANModule(torch.nn.Module, ABC):
    def __init__(
        self,
        param_shape: tuple[int, ...],
        in_feature_dim: int,
        out_feature_dim: int,
        param_specs: ModuleParamSpecs,
        grid_size: int,
        spline_range: tuple[float, float] | torch.Tensor,
        basis_factory: BasisFactory,
        spline_input_norm: Optional[torch.nn.LayerNorm],
    ) -> None:
        super().__init__()
        self.in_feature_dim = in_feature_dim

        self.basis = basis_factory(
            grid_size=grid_size,
            spline_range=_spline_range_to_tensor(
                spline_range, num_features=param_shape[in_feature_dim]
            ),
        )
        self.basis_factory = basis_factory
        self.param_specs = param_specs
        self.spline_input_norm = spline_input_norm

        self.coefficients: torch.nn.Parameter
        self.weight_spline: torch.nn.Parameter | None
        self.weight_residual: torch.nn.Parameter | None
        self.bias_output: torch.nn.Parameter | None

        self._add_parameter(
            "coefficients", (*param_shape, self.basis.num_basis_functions)
        )
        self._add_parameter("weight_spline", param_shape)
        self._add_parameter("weight_residual", param_shape)
        self._add_parameter("bias_output", (param_shape[out_feature_dim],))

    @property
    def weighted_coefficients(self) -> torch.Tensor:
        if self.weight_spline is not None:
            return self.coefficients * self.weight_spline.unsqueeze(dim=-1)
        return self.coefficients

    @weighted_coefficients.setter
    def weighted_coefficients(self, value: torch.Tensor) -> None:
        self.coefficients.data = value
        if self.weight_spline is not None:
            weight_spline_spec = cast(ParamSpec, self.param_specs.weight_spline)
            self.weight_spline.data = weight_spline_spec.initializer(
                torch.empty(self.coefficients.shape[:-1])
            )

    @abstractmethod
    def residual_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def spline_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_spline = x if self.spline_input_norm is None else self.spline_input_norm(x)

        output = self.spline_forward(x_spline)

        if self.weight_residual is not None:
            output += self.residual_forward(x)

        if self.bias_output is not None:
            output += self.bias_output

        return output

    @torch.no_grad
    def refine_grid(self, new_grid_size: int) -> None:
        basis_fine = self.basis_factory(
            grid_size=new_grid_size, spline_range=self.basis.spline_range
        ).to(self.basis.grid.device)

        coeff_fine = compute_coefficients(
            original_coefficients=self.weighted_coefficients.movedim(
                self.in_feature_dim, -2
            ),
            original_basis_values=self.basis(basis_fine.grid.t()),
            target_basis_values=basis_fine(basis_fine.grid.t()),
        )

        self.basis = basis_fine
        self.weighted_coefficients = coeff_fine.movedim(-2, self.in_feature_dim)

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
