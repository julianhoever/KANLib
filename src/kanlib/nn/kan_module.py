from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol, cast

import torch

from kanlib.spline import compute_refined_coefficients

from .spline_basis import SplineBasis


@dataclass
class ParamSpec:
    initializer: Callable[[torch.Tensor], torch.Tensor]
    requires_grad: bool = True


@dataclass
class ModuleParamSpecs:
    coefficients: ParamSpec
    weight_spline: Optional[ParamSpec]
    weight_base: Optional[ParamSpec]


class BasisFactory(Protocol):
    def __call__(
        self, grid_size: int, grid_range: tuple[float, float]
    ) -> SplineBasis: ...


class KANModule(torch.nn.Module, ABC):
    def __init__(
        self,
        base_shape: tuple[int, ...],
        coefficients: ParamSpec,
        weight_spline: Optional[ParamSpec],
        weight_base: Optional[ParamSpec],
        grid_size: int,
        grid_range: tuple[float, float],
        basis_factory: BasisFactory,
    ) -> None:
        super().__init__()
        self.basis = basis_factory(grid_size=grid_size, grid_range=grid_range)
        self.basis_factory = basis_factory

        self.param_specs = ModuleParamSpecs(
            coefficients=coefficients,
            weight_spline=weight_spline,
            weight_base=weight_base,
        )

        self.coefficients: torch.nn.Parameter
        self.weight_spline: torch.nn.Parameter | None
        self.weight_base: torch.nn.Parameter | None

        self._add_parameter(
            "coefficients", (*base_shape, self.basis.num_basis_functions)
        )
        self._add_parameter("weight_spline", (*base_shape, 1))
        self._add_parameter("weight_base", base_shape)

    @abstractmethod
    def base_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def spline_forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.spline_forward(x)

        if self._uses_base_branch:
            output += self.base_forward(x)

        return output

    @property
    def weighted_coefficients(self) -> torch.Tensor:
        if self.weight_spline is not None:
            return self.coefficients * self.weight_spline
        return self.coefficients

    @weighted_coefficients.setter
    def weighted_coefficients(self, value: torch.Tensor) -> None:
        self.coefficients.data = value
        if self.weight_spline is not None:
            weight_spline_spec = cast(ParamSpec, self.param_specs.weight_spline)
            self.weight_spline.data = weight_spline_spec.initializer(
                torch.empty((*self.coefficients.shape[:-1], 1))
            )

    @torch.no_grad
    def refine_grid(self, new_grid_size: int) -> None:
        refined_basis = self.basis_factory(
            grid_size=new_grid_size, grid_range=self.basis.grid_range
        ).to(self.basis.grid.device)
        refined_coefficient = compute_refined_coefficients(
            basis_coarse=self.basis,
            basis_fine=refined_basis,
            coeff_coarse=self.weighted_coefficients,
        )
        self.basis = refined_basis
        self.weighted_coefficients = refined_coefficient

    @property
    def _uses_base_branch(self) -> bool:
        return self.weight_base is not None

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
