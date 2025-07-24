from typing import Protocol

import torch

from kanlib.nn.spline_coefficient import SplineCoefficient
from kanlib.spline import compute_refined_coefficients

from .spline_basis import SplineBasis


class _BasisFactory(Protocol):
    def __call__(
        self, grid_size: int, grid_range: tuple[float, float]
    ) -> SplineBasis: ...


class LinearBase(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        grid_range: tuple[float, float],
        basis_factory: _BasisFactory,
        use_base_branch: bool,
        use_layer_norm: bool,
        use_coefficient_weight: bool,
        init_coefficient_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._basis_factory = basis_factory
        self.basis = basis_factory(grid_size=grid_size, grid_range=grid_range)

        self.base_branch = None
        if use_base_branch:
            self.base_branch = torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(self.in_features, self.out_features, bias=False),
            )

        self.layer_norm = torch.nn.LayerNorm(in_features) if use_layer_norm else None

        self.weight = SplineCoefficient(
            shape=(out_features, in_features, self.basis.num_basis_functions),
            init_coefficient_std=init_coefficient_std,
            use_coefficient_weight=use_coefficient_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x

        if self.layer_norm is not None:
            normalized = self.layer_norm(normalized)

        spline = self._spline_forward(normalized)

        if self.base_branch is not None:
            return spline + self.base_branch(x)

        return spline

    def _spline_forward(self, x: torch.Tensor) -> torch.Tensor:
        basis: torch.Tensor = self.basis(x)
        basis = basis.flatten(start_dim=-2)
        spline = torch.nn.functional.linear(
            input=basis, weight=self.weight.coefficient.flatten(start_dim=-2)
        )
        return spline

    @torch.no_grad
    def refine_grid(self, new_grid_size: int) -> None:
        refined_basis = self._basis_factory(
            grid_size=new_grid_size, grid_range=self.basis.grid_range
        ).to(self.basis.grid.device)
        refined_coefficient = compute_refined_coefficients(
            basis_coarse=self.basis,
            basis_fine=refined_basis,
            coeff_coarse=self.weight.coefficient,
        )
        self.basis = refined_basis
        self.weight.coefficient = refined_coefficient
