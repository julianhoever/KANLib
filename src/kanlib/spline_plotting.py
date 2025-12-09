from dataclasses import dataclass
from typing import Optional, Protocol

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from kanlib.nn.spline_basis import SplineBasis


class _KanLayer(Protocol):
    basis: SplineBasis
    coefficients: torch.Tensor
    weight_spline: Optional[torch.Tensor]


@dataclass
class _SplineSpec:
    layer: _KanLayer
    feature_dim: int
    index: tuple[int, ...]


def linear_spline(linear: _KanLayer, input_idx: int, output_idx: int) -> _SplineSpec:
    return _SplineSpec(layer=linear, feature_dim=1, index=(output_idx, input_idx))


def plot_spline(
    spline_spec: _SplineSpec,
    resolution: int = 1000,
    show_grid: bool = False,
    alpha: float = 1.0,
    ax: Optional[Axes] = None,
) -> Axes:
    x_spline, y_spline = _compute_spline(spline_spec, resolution)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x_spline, y_spline, alpha=alpha, zorder=1)

    if show_grid:
        x_grid, y_grid = _compute_spline_grid(spline_spec)
        ax.scatter(x_grid, y_grid, c="red", marker=".", alpha=alpha, zorder=2)

    return ax


def _compute_spline(
    spline_spec: _SplineSpec, resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    smin, smax = spline_spec.layer.basis.spline_range.unbind(dim=-1)
    x_spline = torch.linspace(0, 1, resolution).unsqueeze(dim=-1) * (smax - smin) + smin
    return _compute_spline_values(spline_spec, x_spline)


def _compute_spline_grid(
    spline_spec: _SplineSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    smin, smax = spline_spec.layer.basis.spline_range.unbind(dim=-1)
    grid = torch.clamp(spline_spec.layer.basis.grid.t(), smin, smax)
    return _compute_spline_values(spline_spec, grid)


def _compute_spline_values(
    spline_spec: _SplineSpec, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    coeff = spline_spec.layer.coefficients.detach()
    coeff_flat = coeff.movedim(spline_spec.feature_dim, -2)
    coeff_flat = coeff_flat.view(-1, *coeff_flat.shape[-2:])

    basis_values = spline_spec.layer.basis(inputs).unsqueeze(dim=-3)

    splines_flat = torch.sum(basis_values * coeff_flat, dim=-1)
    splines = splines_flat.movedim(-2, spline_spec.feature_dim)
    splines = splines.view(-1, *coeff.shape[:-1])

    if spline_spec.layer.weight_spline is not None:
        splines = splines * spline_spec.layer.weight_spline.detach()

    x = inputs[:, spline_spec.index[spline_spec.feature_dim]]
    y = splines[:, *spline_spec.index]

    return x, y
