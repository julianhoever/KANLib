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
    alpha: float = 1.0,
    ax: Optional[Axes] = None,
) -> Axes:
    x, y = _compute_spline(spline_spec, resolution)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, y, alpha=alpha, zorder=1)

    return ax


def _compute_spline(
    spline_spec: _SplineSpec, resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    coeff = spline_spec.layer.coefficients.detach()
    coeff_flat = coeff.movedim(spline_spec.feature_dim, -2)
    coeff_flat = coeff_flat.view(-1, *coeff_flat.shape[-2:])

    smin, smax = spline_spec.layer.basis.spline_range.unbind(dim=-1)
    x_spline = torch.linspace(0, 1, resolution).unsqueeze(dim=-1) * (smax - smin) + smin

    basis_values = spline_spec.layer.basis(x_spline).unsqueeze(dim=-3)

    splines_flat = torch.sum(basis_values * coeff_flat, dim=-1)
    splines = splines_flat.movedim(-2, spline_spec.feature_dim)
    splines = splines.view(-1, *coeff.shape[:-1])

    if spline_spec.layer.weight_spline is not None:
        splines = splines * spline_spec.layer.weight_spline.detach()

    x = x_spline[:, spline_spec.index[spline_spec.feature_dim]]
    y = splines[:, *spline_spec.index]

    return x, y
