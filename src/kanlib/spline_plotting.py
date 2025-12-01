from typing import Optional, Protocol

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from kanlib.nn.spline_basis import SplineBasis


class _KanLayer(Protocol):
    basis: SplineBasis
    coefficients: torch.Tensor


class _LinearLayer(Protocol):
    in_features: int
    out_features: int


def plot_spline(
    layer: _KanLayer,
    spline_index: int,
    resolution: int = 1000,
    show_grid: bool = False,
    alpha: float = 1.0,
    ax: Optional[Axes] = None,
) -> Axes:
    basis = layer.basis
    coefficient = layer.coefficients.view(-1, basis.num_basis_functions).detach()
    x_spline = torch.linspace(*basis.spline_range, resolution)
    y_spline = _compute_spline(layer.basis, coefficient[spline_index], x_spline)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x_spline, y_spline, alpha=alpha, zorder=1)

    if show_grid:
        x_grid = basis.grid[
            (basis.grid >= basis.spline_range[0])
            & (basis.grid <= basis.spline_range[1])
        ]
        y_grid = _compute_spline(layer.basis, coefficient[spline_index], x_grid)
        ax.scatter(x_grid, y_grid, c="red", marker=".", alpha=alpha, zorder=2)

    return ax


def linear_spline_index(linear: _LinearLayer, input_idx: int, output_idx: int) -> int:
    return output_idx * linear.in_features + input_idx


def _compute_spline(
    basis: SplineBasis, coefficient: torch.Tensor, inputs: torch.Tensor
) -> torch.Tensor:
    assert inputs.dim() == 1
    assert coefficient.shape[-1] == basis.num_basis_functions

    inputs = inputs.unsqueeze(dim=-1)
    flat_coeff = coefficient.view(-1, basis.num_basis_functions)
    flat_spline = torch.sum(basis(inputs) * flat_coeff, dim=-1).transpose(0, 1)
    return flat_spline.view(*coefficient.shape[:-1], -1)
