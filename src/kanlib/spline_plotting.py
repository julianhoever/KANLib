from typing import Optional, Protocol

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes

from kanlib.nn.spline_basis import SplineBasis
from kanlib.nn.spline_coefficient import SplineCoefficient
from kanlib.spline import compute_spline


class _KanLayer(Protocol):
    basis: SplineBasis
    weight: SplineCoefficient


class _LinearLayer(Protocol):
    in_features: int
    out_features: int


def plot_spline(
    layer: _KanLayer,
    spline_index: int,
    resolution: int = 1000,
    show_grid: bool = False,
    ax: Optional[Axes] = None,
) -> Axes:
    basis = layer.basis
    coefficient = layer.weight.coefficient.view(-1, basis.num_basis_functions)
    x_spline = torch.linspace(*basis.grid_range, resolution)
    y_spline = compute_spline(layer.basis, coefficient[spline_index], x_spline)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x_spline, y_spline, zorder=1)

    if show_grid:
        x_grid = basis.grid[
            (basis.grid >= basis.grid_range[0]) & (basis.grid <= basis.grid_range[1])
        ]
        y_grid = compute_spline(layer.basis, coefficient[spline_index], x_grid)
        ax.scatter(x_grid, y_grid, c="red", marker=".", zorder=2)

    return ax


def linear_spline_index(linear: _LinearLayer, input_idx: int, output_idx: int) -> int:
    return output_idx * linear.in_features + input_idx
