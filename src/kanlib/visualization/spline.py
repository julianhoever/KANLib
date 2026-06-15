from typing import Optional

import numpy.typing as npt
import torch

from kanlib.nn.kan_base_layer import KANBaseLayer
from kanlib.visualization import SplineExtractor

from .extractors import extractor_for


def spline_curve(
    layer: KANBaseLayer,
    spline_index: tuple[int, ...],
    num_points: int = 1000,
    spline_extractor: Optional[type[SplineExtractor]] = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    if spline_extractor is None:
        spline_extractor = extractor_for(layer)

    _assert_spline_index(layer, spline_index)

    extractor = spline_extractor(layer, spline_index)
    in_feature_idx = extractor.input_feature_index
    coefficients = extractor.spline_coefficients

    smin, smax = layer.basis.spline_range[in_feature_idx].unbind()
    xs = torch.linspace(smin, smax, num_points).to(layer.basis.grid)

    inputs = torch.zeros(num_points, layer.basis.num_features).to(layer.basis.grid)
    inputs[:, in_feature_idx] = xs

    with torch.no_grad():
        basis_values = layer.basis(inputs)[:, in_feature_idx, :]
        ys = torch.matmul(basis_values, coefficients.to(basis_values))

    return xs.cpu().numpy(), ys.cpu().numpy()


def _assert_spline_index(layer: KANBaseLayer, spline_index: tuple[int, ...]) -> None:
    param_shape = tuple(layer.coefficients.shape[:-1])

    if len(param_shape) != len(spline_index):
        raise ValueError(f"`spline_index` must have length {len(param_shape)}.")

    index_is_negative = any(x < 0 for x in spline_index)
    if index_is_negative:
        raise ValueError("`spline_index` contains negative values.")

    max_index = tuple(x - 1 for x in param_shape)
    index_out_of_bounds = any(idx > ub for idx, ub in zip(spline_index, max_index))
    if index_out_of_bounds:
        raise ValueError(
            f"`spline_index` out of bounds. Maximum spline index: {max_index}"
        )
