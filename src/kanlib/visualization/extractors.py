from abc import ABC, abstractmethod

import torch

from kanlib.nn.base_modules.convolution import ConvBase
from kanlib.nn.base_modules.linear import LinearBase
from kanlib.nn.kan_base_layer import KANBaseLayer


class SplineExtractor[L: KANBaseLayer](ABC):
    def __init__(self, layer: L, spline_index: tuple[int, ...]) -> None:
        self.layer = layer
        self.spline_index = spline_index

    @property
    @abstractmethod
    def input_feature_index(self) -> int: ...

    @property
    def spline_coefficients(self) -> torch.Tensor:
        return self.layer.weighted_coefficients[self.spline_index].detach()


class _LinearSplineExtractor(SplineExtractor[LinearBase]):
    @property
    def input_feature_index(self) -> int:
        return self.spline_index[1]


class _ConvSplineExtractor(SplineExtractor[ConvBase]):
    @property
    def input_feature_index(self) -> int:
        out_channel, in_channel = self.spline_index[:2]

        in_channels_per_group = self.layer.in_channels // self.layer.groups
        out_channels_per_group = self.layer.out_channels // self.layer.groups
        group = out_channel // out_channels_per_group

        return group * in_channels_per_group + in_channel


def extractor_for(layer: KANBaseLayer) -> type[SplineExtractor]:
    if isinstance(layer, LinearBase):
        return _LinearSplineExtractor
    if isinstance(layer, ConvBase):
        return _ConvSplineExtractor
    raise TypeError(f"No spline extractor for layer type {type(layer).__name__}")
