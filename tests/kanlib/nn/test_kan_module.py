from typing import cast

import pytest
import torch
from torch.nn.init import ones_ as init_ones
from torch.nn.init import zeros_ as init_zeros
from torch.testing import assert_close

from kanlib.nn.kan_module import KANModule, ModuleParamSpecs, ParamSpec
from kanlib.nn.spline_basis import SplineBasis


class DummyBasis(SplineBasis):
    def __init__(self, grid_size: int, spline_range: torch.Tensor) -> None:
        super().__init__(
            grid_size=grid_size,
            spline_range=spline_range,
            initialize_grid=lambda: torch.empty(*spline_range.shape[:-1], 10),
        )

    @property
    def num_basis_functions(self) -> int:
        return self.grid_size

    def _perform_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class KANModuleImpl(KANModule):
    def __init__(self, use_spline_weight: bool) -> None:
        super().__init__(
            param_shape=(3, 4),
            in_feature_dim=1,
            out_feature_dim=0,
            param_specs=ModuleParamSpecs(
                coefficients=ParamSpec(init_ones),
                weight_spline=ParamSpec(init_ones) if use_spline_weight else None,
                weight_residual=ParamSpec(init_ones),
                bias_output=ParamSpec(init_zeros),
            ),
            grid_size=5,
            spline_range=torch.tensor([[-1, 1]]),
            basis_factory=DummyBasis,
            spline_input_norm=None,
        )

    def residual_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def spline_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@pytest.fixture
def kan_without_spline_weight() -> KANModuleImpl:
    return KANModuleImpl(use_spline_weight=False)


@pytest.fixture
def kan_with_spline_weight() -> KANModuleImpl:
    return KANModuleImpl(use_spline_weight=True)


def test_unweighted_spline_kan_has_no_weight_spline(
    kan_without_spline_weight: KANModule,
) -> None:
    assert kan_without_spline_weight.coefficients is not None
    assert kan_without_spline_weight.weight_spline is None


def test_unweighted_spline_kan_set_coefficient_value(
    kan_without_spline_weight: KANModule,
) -> None:
    ones = torch.ones((2, 3, 4))
    kan_without_spline_weight.weighted_coefficients = ones
    assert kan_without_spline_weight.weight_spline is None
    assert_close(ones, kan_without_spline_weight.coefficients)


def test_unweighted_spline_kan_get_coefficient_returns_unweighted(
    kan_without_spline_weight: KANModule,
) -> None:
    assert_close(
        kan_without_spline_weight.weighted_coefficients,
        kan_without_spline_weight.coefficients,
    )


def test_weighted_spline_kan_has_weight_spline(
    kan_with_spline_weight: KANModule,
) -> None:
    assert kan_with_spline_weight.coefficients is not None
    assert kan_with_spline_weight.weight_spline is not None


def test_get_weighted_coefficient(
    kan_with_spline_weight: KANModule,
) -> None:
    coefficients = kan_with_spline_weight.coefficients
    weight_spline = cast(torch.nn.Parameter, kan_with_spline_weight.weight_spline)
    assert_close(
        kan_with_spline_weight.weighted_coefficients,
        coefficients * weight_spline.unsqueeze(dim=-1),
    )


def test_weighted_spline_kan_set_weighted_coefficient_updates_coefficients(
    kan_with_spline_weight: KANModule,
) -> None:
    target_coefficient = torch.ones((2, 3, 4)) * 6
    kan_with_spline_weight.weighted_coefficients = target_coefficient
    assert_close(kan_with_spline_weight.coefficients, target_coefficient)


def test_weighted_spline_kan_set_weighted_coefficient_resets_weight_spline_to_one(
    kan_with_spline_weight: KANModule,
) -> None:
    target_coefficient = torch.ones((2, 3, 4)) * 6
    target_weight = torch.ones((2, 3))
    kan_with_spline_weight.weighted_coefficients = target_coefficient
    assert_close(kan_with_spline_weight.coefficients, target_coefficient)
    assert_close(kan_with_spline_weight.weight_spline, target_weight)
