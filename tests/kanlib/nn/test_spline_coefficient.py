from typing import cast

import pytest
import torch
from torch.testing import assert_close

from kanlib.nn.spline_coefficient import SplineCoefficient


@pytest.fixture
def coeff_shape() -> tuple[int, ...]:
    return (3, 4, 5)


@pytest.fixture
def unweighted_coeff(coeff_shape: tuple[int, ...]) -> SplineCoefficient:
    coeff = SplineCoefficient(
        shape=coeff_shape, init_coefficient_std=0.1, use_coefficient_weight=False
    )
    coeff.unweighted.data = torch.ones_like(coeff.unweighted) * 2
    return coeff


@pytest.fixture
def weighted_coeff(coeff_shape: tuple[int, ...]) -> SplineCoefficient:
    return SplineCoefficient(
        shape=coeff_shape, init_coefficient_std=0.1, use_coefficient_weight=True
    )


def test_unweighted_coefficient_has_no_weight(
    unweighted_coeff: SplineCoefficient,
) -> None:
    assert unweighted_coeff.unweighted is not None
    assert unweighted_coeff.weight is None


def test_set_unweighted_coefficient_value(unweighted_coeff: SplineCoefficient) -> None:
    ones = torch.ones((2, 3, 4))
    unweighted_coeff.coefficient = ones
    assert_close(ones, unweighted_coeff.unweighted)


def test_get_coefficient_returns_unweighted(
    unweighted_coeff: SplineCoefficient,
) -> None:
    assert_close(unweighted_coeff.coefficient, unweighted_coeff.unweighted)


def test_weighted_coefficient_has_weight(weighted_coeff: SplineCoefficient) -> None:
    assert weighted_coeff.unweighted is not None
    assert weighted_coeff.weight is not None


def test_get_coefficient_returns_weighted_coefficients(
    weighted_coeff: SplineCoefficient,
) -> None:
    unweighted = weighted_coeff.unweighted
    weight = cast(torch.nn.Parameter, weighted_coeff.weight)
    assert_close(weighted_coeff.coefficient, unweighted * weight)


def test_set_weighted_coefficient_updates_unweighted(
    weighted_coeff: SplineCoefficient,
) -> None:
    target_coefficient = torch.ones((2, 3, 4)) * 6
    weighted_coeff.coefficient = target_coefficient
    assert_close(weighted_coeff.unweighted, target_coefficient)


def test_set_weighted_coefficient_resets_weight_to_one(
    weighted_coeff: SplineCoefficient,
) -> None:
    target_coefficient = torch.ones((2, 3, 4)) * 6
    target_weight = torch.ones((2, 3, 1))
    weighted_coeff.coefficient = target_coefficient
    assert_close(weighted_coeff.unweighted, target_coefficient)
    assert_close(weighted_coeff.weight, target_weight)
