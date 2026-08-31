"""Tests for the waves that cover the whole box.

Two of these classes are the reason the module exists rather than a copy of
what a book says. `TestNothingJumps` is the property a tile coder does not
have and this one does, and `TestTheStepScales` is the one setting the
literature attaches to this basis, checked against the arithmetic it claims
rather than against itself.
"""

from __future__ import annotations

import math
from functools import cache
from itertools import pairwise

import pytest

from rel.agents.base import Transition
from rel.agents.fourier import FlatSteps, FourierBasis
from rel.agents.linear import SemiGradientQ
from rel.agents.linear_prediction import SemiGradientTD, fixed
from rel.rng import Rng
from rel.spaces import Box, Discrete

UNIT = Box([0.0], [1.0])
SQUARE = Box([0.0, 0.0], [1.0, 1.0])

#: A box whose two dimensions differ by a factor of thirteen. The mountain car
#: is close to this, and it is where a basis that forgot to scale a dimension
#: onto zero to one would show it.
LOPSIDED = Box([-1.2, -0.07], [0.6, 0.07])

TWO = Discrete(2)
EVEN = fixed([0.5, 0.5])


def valued(basis: FourierBasis, point: tuple[float, ...]) -> tuple[float, ...]:
    """What the basis says about a point, without the indices."""
    _, values = basis.encode(point)
    return tuple(values)


@cache
def _fit(order: int) -> tuple[float, float]:
    """Fit the line `value(x) = x` on the unit box and read what is left.

    The two numbers are the worst gap over the whole line and the worst gap
    away from the two ends of it. They are cached because several tests
    compare orders and each fit is four thousand updates.
    """
    agent = SemiGradientTD(
        Rng(1).stream("agent"),
        TWO,
        FourierBasis(UNIT, order=order),
        EVEN,
        step_size=0.1,
    )
    rng = Rng(4)
    for _ in range(4000):
        at = rng.uniform(0.0, 1.0)
        agent.observe(Transition((at,), 0, at, (at,), True, False))

    gaps = [abs(agent.value((at / 50.0,)) - at / 50.0) for at in range(51)]
    return max(gaps), max(gaps[5:46])


def fitted(order: int) -> float:
    """The worst gap away from the ends of the line."""
    return _fit(order)[1]


def worst_of(order: int) -> float:
    """The worst gap anywhere on the line, ends included."""
    return _fit(order)[0]


class TestHowManyThereAre:
    def test_one_for_each_vector_of_whole_numbers(self) -> None:
        assert FourierBasis(UNIT, order=3).features == 4
        assert FourierBasis(SQUARE, order=3).features == 16
        assert FourierBasis(Box([0.0] * 4, [1.0] * 4), order=3).features == 256

    def test_an_order_of_zero_is_the_constant_and_nothing_else(self) -> None:
        basis = FourierBasis(SQUARE, order=0)
        assert basis.features == 1
        assert basis.coefficients == ((0, 0),)

    def test_a_negative_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="zero or more"):
            FourierBasis(UNIT, order=-1)

    def test_the_constant_comes_first_and_the_lowest_dimension_moves_fastest(
        self,
    ) -> None:
        # Written out because the order of the coefficients is what the
        # weights are indexed by, and a change to it would silently move
        # every weight a saved run had.
        assert FourierBasis(SQUARE, order=1).coefficients == (
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
        )

    def test_it_says_its_order_and_its_size(self) -> None:
        assert repr(FourierBasis(SQUARE, order=2)) == (
            "FourierBasis(order=2, dimensions=2, features=9)"
        )

    def test_the_flat_one_says_which_it_is(self) -> None:
        # An agent's repr is its own name and its coder's, so a run over the
        # flat steps has to be tellable from a run over the scaled ones.
        assert repr(FlatSteps(SQUARE, order=2)) == (
            "FlatSteps(order=2, dimensions=2, features=9)"
        )


class TestWhatItSaysAboutAPoint:
    def test_every_feature_is_on(self) -> None:
        # Dense, unlike a tile coder. A wave has no off.
        basis = FourierBasis(SQUARE, order=2)
        indices, values = basis.encode((0.3, 0.7))
        assert list(indices) == list(range(9))
        assert len(values) == 9

    def test_the_low_corner_is_every_wave_at_its_top(self) -> None:
        # The scaled point is all zeros there, and the cosine of zero is one
        # whatever the coefficient.
        basis = FourierBasis(LOPSIDED, order=3)
        assert valued(basis, (-1.2, -0.07)) == pytest.approx([1.0] * 16)

    def test_the_high_corner_alternates_with_the_coefficient(self) -> None:
        # The scaled point is all ones, so each wave is the cosine of pi times
        # the sum of its coefficient: one when that sum is even, minus one when
        # it is odd.
        basis = FourierBasis(LOPSIDED, order=3)
        expected = [(-1.0) ** sum(c) for c in basis.coefficients]
        assert valued(basis, (0.6, 0.07)) == pytest.approx(expected)

    def test_no_feature_leaves_minus_one_to_one(self) -> None:
        basis = FourierBasis(LOPSIDED, order=4)
        rng = Rng(1)
        for _ in range(200):
            for value in valued(basis, LOPSIDED.sample(rng)):
                assert -1.0 <= value <= 1.0

    def test_the_constant_is_one_everywhere(self) -> None:
        basis = FourierBasis(LOPSIDED, order=3)
        rng = Rng(2)
        for _ in range(50):
            assert valued(basis, LOPSIDED.sample(rng))[0] == 1.0

    def test_a_point_outside_the_box_is_the_nearest_corner(self) -> None:
        basis = FourierBasis(UNIT, order=3)
        assert valued(basis, (-5.0,)) == valued(basis, (0.0,))
        assert valued(basis, (5.0,)) == valued(basis, (1.0,))

    def test_each_dimension_is_scaled_by_its_own_width(self) -> None:
        # The two dimensions of the lopsided box differ by a factor of
        # thirteen. A basis that used the raw numbers would give the middle of
        # one dimension and the middle of the other different features.
        basis = FourierBasis(LOPSIDED, order=2)
        middle = valued(basis, (-0.3, 0.0))
        assert middle == pytest.approx(valued(FourierBasis(SQUARE, 2), (0.5, 0.5)))

    def test_two_different_points_are_told_apart(self) -> None:
        basis = FourierBasis(UNIT, order=3)
        assert valued(basis, (0.25,)) != valued(basis, (0.75,))


class TestNothingJumps:
    """The property that a tile coder does not have.

    A tile coder cuts the box, so two points a hair apart across a cut have
    features that differ by a whole tile. Every feature here is a cosine of
    the point, so a hair of movement is a hair of change.
    """

    def test_a_hair_of_movement_is_a_hair_of_change(self) -> None:
        basis = FourierBasis(UNIT, order=8)
        for at in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
            here = valued(basis, (min(at, 1.0 - 1e-9),))
            there = valued(basis, (min(at, 1.0 - 1e-9) + 1e-9,))
            for first, second in zip(here, there, strict=True):
                assert abs(first - second) < 1e-7

    def test_walking_across_the_box_never_steps(self) -> None:
        basis = FourierBasis(UNIT, order=4)
        walk = [valued(basis, (index / 400.0,)) for index in range(401)]
        largest = max(
            abs(first - second)
            for here, there in pairwise(walk)
            for first, second in zip(here, there, strict=True)
        )
        # Four waves across a step of a four hundredth. The bound is the
        # fastest wave's own slope and nothing looser.
        assert largest < 4.0 * math.pi / 400.0


class TestTheSquaredLength:
    def test_it_is_the_features_dotted_with_themselves(self) -> None:
        basis = FourierBasis(SQUARE, order=2)
        values = valued(basis, (0.3, 0.7))
        assert basis.squared_length(values) == pytest.approx(
            sum(value * value for value in values)
        )

    def test_at_a_corner_it_is_the_number_of_features(self) -> None:
        # Every wave is one or minus one there, so each contributes one.
        basis = FourierBasis(SQUARE, order=3)
        assert basis.squared_length(valued(basis, (0.0, 0.0))) == pytest.approx(16.0)

    def test_it_never_falls_below_one(self) -> None:
        """Which is what stops a step size from blowing up.

        An agent divides its step size by this, so a point where every
        feature was near zero would take a step of near infinity. There is no
        such point here: the constant feature is one everywhere, so the sum
        is at least one whatever the waves say.
        """
        basis = FourierBasis(LOPSIDED, order=4)
        rng = Rng(5)
        for _ in range(500):
            values = valued(basis, LOPSIDED.sample(rng))
            assert basis.squared_length(values) >= 1.0


class TestTheStepScales:
    """One over the length of each coefficient, and one for the constant."""

    def test_there_is_one_for_each_feature(self) -> None:
        basis = FourierBasis(SQUARE, order=3)
        scales = basis.step_scales()
        assert scales is not None
        assert len(scales) == basis.features

    def test_the_constant_takes_the_whole_step(self) -> None:
        scales = FourierBasis(SQUARE, order=3).step_scales()
        assert scales is not None
        assert scales[0] == 1.0

    def test_a_wave_that_crosses_eight_times_takes_an_eighth_of_a_step(self) -> None:
        basis = FourierBasis(UNIT, order=8)
        scales = basis.step_scales()
        assert scales is not None
        assert basis.coefficients[8] == (8,)
        assert scales[8] == pytest.approx(1.0 / 8.0)

    def test_a_wave_in_two_dimensions_takes_the_length_of_both(self) -> None:
        basis = FourierBasis(SQUARE, order=4)
        scales = basis.step_scales()
        assert scales is not None
        index = basis.coefficients.index((3, 4))
        assert scales[index] == pytest.approx(1.0 / 5.0)

    def test_they_fall_as_the_waves_get_faster(self) -> None:
        basis = FourierBasis(UNIT, order=6)
        scales = basis.step_scales()
        assert scales is not None
        assert list(scales) == sorted(scales, reverse=True)

    def test_the_flat_one_asks_for_nothing(self) -> None:
        assert FlatSteps(SQUARE, order=3).step_scales() is None

    def test_the_flat_one_is_the_same_waves(self) -> None:
        # Only the step sizes differ, which is what makes the two comparable.
        flat = FlatSteps(LOPSIDED, order=3)
        scaled = FourierBasis(LOPSIDED, order=3)
        assert flat.coefficients == scaled.coefficients
        rng = Rng(3)
        for _ in range(20):
            point = LOPSIDED.sample(rng)
            assert valued(flat, point) == valued(scaled, point)


class TestTheStartingWeight:
    def test_nothing_is_the_only_answer(self) -> None:
        assert FourierBasis(UNIT, order=3).starting_weight(0.0) == 0.0

    def test_an_optimistic_start_is_refused_rather_than_faked(self) -> None:
        # A tile coder can share an optimistic value out between its grids
        # because every point lights the same number of them. Here the
        # constant is one everywhere and the waves are not, so no single
        # weight makes every point worth the same.
        with pytest.raises(ValueError, match="no single weight"):
            FourierBasis(UNIT, order=3).starting_weight(1.0)


class TestItWorksAsACoder:
    def test_an_agent_starts_at_nothing_everywhere(self) -> None:
        agent = SemiGradientQ(Rng(1), TWO, FourierBasis(UNIT, order=3))
        assert agent.action_values((0.4,)) == [0.0, 0.0]

    def test_an_agent_takes_the_scaled_step(self) -> None:
        basis = FourierBasis(UNIT, order=3)
        agent = SemiGradientQ(Rng(1), TWO, basis, step_size=0.5)
        assert list(agent._scales) == pytest.approx(list(basis.step_scales() or []))

    def test_a_flat_agent_takes_one_step_for_all_of_them(self) -> None:
        agent = SemiGradientQ(Rng(1), TWO, FlatSteps(UNIT, order=3), step_size=0.5)
        assert list(agent._scales) == [1.0, 1.0, 1.0, 1.0]

    def test_it_learns_a_value_that_is_not_a_constant(self) -> None:
        """The test that says the whole thing works rather than its parts.

        The value of a point on the line is the point itself. A constant
        cannot fit that and the waves can, so an agent with only the constant
        is off by a quarter of the line and an agent with waves is not.
        """
        assert fitted(0) > 0.4
        assert fitted(3) < 0.02

    def test_more_waves_fit_better(self) -> None:
        # Not by as much as the count of them suggests. The value here is a
        # straight line, and a straight line is made of the odd waves only:
        # the even ones have nothing to contribute to it, so order two fits
        # about as well as order one and order four about as well as three.
        assert fitted(8) < fitted(3) < fitted(1)
        assert fitted(2) == pytest.approx(fitted(1), rel=0.2)

    def test_the_corner_is_where_the_waves_are_short(self) -> None:
        # A truncated cosine series is furthest from a straight line at the
        # ends of it. That is a fact about the basis rather than about the
        # learning, so it is stated here rather than hidden in a loose bound.
        assert worst_of(4) > 3.0 * fitted(4)
