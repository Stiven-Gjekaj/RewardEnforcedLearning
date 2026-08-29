"""Tests for turning real numbers into how close they are to some places.

The tile coder next door is tested for the shape of its answer. This one is
tested for the shape of its answer and for the promise the module makes about
it, which is that nothing about the answer jumps. A jump is the whole reason
this encoder exists, so a test that does not go looking for one is not testing
the thing that was built.

Going looking for one found one. `TestKeepingBringsTheBoundaryBack` is what
came of it and is why `kept` is off by default.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import ClassVar

import pytest

from rel.agents.basis import RadialBasis
from rel.agents.tiles import TileCoder
from rel.spaces import Box

UNIT = Box([0.0, 0.0], [1.0, 1.0])

#: A box whose two dimensions differ by a factor of thirteen, for the tests
#: about scaling. The mountain car is close to this and it is the environment
#: the module was written against.
LOPSIDED = Box([-1.2, -0.07], [0.6, 0.07])

#: Small enough to straddle a boundary and large enough that the values either
#: side of it differ by far less than any jump would.
HAIR = 1e-9


def weights(basis: RadialBasis, point: tuple[float, ...]) -> dict[int, float]:
    """What the basis says about a point, as a lookup from index to value."""
    return dict(zip(*basis.encode(point), strict=True))


def moved(first: dict[int, float], second: dict[int, float]) -> float:
    """The largest change in any one feature between two encodings.

    A feature in one and not the other counts as a change of its whole value,
    which is the point: the agent read a weight there and now does not.
    """
    return max(
        abs(first.get(index, 0.0) - second.get(index, 0.0))
        for index in set(first) | set(second)
    )


class TestShape:
    def test_a_centre_for_each_corner_of_the_grid(self) -> None:
        assert RadialBasis(UNIT, bins=6).features == 36
        assert RadialBasis(Box([0.0], [1.0]), bins=6).features == 6

    def test_four_dimensions_cost_what_the_module_says_they_cost(self) -> None:
        # The docstring calls 4096 the reason this is a demonstration. If that
        # number is wrong the argument for keeping the tile coder is wrong.
        box = Box([0.0] * 4, [1.0] * 4)
        assert RadialBasis(box, bins=8).features == 4096

    def test_encode_gives_two_tuples_of_the_same_length(self) -> None:
        indices, values = RadialBasis(UNIT).encode((0.3, 0.7))
        assert len(indices) == len(values)

    def test_the_indices_come_back_in_order(self) -> None:
        # A caller walking them alongside a weight row does not care, and a
        # reader comparing two encodings by eye does.
        indices, _ = RadialBasis(UNIT, kept=8).encode((0.31, 0.62))
        assert list(indices) == sorted(indices)

    def test_no_index_appears_twice(self) -> None:
        # One repeated index would update one weight twice for one point and
        # quietly double the step size for it.
        indices, _ = RadialBasis(UNIT, kept=8).encode((0.31, 0.62))
        assert len(set(indices)) == len(indices)

    def test_every_index_is_inside_the_range(self) -> None:
        basis = RadialBasis(UNIT, bins=5, kept=4)
        for x in range(11):
            for y in range(11):
                indices, _ = basis.encode((x / 10, y / 10))
                for index in indices:
                    assert 0 <= index < basis.features


class TestTheValues:
    def test_they_add_up_to_one(self) -> None:
        for point in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.31, 0.62)]:
            _, values = RadialBasis(UNIT).encode(point)
            assert sum(values) == pytest.approx(1.0)

    def test_they_add_up_to_one_however_many_are_kept(self) -> None:
        for kept in (1, 2, 8, 36, None):
            _, values = RadialBasis(UNIT, kept=kept).encode((0.31, 0.62))
            assert sum(values) == pytest.approx(1.0)

    def test_every_value_is_above_zero(self) -> None:
        _, values = RadialBasis(UNIT).encode((0.31, 0.62))
        assert all(value > 0.0 for value in values)

    def test_a_point_on_a_centre_gives_that_centre_the_most(self) -> None:
        basis = RadialBasis(UNIT, bins=6)
        # The centres are laid out by `product`, so the one at (0.2, 0.4) is
        # at index 1 * 6 + 2.
        assert max(weights(basis, (0.2, 0.4)).items(), key=lambda p: p[1])[0] == 8

    def test_the_nearer_centre_always_gets_the_larger_value(self) -> None:
        basis = RadialBasis(UNIT, bins=6)
        point = (0.31, 0.62)
        distances = basis.squared_distances(point)
        values = basis.all_values(point)
        order = sorted(range(basis.features), key=distances.__getitem__)
        assert all(values[near] >= values[far] for near, far in pairwise(order))

    def test_one_spread_from_a_centre_is_the_usual_fraction(self) -> None:
        basis = RadialBasis(Box([0.0], [1.0]), bins=6, width=1.0)
        assert basis.spread == pytest.approx(0.2)
        # The centre at index 1 sits at 0.2, so 0.4 is one spread away.
        assert basis.all_values((0.4,))[1] == pytest.approx(math.exp(-0.5))


class TestTheDefaultWidth:
    """Three quarters of a spacing, which was measured rather than reasoned.

    One whole spacing is the value that looks right, and it loses on both of
    the environments this project has to try it on: by 11 points of return on
    the mountain car over twelve seeds and by 184 on the cart pole over eight,
    with the bootstrap interval clear of zero both times.

    Written down here so that changing it means reading that.
    """

    def test_it_is_three_quarters_of_the_spacing_between_centres(self) -> None:
        basis = RadialBasis(UNIT, bins=6)
        assert basis.width == 0.75
        assert basis.spread == pytest.approx(0.75 / 5.0)

    def test_the_registry_agents_use_it_too(self) -> None:
        # The builder repeats the number rather than reading it off the class,
        # so the two can drift apart and nothing else would say so.
        from rel.agents import AGENTS
        from rel.envs import ENVIRONMENTS
        from rel.rng import Rng

        env = ENVIRONMENTS.make("mountaincar", Rng(1).stream("env"))
        for name in ("rbf-sarsa", "rbf-q"):
            agent = AGENTS.make(name, Rng(1).stream("agent"), env)
            assert agent.coder.width == RadialBasis(UNIT).width


class TestTheWidth:
    def test_wider_makes_the_far_centres_matter_more(self) -> None:
        point = (0.5, 0.5)
        narrow = min(RadialBasis(UNIT, width=0.5).all_values(point))
        wide = min(RadialBasis(UNIT, width=2.0).all_values(point))
        assert wide > narrow

    def test_wider_flattens_the_encoded_values(self) -> None:
        # The measurable version of "the features carry no information about
        # where the point is": the gap between the largest and smallest value
        # closes as the width grows.
        point = (0.31, 0.62)
        gaps = []
        for width in (0.5, 1.0, 2.0, 4.0):
            _, values = RadialBasis(UNIT, width=width).encode(point)
            gaps.append(max(values) - min(values))
        assert all(later < earlier for earlier, later in pairwise(gaps))

    def test_the_width_is_in_the_units_of_the_scaled_box(self) -> None:
        """A box thirteen times wider in one dimension encodes the same.

        Without the scaling a basis over the mountain car would be round in
        the units of the velocity and flat in the units of the position, and
        which dimension the agent could tell apart would depend on what units
        the environment happened to report in.
        """
        square = RadialBasis(UNIT, bins=6).encode((0.25, 0.75))
        lopsided = RadialBasis(LOPSIDED, bins=6).encode(
            (-1.2 + 0.25 * 1.8, -0.07 + 0.75 * 0.14)
        )
        assert square[0] == lopsided[0]
        assert square[1] == pytest.approx(lopsided[1])


class TestKeeping:
    def test_the_default_keeps_every_centre(self) -> None:
        basis = RadialBasis(UNIT, bins=6)
        indices, _ = basis.encode((0.31, 0.62))
        assert len(indices) == basis.features

    def test_it_keeps_the_number_it_was_asked_for(self) -> None:
        indices, _ = RadialBasis(UNIT, bins=6, kept=5).encode((0.31, 0.62))
        assert len(indices) == 5

    def test_asking_for_more_than_there_are_keeps_every_centre(self) -> None:
        basis = RadialBasis(UNIT, bins=4, kept=500)
        indices, _ = basis.encode((0.31, 0.62))
        assert len(indices) == basis.features

    def test_the_ones_it_keeps_are_the_largest(self) -> None:
        basis = RadialBasis(UNIT, bins=6, kept=8)
        point = (0.31, 0.62)
        every = basis.all_values(point)
        indices, _ = basis.encode(point)
        largest = sorted(every, reverse=True)[:8]
        assert sorted((every[index] for index in indices), reverse=True) == largest

    def test_what_it_drops_is_not_near_zero(self) -> None:
        """The sentence this replaced said the dropped values were near zero.

        Eight of thirty six centres carry ninety percent of the total at the
        default width. Dividing those eight by their own smaller total moves
        the largest of them by an eighth of itself.
        """
        point = (0.31, 0.62)
        exact = weights(RadialBasis(UNIT, bins=6), point)
        cheap = weights(RadialBasis(UNIT, bins=6, kept=8), point)

        assert sum(exact[index] for index in cheap) == pytest.approx(0.901, abs=0.001)
        assert max(abs(cheap[i] - exact[i]) for i in cheap) == pytest.approx(
            0.026, abs=0.001
        )

    def test_a_narrow_enough_width_makes_dropping_nearly_free(self) -> None:
        # The setting in which the sentence would have been true. It is a
        # setting and not the default, which is the whole correction.
        point = (0.31, 0.62)
        exact = weights(RadialBasis(UNIT, bins=6, width=0.4), point)
        cheap = weights(RadialBasis(UNIT, bins=6, width=0.4, kept=8), point)
        assert max(abs(cheap[i] - exact[i]) for i in cheap) < 0.001


class TestNothingJumps:
    """The promise the module is built to keep, against the tile coder.

    A tile boundary is a place where two points a hair apart share one switch
    fewer. This asks the same question of both encoders at the same points.
    """

    def test_a_tile_boundary_costs_a_switch(self) -> None:
        # The control. Without this the test below says only that the radial
        # basis is smooth somewhere, which is true of any encoder anywhere.
        coder = TileCoder(UNIT, bins=8, grids=8)
        lost = [
            len(set(coder.active((x - HAIR, 0.5))) - set(coder.active((x, 0.5))))
            for x in (0.125, 0.25, 0.375, 0.5)
        ]
        assert lost == [1, 1, 1, 1]

    def test_the_same_places_move_a_radial_basis_by_nothing(self) -> None:
        basis = RadialBasis(UNIT, bins=6)
        for x in (0.125, 0.25, 0.375, 0.5):
            assert moved(weights(basis, (x - HAIR, 0.5)), weights(basis, (x, 0.5))) < (
                1e-6
            )

    def test_nowhere_along_a_line_across_the_box_does_it_jump(self) -> None:
        """The sweep the whole class exists for.

        A thousand points along a line, and the largest move between any two
        of them is about the step between them. A jump anywhere would stand
        out from that by orders of magnitude, which is exactly how the one
        `kept` puts back was found.
        """
        basis = RadialBasis(UNIT, bins=6)
        step = 1e-3
        rows = [weights(basis, (n * step, 0.5)) for n in range(1001)]
        assert max(moved(a, b) for a, b in pairwise(rows)) < 10.0 * step


class TestKeepingBringsTheBoundaryBack:
    """What `kept` costs, which is the thing the module claimed not to have.

    Between two centres there is a point where the smallest kept value and the
    largest dropped one cross. Either side of it the agent reads a different
    weight for a feature of the same size, so the value it computes jumps.
    """

    def test_the_kept_set_changes_along_the_same_line(self) -> None:
        basis = RadialBasis(UNIT, bins=6, kept=8)
        step = 1e-3
        rows = [set(basis.encode((n * step, 0.5))[0]) for n in range(1001)]
        assert sum(1 for a, b in pairwise(rows) if a != b) > 0

    def test_the_jumps_sit_between_the_centres(self) -> None:
        # Six centres a side puts them at 0, 0.2, 0.4, 0.6, 0.8 and 1.0, so
        # the halfway points are 0.1, 0.3, 0.5, 0.7 and 0.9. Nowhere else.
        #
        # Within one sweep step of one, rather than at one. The line is swept
        # at a thousandth and the crossing is somewhere inside the step that
        # contains it. Each midpoint shows up twice for the same reason: the
        # point sitting on it has centres tied either side, and which of the
        # tied ones is kept is settled by index.
        basis = RadialBasis(UNIT, bins=6, kept=8)
        step = 1e-3
        rows = [set(basis.encode((n * step, 0.5))[0]) for n in range(1001)]
        changed = [
            (index + 1) * step for index, (a, b) in enumerate(pairwise(rows)) if a != b
        ]
        assert changed
        assert all(
            min(abs(x - middle) for middle in (0.1, 0.3, 0.5, 0.7, 0.9)) <= 1.5 * step
            for x in changed
        )

    def test_the_jump_is_the_size_of_the_smallest_kept_value(self) -> None:
        basis = RadialBasis(UNIT, bins=6, kept=8)
        below = weights(basis, (0.1 - HAIR, 0.5))
        above = weights(basis, (0.1 + HAIR, 0.5))
        assert set(below) != set(above)
        assert moved(below, above) == pytest.approx(min(below.values()), abs=1e-6)

    def test_it_never_exceeds_the_smallest_kept_value(self) -> None:
        basis = RadialBasis(UNIT, bins=6, kept=8)
        step = 1e-3
        rows = [weights(basis, (n * step, 0.5)) for n in range(1001)]
        for first, second in pairwise(rows):
            assert moved(first, second) <= min(first.values()) + 10.0 * step

    def test_keeping_more_makes_the_jump_smaller(self) -> None:
        sizes = []
        for kept in (8, 16, 24):
            basis = RadialBasis(UNIT, bins=6, kept=kept)
            sizes.append(
                moved(
                    weights(basis, (0.1 - HAIR, 0.5)), weights(basis, (0.1 + HAIR, 0.5))
                )
            )
        assert all(later < earlier for earlier, later in pairwise(sizes))

    def test_it_is_still_smaller_than_a_tile_boundary(self) -> None:
        """The jump is real and it is not as large as the one it replaces.

        A tile coder loses a whole switch of eight, so a value built from
        switches of equal weight loses an eighth of itself. This loses 0.036
        where the largest feature at that point is 0.214.
        """
        basis = RadialBasis(UNIT, bins=6, kept=8)
        below = weights(basis, (0.1 - HAIR, 0.5))
        jump = moved(below, weights(basis, (0.1 + HAIR, 0.5)))
        assert jump == pytest.approx(0.036, abs=0.001)
        assert max(below.values()) == pytest.approx(0.214, abs=0.001)
        assert jump < 1.0 / 8.0


class TestOutsideTheBox:
    def test_a_point_past_a_bound_is_pulled_back_to_it(self) -> None:
        basis = RadialBasis(UNIT)
        assert basis.encode((5.0, 0.5)) == basis.encode((1.0, 0.5))
        assert basis.encode((-5.0, 0.5)) == basis.encode((0.0, 0.5))

    def test_a_corner_still_adds_up_to_one(self) -> None:
        _, values = RadialBasis(UNIT).encode((99.0, -99.0))
        assert sum(values) == pytest.approx(1.0)


class TestWhenEverythingUnderflows:
    """A width small enough that every centre answers exactly zero.

    Absurd as a setting and reachable by hand, so it has an answer rather than
    a division by zero. The answer is the nearest centre at full strength.
    """

    #: Narrow enough that `exp` underflows to zero at a twentieth of the box
    #: away from a centre, and wide enough that its square is still a number.
    TINY = 0.005

    #: Points a twentieth of the box from a centre, and which centre that is.
    #: A point sitting exactly on a centre is worth 1.0 however narrow the
    #: width, so a test about underflow has to stand away from one.
    OFF_CENTRE: ClassVar[list[tuple[tuple[float, float], int]]] = [
        ((0.05, 0.05), 0),
        ((0.95, 0.95), 35),
        ((0.25, 0.45), 1 * 6 + 2),
        ((0.75, 0.95), 4 * 6 + 5),
    ]

    def test_every_value_really_is_zero(self) -> None:
        basis = RadialBasis(UNIT, bins=6, width=self.TINY)
        assert set(basis.all_values((0.05, 0.05))) == {0.0}

    def test_one_centre_comes_back_at_full_strength(self) -> None:
        indices, values = RadialBasis(UNIT, bins=6, width=self.TINY).encode(
            (0.05, 0.05)
        )
        assert values == (1.0,)
        assert len(indices) == 1

    def test_it_is_the_nearest_centre_and_not_the_first(self) -> None:
        """The bug this test is named after.

        The first version chose by value. Every value is zero here, so the
        largest of them was the first of them, and every point in the box
        encoded as the corner. Choosing by distance is the fix and this is
        the only setting in which the two differ.
        """
        basis = RadialBasis(UNIT, bins=6, width=self.TINY)
        for point, expected in self.OFF_CENTRE:
            assert basis.encode(point)[0] == (expected,)

    def test_a_point_on_a_centre_still_finds_itself(self) -> None:
        # Not the underflow path at all: the value there is exactly 1.0.
        basis = RadialBasis(UNIT, bins=6, width=self.TINY)
        assert basis.encode((0.2, 0.4)) == ((8,), (1.0,))

    def test_a_width_with_no_square_at_all_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to divide by"):
            RadialBasis(UNIT, width=1e-200)


class TestItIsRefusedWhenItCannotWork:
    def test_one_centre_a_side_has_no_spacing(self) -> None:
        with pytest.raises(ValueError, match="at least two centres"):
            RadialBasis(UNIT, bins=1)

    def test_a_width_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above zero"):
            RadialBasis(UNIT, width=0.0)

    def test_a_negative_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above zero"):
            RadialBasis(UNIT, width=-1.0)

    def test_keeping_no_centre_at_all_is_refused(self) -> None:
        # Not the same as keeping every centre, which is `None`.
        with pytest.raises(ValueError, match="says nothing at all"):
            RadialBasis(UNIT, kept=0)

    def test_keeping_a_negative_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="says nothing at all"):
            RadialBasis(UNIT, kept=-1)


class TestItRepeats:
    def test_the_same_point_encodes_the_same_way_twice(self) -> None:
        basis = RadialBasis(UNIT)
        assert basis.encode((0.31, 0.62)) == basis.encode((0.31, 0.62))

    def test_two_bases_built_the_same_way_agree(self) -> None:
        first = RadialBasis(LOPSIDED, bins=5, width=1.5, kept=6)
        second = RadialBasis(LOPSIDED, bins=5, width=1.5, kept=6)
        assert first.encode((-0.3, 0.01)) == second.encode((-0.3, 0.01))

    def test_it_reads_back_something_useful(self) -> None:
        assert repr(RadialBasis(UNIT, bins=6, width=1.5, kept=4)) == (
            "RadialBasis(bins=6, width=1.5, kept=4, features=36)"
        )

    def test_keeping_everything_reads_back_as_such(self) -> None:
        assert "kept=None" in repr(RadialBasis(UNIT))
