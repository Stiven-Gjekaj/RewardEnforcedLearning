"""Tests for the space types."""

from __future__ import annotations

import pytest

from rel.rng import Rng
from rel.spaces import Box, Discrete


class TestDiscrete:
    def test_it_holds_the_values_it_says_it_holds(self) -> None:
        space = Discrete(4)
        assert [space.contains(value) for value in (-1, 0, 3, 4)] == [
            False,
            True,
            True,
            False,
        ]

    def test_a_start_moves_the_range(self) -> None:
        space = Discrete(3, start=5)
        assert space.contains(5)
        assert space.contains(7)
        assert not space.contains(4)
        assert not space.contains(8)

    def test_a_bool_is_not_an_action(self) -> None:
        # `True == 1` in Python, so a bool passes an unguarded range check. A
        # caller that sends a flag where an action belongs then gets action 1
        # and no complaint, and the mistake shows up as an agent that behaves
        # oddly rather than as an error.
        space = Discrete(4)
        assert not space.contains(True)
        assert not space.contains(False)

    def test_a_float_is_not_an_action(self) -> None:
        assert not Discrete(4).contains(1.0)

    def test_it_needs_at_least_one_member(self) -> None:
        with pytest.raises(ValueError):
            Discrete(0)

    def test_a_sample_is_always_a_member(self) -> None:
        space = Discrete(6, start=2)
        rng = Rng(1)
        assert all(space.contains(space.sample(rng)) for _ in range(500))

    def test_a_sample_reaches_every_member(self) -> None:
        space = Discrete(4)
        rng = Rng(2)
        assert {space.sample(rng) for _ in range(300)} == {0, 1, 2, 3}

    def test_it_iterates_over_its_members(self) -> None:
        assert list(Discrete(3, start=10)) == [10, 11, 12]
        assert len(Discrete(3)) == 3

    def test_two_of_the_same_shape_are_equal(self) -> None:
        assert Discrete(4) == Discrete(4)
        assert Discrete(4) != Discrete(5)
        assert Discrete(4, start=1) != Discrete(4)
        assert len({Discrete(4), Discrete(4)}) == 1

    def test_it_reads_back_the_way_it_was_written(self) -> None:
        assert repr(Discrete(4)) == "Discrete(4)"
        assert repr(Discrete(4, start=1)) == "Discrete(4, start=1)"


class TestBox:
    def test_it_holds_the_values_inside_its_bounds(self) -> None:
        space = Box([-1.0, 0.0], [1.0, 10.0])
        assert space.contains((0.0, 5.0))
        assert space.contains((-1.0, 0.0))
        assert space.contains((1.0, 10.0))
        assert not space.contains((1.1, 5.0))
        assert not space.contains((0.0, -0.1))

    def test_the_wrong_number_of_dimensions_is_not_a_member(self) -> None:
        space = Box([-1.0], [1.0])
        assert not space.contains((0.0, 0.0))
        assert not space.contains(0.0)
        assert not space.contains([0.0])

    def test_the_bounds_have_to_agree_in_number(self) -> None:
        with pytest.raises(ValueError):
            Box([0.0, 1.0], [1.0])

    def test_a_bound_has_to_be_below_its_upper(self) -> None:
        with pytest.raises(ValueError):
            Box([1.0], [1.0])

    def test_it_needs_a_dimension(self) -> None:
        with pytest.raises(ValueError):
            Box([], [])

    def test_a_sample_is_always_a_member(self) -> None:
        space = Box([-1.2, -0.07], [0.6, 0.07])
        rng = Rng(3)
        assert all(space.contains(space.sample(rng)) for _ in range(500))

    def test_clip_brings_a_value_back_inside(self) -> None:
        space = Box([-1.0, 0.0], [1.0, 4.0])
        assert space.clip((-5.0, 9.0)) == (-1.0, 4.0)
        assert space.clip((0.5, 1.0)) == (0.5, 1.0)

    def test_scaled_puts_the_bounds_at_zero_and_one(self) -> None:
        space = Box([-1.2, -0.07], [0.6, 0.07])
        assert space.scaled((-1.2, -0.07)) == (0.0, 0.0)
        assert space.scaled((0.6, 0.07)) == (1.0, 1.0)
        middle = space.scaled((-0.3, 0.0))
        assert abs(middle[0] - 0.5) < 1e-12
        assert abs(middle[1] - 0.5) < 1e-12

    def test_names_default_to_a_position(self) -> None:
        assert Box([0.0, 0.0], [1.0, 1.0]).names == ("x0", "x1")

    def test_names_have_to_agree_in_number(self) -> None:
        with pytest.raises(ValueError):
            Box([0.0, 0.0], [1.0, 1.0], names=["only one"])

    def test_it_reads_back_with_its_names(self) -> None:
        space = Box([-1.0], [1.0], names=["position"])
        assert repr(space) == "Box(position=[-1, 1])"

    def test_two_of_the_same_shape_are_equal(self) -> None:
        assert Box([0.0], [1.0]) == Box([0.0], [1.0])
        assert Box([0.0], [1.0]) != Box([0.0], [2.0])
        assert len({Box([0.0], [1.0]), Box([0.0], [1.0])}) == 1
