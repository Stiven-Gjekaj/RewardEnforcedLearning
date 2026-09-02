"""Tests for the partial sums that answer a weighted draw."""

from __future__ import annotations

import bisect
import math

import pytest

from rel.agents.sums import Smallest, Sums
from rel.rng import Rng


def loaded(weights: list[float]) -> Sums:
    tree = Sums(len(weights))
    for place, weight in enumerate(weights):
        tree[place] = weight
    return tree


def scanned(weights: list[float], target: float) -> int:
    """Where the linear scan in `rel.agents.replay` would land."""
    running: list[float] = []
    total = 0.0
    for weight in weights:
        total += weight
        running.append(total)
    place = bisect.bisect_right(running, target)
    return min(place, len(weights) - 1)


class TestHolding:
    def test_it_starts_at_nothing(self) -> None:
        tree = Sums(4)
        assert tree.total() == 0.0
        assert [tree[place] for place in range(4)] == [0.0] * 4

    def test_it_gives_back_what_it_was_given(self) -> None:
        tree = loaded([1.0, 2.0, 3.0])
        assert [tree[place] for place in range(3)] == [1.0, 2.0, 3.0]

    def test_it_is_the_size_it_was_asked_for(self) -> None:
        assert len(Sums(5)) == 5

    def test_a_tree_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one place"):
            Sums(0)

    def test_a_negative_weight_is_refused(self) -> None:
        tree = Sums(3)
        with pytest.raises(ValueError, match="not negative"):
            tree[1] = -1.0

    def test_a_place_the_tree_does_not_hold_is_refused(self) -> None:
        tree = Sums(3)
        with pytest.raises(IndexError, match="No place 3"):
            tree[3] = 1.0
        with pytest.raises(IndexError, match="No place -1"):
            _ = tree[-1]


class TestTotalling:
    def test_the_total_is_the_sum_of_the_weights(self) -> None:
        assert loaded([1.0, 2.0, 3.0, 4.0]).total() == 10.0

    def test_a_changed_weight_changes_the_total(self) -> None:
        tree = loaded([1.0, 2.0, 3.0])
        tree[1] = 10.0
        assert tree.total() == 14.0

    def test_the_total_holds_at_a_size_that_is_not_a_power_of_two(self) -> None:
        assert loaded([1.0] * 7).total() == 7.0

    def test_a_weight_set_twice_is_not_counted_twice(self) -> None:
        tree = Sums(4)
        tree[0] = 5.0
        tree[0] = 2.0
        assert tree.total() == 2.0


class TestFinding:
    def test_a_target_lands_in_the_place_that_covers_it(self) -> None:
        tree = loaded([1.0, 2.0, 3.0, 4.0])
        assert tree.find(0.5) == 0
        assert tree.find(1.5) == 1
        assert tree.find(3.5) == 2
        assert tree.find(6.5) == 3

    def test_a_boundary_belongs_to_the_place_after_it(self) -> None:
        tree = loaded([1.0, 2.0, 3.0])
        assert tree.find(1.0) == 1
        assert tree.find(3.0) == 2

    def test_a_target_at_the_total_is_held_to_the_last_place(self) -> None:
        tree = loaded([1.0, 2.0, 3.0])
        assert tree.find(6.0) == 2
        assert tree.find(100.0) == 2

    def test_a_target_below_zero_is_held_to_the_first_place(self) -> None:
        assert loaded([1.0, 2.0, 3.0]).find(-1.0) == 0

    def test_a_place_of_no_weight_is_never_found(self) -> None:
        tree = loaded([1.0, 0.0, 1.0])
        found = {tree.find(step / 100.0 * tree.total()) for step in range(100)}
        assert found == {0, 2}

    def test_the_padding_beyond_the_last_place_is_never_found(self) -> None:
        tree = loaded([1.0, 1.0, 1.0])
        found = {tree.find(step / 1000.0 * tree.total()) for step in range(1000)}
        assert found == {0, 1, 2}

    def test_it_finds_what_the_scan_finds_on_whole_numbers(self) -> None:
        weights = [1.0, 4.0, 2.0, 8.0, 3.0]
        tree = loaded(weights)
        for step in range(180):
            target = step / 10.0
            assert tree.find(target) == scanned(weights, target)

    def test_it_finds_what_the_scan_finds_on_a_run_of_random_weights(self) -> None:
        rng = Rng(7)
        weights = [rng.uniform(0.1, 3.0) for _ in range(64)]
        tree = loaded(weights)
        total = sum(weights)
        for _ in range(2000):
            target = rng.random() * total
            assert tree.find(target) == scanned(weights, target)


class TestCost:
    def test_a_change_touches_one_cell_per_level(self) -> None:
        counted = 0

        class Counting(Sums):
            __slots__ = ()

            def __setitem__(self, place: int, weight: float) -> None:
                nonlocal counted
                counted += 1
                super().__setitem__(place, weight)

        tree = Counting(1024)
        tree[500] = 1.0
        assert counted == 1

    def test_a_tree_is_the_next_power_of_two_of_cells(self) -> None:
        assert len(Sums(1000)._cell) == 2048
        assert len(Sums(1024)._cell) == 2048
        assert len(Sums(1025)._cell) == 4096


class TestSmallest:
    def test_a_tree_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one place"):
            Smallest(0)

    def test_a_place_never_set_holds_infinity(self) -> None:
        tree = Smallest(4)
        assert tree.least() == math.inf
        assert tree[2] == math.inf

    def test_it_is_the_size_it_was_asked_for(self) -> None:
        assert len(Smallest(5)) == 5

    def test_a_place_the_tree_does_not_hold_is_refused(self) -> None:
        tree = Smallest(3)
        with pytest.raises(IndexError, match="No place 3"):
            tree[3] = 1.0
        with pytest.raises(IndexError, match="No place -1"):
            _ = tree[-1]

    def test_a_negative_weight_is_refused(self) -> None:
        # The same weights go into both trees, so the two refuse the same
        # things. A weight the sums will not take is not one to hold a
        # minimum over either.
        tree = Smallest(3)
        with pytest.raises(ValueError, match="not negative"):
            tree[1] = -1.0

    def test_the_least_is_the_smallest_weight_set(self) -> None:
        tree = Smallest(4)
        for place, weight in enumerate([3.0, 1.0, 2.0]):
            tree[place] = weight
        assert tree.least() == 1.0

    def test_the_padding_does_not_pull_the_least_down(self) -> None:
        tree = Smallest(3)
        for place in range(3):
            tree[place] = 5.0
        assert tree.least() == 5.0

    def test_the_least_rises_when_the_place_holding_it_changes(self) -> None:
        tree = Smallest(3)
        for place, weight in enumerate([4.0, 1.0, 7.0]):
            tree[place] = weight
        assert tree.least() == 1.0
        tree[1] = 9.0
        assert tree.least() == 4.0

    def test_it_agrees_with_a_pass_over_the_weights(self) -> None:
        rng = Rng(11)
        weights = [rng.uniform(0.1, 3.0) for _ in range(37)]
        tree = Smallest(len(weights))
        for place, weight in enumerate(weights):
            tree[place] = weight
            assert tree.least() == min(weights[: place + 1])

        for _ in range(200):
            place = rng.below(len(weights))
            weights[place] = rng.uniform(0.1, 3.0)
            tree[place] = weights[place]
            assert tree.least() == min(weights)

    def test_it_says_what_it_holds(self) -> None:
        tree = Smallest(4)
        tree[0] = 2.5
        assert repr(tree) == "Smallest(4 places, least 2.5)"
