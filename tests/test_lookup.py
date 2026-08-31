"""Tests for the coder that is handed its features instead of computing them.

Two of these are about `starting_weight`. It is the one method that a lookup
table cannot always answer, and the tests fix which cases it answers and which
it refuses, because a wrong answer there is silent: the run trains and the
optimistic start it was given was not the one it got.
"""

from __future__ import annotations

import pytest

from rel.agents.lookup import Lookup, aggregated

#: Baird's counterexample, whose rows are the whole of the counterexample.
#: Six upper states worth `2 * w[i] + w[7]` and one lower state worth
#: `w[6] + 2 * w[7]`.
BAIRD: list[list[float]] = [
    [2.0 if column == state else 0.0 for column in range(7)] + [1.0]
    for state in range(6)
] + [[0.0] * 6 + [1.0, 2.0]]

#: A thousand states in four groups, which is the friendly use of the same
#: class: every state in a group shares every bit of what any of them learns.
GROUPS: list[list[float]] = [
    [1.0 if group == state // 250 else 0.0 for group in range(4)]
    for state in range(1000)
]


class TestTheShapeOfTheTable:
    def test_the_feature_count_is_the_width_of_a_row(self) -> None:
        assert Lookup(BAIRD).features == 8

    def test_the_state_count_is_the_number_of_rows(self) -> None:
        assert Lookup(BAIRD).states == 7

    def test_a_table_with_no_rows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one state"):
            Lookup([])

    def test_a_table_with_no_features_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one feature"):
            Lookup([[], []])

    def test_rows_of_different_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\[2, 3\]"):
            Lookup([[1.0, 0.0], [1.0, 0.0, 0.0]])

    def test_the_rows_are_copied_rather_than_held(self) -> None:
        rows = [[1.0, 0.0], [0.0, 1.0]]
        coder = Lookup(rows)
        rows[0][0] = 99.0
        assert coder.encode(0) == ((0,), (1.0,))

    def test_whole_numbers_become_floats(self) -> None:
        coder = Lookup([[1, 0], [0, 2]])
        assert coder.encode(1) == ((1,), (2.0,))

    def test_it_says_its_shape(self) -> None:
        assert repr(Lookup(BAIRD)) == "Lookup(states=7, features=8)"


class TestWhatEncodingGivesBack:
    def test_a_row_comes_back_without_its_zeros(self) -> None:
        assert Lookup(BAIRD).encode(0) == ((0, 7), (2.0, 1.0))

    def test_the_last_row_of_bairds_table_is_the_odd_one(self) -> None:
        assert Lookup(BAIRD).encode(6) == ((6, 7), (1.0, 2.0))

    def test_a_group_lights_one_feature(self) -> None:
        coder = Lookup(GROUPS)
        assert coder.encode(0) == ((0,), (1.0,))
        assert coder.encode(249) == ((0,), (1.0,))
        assert coder.encode(250) == ((1,), (1.0,))

    def test_a_state_below_the_table_is_an_error(self) -> None:
        with pytest.raises(IndexError, match="outside a table of 7 states"):
            Lookup(BAIRD).encode(-1)

    def test_a_state_above_the_table_is_an_error(self) -> None:
        with pytest.raises(IndexError, match="outside a table of 7 states"):
            Lookup(BAIRD).encode(7)

    def test_a_negative_feature_is_kept(self) -> None:
        # Only zero is dropped. A feature of minus one is a feature.
        assert Lookup([[-1.0, 0.0, 3.0]]).encode(0) == ((0, 2), (-1.0, 3.0))


class TestTheStepSize:
    def test_the_squared_length_of_bairds_rows_is_five_everywhere(self) -> None:
        # Both kinds of row: two squared plus one, and one plus two squared.
        # It matters that they agree, because it makes the step size this
        # project divides a constant times the one the counterexample uses.
        coder = Lookup(BAIRD)
        lengths = {coder.squared_length(coder.encode(state)[1]) for state in range(7)}
        assert lengths == {5.0}

    def test_a_group_has_a_squared_length_of_one(self) -> None:
        coder = Lookup(GROUPS)
        assert coder.squared_length(coder.encode(500)[1]) == 1.0


class TestTheOptimisticStart:
    def test_no_optimism_is_no_weight(self) -> None:
        assert Lookup(BAIRD).starting_weight(0.0) == 0.0

    def test_rows_that_agree_share_the_value_out(self) -> None:
        # Every row of Baird's table adds up to three, so a third of the value
        # in each weight makes every state worth the value.
        coder = Lookup(BAIRD)
        assert coder.starting_weight(3.0) == 1.0

    def test_a_shared_weight_really_reaches_the_value(self) -> None:
        coder = Lookup(BAIRD)
        share = coder.starting_weight(6.0)
        weights = [share] * coder.features
        for state in range(coder.states):
            indices, values = coder.encode(state)
            worth = sum(
                weights[index] * value
                for index, value in zip(indices, values, strict=True)
            )
            assert worth == pytest.approx(6.0)

    def test_rows_that_disagree_are_refused_rather_than_averaged(self) -> None:
        with pytest.raises(ValueError, match="do not add up to the same number"):
            Lookup([[1.0, 0.0], [1.0, 1.0]]).starting_weight(1.0)

    def test_rows_that_disagree_still_answer_zero(self) -> None:
        # Nothing is ambiguous about starting at nothing, so the refusal above
        # must not reach the case every caller in this project uses.
        assert Lookup([[1.0, 0.0], [1.0, 1.0]]).starting_weight(0.0) == 0.0

    def test_a_table_of_zeros_says_so_rather_than_dividing(self) -> None:
        with pytest.raises(ValueError, match="worth zero whatever"):
            Lookup([[0.0, 0.0], [0.0, 0.0]]).starting_weight(1.0)


class TestGroupingStatesTogether:
    """The friendly use of this class, and the smallest approximation there is.

    What is worth holding is the shape of the groups. A split that piled the
    remainder onto one end would give that group more states to average over
    than any other, and the staircase it draws would have one step of a
    different width, which is not what a reader of a group count expects.
    """

    def test_every_state_is_in_exactly_one_group(self) -> None:
        coder = aggregated(1000, 10)
        for state in range(1000):
            indices, values = coder.encode(state)
            assert len(indices) == 1
            assert values == (1.0,)

    def test_the_weight_count_is_the_group_count(self) -> None:
        assert aggregated(1000, 10).features == 10
        assert aggregated(1000, 250).states == 1000

    def test_the_groups_are_in_order(self) -> None:
        groups = [aggregated(20, 4).encode(state)[0][0] for state in range(20)]
        assert groups == sorted(groups)

    def test_states_that_divide_evenly_give_even_groups(self) -> None:
        groups = [aggregated(12, 3).encode(state)[0][0] for state in range(12)]
        assert groups == [0] * 4 + [1] * 4 + [2] * 4

    def test_a_remainder_is_spread_rather_than_piled_on_the_end(self) -> None:
        groups = [aggregated(11, 3).encode(state)[0][0] for state in range(11)]
        assert groups == [0] * 4 + [1] * 4 + [2] * 3

        sizes = [groups.count(group) for group in range(3)]
        assert max(sizes) - min(sizes) <= 1

    def test_every_group_holds_at_least_one_state(self) -> None:
        for states, groups in ((11, 3), (1000, 7), (100, 99), (5, 5)):
            found = {aggregated(states, groups).encode(s)[0][0] for s in range(states)}
            assert found == set(range(groups)), (states, groups)

    def test_one_group_is_one_number_for_everything(self) -> None:
        coder = aggregated(50, 1)
        assert coder.features == 1
        assert {coder.encode(state) for state in range(50)} == {((0,), (1.0,))}

    def test_a_group_for_each_state_is_a_table(self) -> None:
        coder = aggregated(6, 6)
        assert [coder.encode(state)[0][0] for state in range(6)] == list(range(6))

    def test_more_groups_than_states_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no more of them than there are states"):
            aggregated(5, 6)

    def test_no_groups_at_all_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one group"):
            aggregated(5, 0)

    def test_nothing_to_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to group"):
            aggregated(0, 1)

    def test_an_optimistic_start_works_because_the_rows_agree(self) -> None:
        # Every row adds up to one, so the share is the value itself.
        assert aggregated(100, 5).starting_weight(0.4) == pytest.approx(0.4)
