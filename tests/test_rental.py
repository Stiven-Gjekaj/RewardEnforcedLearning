"""Tests for the two car parks and the van between them.

The load bearing tests are `TestThePoissonBranchesAddUp` and
`TestTheModelAgreesWithStepping`. Every other model in this project is a
handful of branches written out by hand; this one is built from a
distribution, and a distribution truncated at a car park's capacity is a
distribution that has to be made to add up again. The stepping draws from the
same tables, so the two cannot drift apart, and that is worth a test because
it is the kind of agreement that holds until somebody changes one of them.

Most tests run a small board. The book's is 441 states and eleven actions with
up to 441 branches each, which is a second of model building and half a minute
of sweeping, and none of what is checked here needs that size.
"""

from __future__ import annotations

import math

import pytest

from rel.agents.dp import value_iteration
from rel.envs.rental import CarRental, poisson
from rel.rng import Rng


def small(capacity: int = 4, van: int = 2, **options: object) -> CarRental:
    return CarRental(
        Rng(1).stream("env"),
        capacity=capacity,
        van=van,
        **options,  # type: ignore[arg-type]
    )


class TestThePoissonBranchesAddUp:
    """A Poisson has no largest count and a car park does."""

    def test_they_add_up_to_one(self) -> None:
        for rate in (0.5, 3.0, 4.0):
            for cap in (0, 1, 5, 20):
                assert sum(poisson(rate, cap)) == pytest.approx(1.0)

    def test_there_is_one_share_for_each_count_up_to_the_cap(self) -> None:
        assert len(poisson(3.0, 5)) == 6

    def test_the_shares_below_the_cap_are_the_formula(self) -> None:
        # Written out rather than trusted, because the folded tail at the end
        # is the only entry that is not the formula and a test that checked
        # the whole vector against itself would not say so.
        rate, cap = 3.0, 8
        got = poisson(rate, cap)
        for count in range(cap):
            want = math.exp(-rate) * rate**count / math.factorial(count)
            assert got[count] == pytest.approx(want)

    def test_the_last_count_carries_everything_past_it(self) -> None:
        # More requests than there are cars is the same day as exactly as
        # many requests as there are cars, so the tail belongs on the end
        # rather than thrown away.
        rate, cap = 3.0, 4
        got = poisson(rate, cap)
        below = sum(
            math.exp(-rate) * rate**count / math.factorial(count)
            for count in range(cap)
        )
        assert got[cap] == pytest.approx(1.0 - below)

    def test_a_cap_of_nothing_is_all_of_it_in_one_place(self) -> None:
        assert poisson(3.0, 0) == (1.0,)

    def test_no_share_is_negative(self) -> None:
        for cap in (0, 1, 3, 20, 40):
            assert all(share >= 0.0 for share in poisson(4.0, cap))


class TestFoldingAState:
    def test_a_state_for_every_pair_of_counts(self) -> None:
        assert small(capacity=4).observation_space.n == 25

    def test_folding_and_unfolding_come_back_to_the_same_place(self) -> None:
        env = small(capacity=4)
        for first in range(5):
            for second in range(5):
                assert env.unfold(env.fold(first, second)) == (first, second)

    def test_every_state_number_unfolds_to_a_pair_on_the_board(self) -> None:
        env = small(capacity=4)
        for state in range(env.observation_space.n):
            first, second = env.unfold(state)
            assert 0 <= first <= 4
            assert 0 <= second <= 4


class TestTheMove:
    def test_the_middle_action_moves_nothing_anywhere(self) -> None:
        env = small(capacity=4, van=2)
        for state in range(env.observation_space.n):
            assert env.moved(state, env.van) == 0

    def test_a_positive_move_takes_cars_to_the_first_location(self) -> None:
        env = small(capacity=4, van=2)
        assert env.moved(env.fold(0, 4), 2 + 2) == 2

    def test_a_negative_move_takes_them_the_other_way(self) -> None:
        env = small(capacity=4, van=2)
        assert env.moved(env.fold(4, 0), 2 - 2) == -2

    def test_no_car_moves_that_is_not_there(self) -> None:
        env = small(capacity=4, van=2)
        assert env.moved(env.fold(0, 1), 2 + 2) == 1
        assert env.moved(env.fold(1, 0), 2 - 2) == -1

    def test_no_car_moves_to_a_location_with_no_room(self) -> None:
        env = small(capacity=4, van=2)
        assert env.moved(env.fold(4, 4), 2 + 2) == 0
        # Two cars are there to move and only one space to move them into.
        assert env.moved(env.fold(3, 4), 2 + 2) == 1

    def test_several_actions_clip_onto_the_same_move(self) -> None:
        # Which is why a policy over this environment is read as moves rather
        # than as action numbers.
        env = small(capacity=4, van=2)
        state = env.fold(4, 4)
        assert {env.moved(state, action) for action in range(env.action_space.n)} == {0}


class TestTheModel:
    def test_every_pair_of_branches_adds_up(self) -> None:
        env = small(capacity=4, van=2)
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                total = sum(
                    branch.probability for branch in env.transitions(state, action)
                )
                assert total == pytest.approx(1.0), (state, action)

    def test_nothing_ever_ends(self) -> None:
        env = small(capacity=4)
        assert env.terminal_states() == frozenset()
        for state in range(env.observation_space.n):
            for branch in env.transitions(state, 0):
                assert not branch.terminated

    def test_moving_a_car_costs_the_same_whichever_way_it_goes(self) -> None:
        env = small(capacity=4, van=2, asked=(0.0001, 0.0001))
        left = env.transitions(env.fold(4, 0), env.van - 2)[0].reward
        right = env.transitions(env.fold(0, 4), env.van + 2)[0].reward
        assert left == pytest.approx(right)

    def test_a_move_that_clips_to_nothing_costs_nothing(self) -> None:
        env = small(capacity=4, van=2)
        moved = env.transitions(env.fold(4, 4), env.van + 2)[0].reward
        still = env.transitions(env.fold(4, 4), env.van)[0].reward
        assert moved == pytest.approx(still)

    def test_an_empty_pair_of_locations_takes_nothing(self) -> None:
        env = small(capacity=4, van=2)
        assert env.transitions(env.fold(0, 0), env.van)[0].reward == 0.0

    def test_more_cars_never_take_less(self) -> None:
        env = small(capacity=6, van=2)
        paid = [
            env.transitions(env.fold(here, 0), env.van)[0].reward for here in range(7)
        ]
        assert paid == sorted(paid)


class TestTheModelAgreesWithStepping:
    """The stepping draws from the tables the model lists.

    So a day the model gives no chance to cannot happen, and the two cannot
    drift apart. This drives the environment and checks that every landing it
    reaches is one the model named.
    """

    def test_every_landing_is_a_branch_the_model_named(self) -> None:
        env = small(capacity=4, van=2)
        env.reset()
        for _ in range(400):
            state = env.at
            action = env.rng.below(env.action_space.n)
            allowed = {
                branch.observation
                for branch in env.transitions(state, action)
                if branch.probability > 0.0
            }
            step = env.step(action)
            assert step.observation in allowed, (state, action)
            # It never ends inside the rules, so every episode runs to the
            # step limit and the run has to be started again to carry on.
            if step.truncated:
                env.reset()

    def test_the_takings_average_out_to_what_the_model_expects(self) -> None:
        # Not every step, because the model gives the expected takings and a
        # day gives one draw of them. Over enough days the two meet.
        env = small(capacity=4, van=2)
        env.reset()
        expected = env.transitions(env.fold(2, 2), env.van)[0].reward

        got = []
        for _ in range(4000):
            env.at = env.fold(2, 2)
            step = env.step(env.van)
            got.append(step.reward)
            if step.truncated:
                env.reset()
        assert sum(got) / len(got) == pytest.approx(expected, abs=0.3)

    def test_a_seed_replays_the_days(self) -> None:
        first, second = small(), small()
        first.reset()
        second.reset()
        for _ in range(50):
            assert first.step(2).observation == second.step(2).observation


class TestSolvingIt:
    def test_the_two_solvers_agree(self) -> None:
        from rel.agents.dp import policy_iteration

        env = small(capacity=4, van=2)
        swept = value_iteration(env, discount=0.9)
        improved = policy_iteration(env, discount=0.9)
        worst = max(
            abs(one - other)
            for one, other in zip(swept.values, improved.values, strict=True)
        )
        assert worst < 1e-4

    def test_it_moves_cars_towards_the_empty_location(self) -> None:
        """The shape of the book's figure, on a board small enough to sweep.

        Cars are asked for at both locations, so a location with none earns
        nothing and one that is full cannot hold the returns. The policy moves
        cars from the fuller side to the emptier one.
        """
        env = small(capacity=6, van=3)
        solved = value_iteration(env, discount=0.9)
        assert env.moved(env.fold(6, 0), solved.policy[env.fold(6, 0)]) < 0
        assert env.moved(env.fold(0, 6), solved.policy[env.fold(0, 6)]) > 0

    def test_a_full_board_is_worth_more_than_an_empty_one(self) -> None:
        env = small(capacity=4, van=2)
        solved = value_iteration(env, discount=0.9)
        assert solved.values[env.fold(4, 4)] > solved.values[env.fold(0, 0)]


class TestWhatItRefuses:
    def test_a_location_that_holds_nothing(self) -> None:
        with pytest.raises(ValueError, match="at least one car"):
            small(capacity=0)

    def test_a_van_larger_than_a_location(self) -> None:
        with pytest.raises(ValueError, match="between nothing and a full"):
            small(capacity=4, van=5)

    def test_a_van_that_holds_nothing_is_allowed(self) -> None:
        # One action, which moves nothing. The environment then measures the
        # takings with no van at all, which is the control for what the van
        # buys.
        env = small(capacity=4, van=0)
        assert env.action_space.n == 1
        assert env.moved(env.fold(4, 0), 0) == 0


class TestWhatTheRunReallyDid:
    def test_it_reports_nothing_before_a_day(self) -> None:
        env = small()
        env.reset()
        assert env.audit() == {}

    def test_it_counts_the_cars_rented_and_the_cars_moved(self) -> None:
        env = small(capacity=4, van=2)
        env.reset()
        env.step(env.van + 2)
        audit = env.audit()
        assert audit["moved_a_day"] == 2.0
        assert audit["rented_a_day"] >= 0.0

    def test_the_takings_cannot_tell_the_two_apart_and_this_can(self) -> None:
        # A day that rents ten and moves five pays what a day that rents nine
        # and moves none pays.
        env = small(capacity=4, van=2)
        env.reset()
        env.step(env.van)
        assert env.audit()["moved_a_day"] == 0.0


class TestItDraws:
    def test_both_locations_are_drawn(self) -> None:
        env = small(capacity=4)
        env.reset()
        drawn = env.render()
        assert "first" in drawn
        assert "second" in drawn

    def test_the_bars_fill_with_the_cars(self) -> None:
        env = small(capacity=4)
        env.reset()
        env.at = env.fold(4, 0)
        first, second = env.render().splitlines()
        assert first.count("#") == 4
        assert second.count("#") == 0


class TestTheRegistryEntry:
    def test_it_is_tabular_and_endless(self) -> None:
        from rel.envs import ENVIRONMENTS

        assert set(ENVIRONMENTS["rental"].tags) == {"tabular", "endless"}

    def test_it_is_built_at_the_book_numbers(self) -> None:
        from rel.envs import ENVIRONMENTS

        env = ENVIRONMENTS.make("rental", Rng(1).stream("env"))
        assert isinstance(env, CarRental)
        assert (env.capacity, env.van) == (20, 5)
        assert (env.asked, env.returned) == ((3.0, 4.0), (3.0, 2.0))

    def test_it_asks_to_be_discounted(self) -> None:
        # It never ends, so an undiscounted return is the return of a run that
        # goes on for ever.
        from rel.envs import ENVIRONMENTS

        env = ENVIRONMENTS.make("rental", Rng(1).stream("env"))
        assert not env.spec.ends
        assert env.spec.suggested_discount == 0.9
