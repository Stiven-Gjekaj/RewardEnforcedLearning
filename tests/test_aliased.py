"""Tests for the corridor whose cells cannot be told apart.

The load bearing test is `TestTheAnswerAgainstTwoOtherWays`. The whole point
of this environment is that it has an exact answer while not being a
`TabularEnv`, so nothing else in this project checks that answer for it. The
closed form is therefore held against a solve of the three equations it came
from and against a run of the environment itself, neither of which shares a
line of code with it.
"""

from __future__ import annotations

import math
import statistics

import pytest

from rel.envs import ENVIRONMENTS
from rel.envs.aliased import ANYWHERE, CELLS, LEFT, RIGHT, SWITCHED, AliasedCorridor
from rel.rng import Rng


def build(**extra: int) -> AliasedCorridor:
    return AliasedCorridor(Rng(1).stream("env"), **extra)


def iterated(share: float, rounds: int = 200_000) -> float:
    """Expected steps from the start, by solving the three equations.

    Written out from the transitions rather than from the closed form, so an
    error in the algebra shows up as a disagreement.
    """
    values = [0.0, 0.0, 0.0]
    for _ in range(rounds):
        values = [
            1.0 + share * values[1] + (1.0 - share) * values[0],
            1.0 + share * values[0] + (1.0 - share) * values[2],
            1.0 + (1.0 - share) * values[1],
        ]
    return values[0]


def played(share: float, episodes: int, seed: int) -> float:
    """Mean steps to the goal over episodes really run."""
    env = AliasedCorridor(Rng(seed).stream("env"), steps=100_000)
    chance = Rng(seed).stream("policy")
    lengths = []
    for _ in range(episodes):
        env.reset()
        taken = 0
        while True:
            action = RIGHT if chance.chance(share) else LEFT
            taken += 1
            if env.step(action).terminated:
                break
        lengths.append(taken)
    return statistics.mean(lengths)


class TestTheAnswerAgainstTwoOtherWays:
    @pytest.mark.parametrize("share", [0.2, 0.4, 0.5857864376269049, 0.7, 0.9])
    def test_the_closed_form_solves_the_three_equations(self, share: float) -> None:
        assert AliasedCorridor.steps_from_start(share) == pytest.approx(iterated(share))

    @pytest.mark.parametrize("share", [0.4, 0.5857864376269049, 0.75])
    def test_the_closed_form_matches_running_it(self, share: float) -> None:
        wanted = AliasedCorridor.steps_from_start(share)
        assert played(share, 4000, 5) == pytest.approx(wanted, rel=0.06)

    def test_the_best_share_is_two_minus_the_root_of_two(self) -> None:
        assert AliasedCorridor.best_share() == pytest.approx(2.0 - math.sqrt(2.0))

    def test_the_best_steps_are_six_plus_four_roots_of_two(self) -> None:
        assert AliasedCorridor.best_steps() == pytest.approx(6.0 + 4.0 * math.sqrt(2))

    def test_the_best_share_really_is_the_smallest(self) -> None:
        # Swept rather than argued, because the derivative was done by hand.
        best = AliasedCorridor.best_share()
        wanted = AliasedCorridor.steps_from_start(best)
        for step in range(1, 1000):
            share = step / 1000.0
            assert AliasedCorridor.steps_from_start(share) >= wanted - 1e-12

    def test_the_best_steps_are_the_steps_at_the_best_share(self) -> None:
        assert AliasedCorridor.best_steps() == pytest.approx(
            AliasedCorridor.steps_from_start(AliasedCorridor.best_share())
        )


class TestNoFixedChoiceFinishes:
    def test_a_share_at_either_end_is_refused(self) -> None:
        for share in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="never reaches the goal"):
                AliasedCorridor.steps_from_start(share)

    def test_always_going_right_bounces_between_the_first_two(self) -> None:
        env = build()
        env.reset()
        seen = []
        for _ in range(20):
            env.step(RIGHT)
            seen.append(env.at)
        assert set(seen) == {0, 1}

    def test_always_going_left_stays_at_the_start(self) -> None:
        env = build()
        env.reset()
        for _ in range(20):
            env.step(LEFT)
            assert env.at == 0

    def test_neither_fixed_choice_reaches_the_goal(self) -> None:
        for action in (LEFT, RIGHT):
            env = build(steps=500)
            env.reset()
            for _ in range(500):
                assert not env.step(action).terminated


class TestWhatARankingCanReach:
    def test_it_is_the_better_of_the_two_ends(self) -> None:
        best = AliasedCorridor.best_ranking_share(0.1)
        assert best in (0.95, pytest.approx(0.05))
        assert AliasedCorridor.steps_from_start(best) == pytest.approx(44.2, abs=0.1)

    def test_it_is_far_from_what_the_best_policy_reaches(self) -> None:
        # Which is the point of the environment. The gap is the family of
        # methods, not a badly tuned one.
        ranking = AliasedCorridor.steps_from_start(
            AliasedCorridor.best_ranking_share(0.1)
        )
        assert ranking > 3.0 * AliasedCorridor.best_steps()

    def test_exploring_everything_lands_on_a_half(self) -> None:
        assert AliasedCorridor.best_ranking_share(1.0) == pytest.approx(0.5)

    def test_an_exploring_share_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="zero to one"):
            AliasedCorridor.best_ranking_share(1.5)

    def test_never_exploring_is_refused_by_the_arithmetic(self) -> None:
        # An agent that never explores takes one action for ever, and that
        # never finishes, so there is no number to return.
        with pytest.raises(ValueError, match="never reaches the goal"):
            AliasedCorridor.best_ranking_share(0.0)


class TestTheObservationHidesTheState:
    def test_every_cell_gives_the_same_observation(self) -> None:
        env = build()
        assert env.reset() == ANYWHERE
        seen = set()
        for _ in range(200):
            landed = env.step(RIGHT if env.rng.chance(0.6) else LEFT)
            seen.add(landed.observation)
            if landed.done:
                env.reset()
        assert seen == {ANYWHERE}

    def test_the_space_holds_one_observation(self) -> None:
        assert build().observation_space.n == 1

    def test_the_true_cell_still_moves(self) -> None:
        # The state is there. It is only the observation that hides it.
        env = build()
        env.reset()
        assert env.at == 0
        env.step(RIGHT)
        assert env.at == SWITCHED

    def test_it_is_not_a_tabular_environment(self) -> None:
        # Dynamic programming works over states and this agent is given
        # observations, so a model over the observation would be a model of
        # the confusion rather than of the corridor.
        from rel.core import TabularEnv

        assert not isinstance(build(), TabularEnv)


class TestTheMiddleIsReversed:
    def test_right_at_the_middle_goes_back(self) -> None:
        env = build()
        env.reset()
        env.step(RIGHT)
        assert env.at == SWITCHED
        env.step(RIGHT)
        assert env.at == 0

    def test_left_at_the_middle_goes_on(self) -> None:
        env = build()
        env.reset()
        env.step(RIGHT)
        env.step(LEFT)
        assert env.at == 2

    def test_right_at_the_last_cell_reaches_the_goal(self) -> None:
        env = build()
        env.reset()
        env.step(RIGHT)
        env.step(LEFT)
        assert env.step(RIGHT).terminated

    def test_the_shortest_route_is_three_steps(self) -> None:
        env = build()
        env.reset()
        assert not env.step(RIGHT).terminated
        assert not env.step(LEFT).terminated
        assert env.step(RIGHT).terminated


class TestWhatItReports:
    def test_every_step_pays_one(self) -> None:
        env = build()
        env.reset()
        for _ in range(30):
            landed = env.step(RIGHT if env.rng.chance(0.5) else LEFT)
            assert landed.reward == -1.0
            if landed.done:
                env.reset()

    def test_the_audit_says_whether_it_reached_the_goal(self) -> None:
        env = build()
        env.reset()
        env.step(RIGHT)
        env.step(LEFT)
        env.step(RIGHT)
        assert env.audit() == {"reached": 1.0}

    def test_a_run_that_was_cut_off_says_so(self) -> None:
        # Which the return cannot: every step pays -1, so an episode that
        # walked into the goal on its last step and one that was cut off are
        # both worth the same.
        env = build(steps=20)
        env.reset()
        for _ in range(20):
            env.step(LEFT)
        assert env.audit() == {"reached": 0.0}

    def test_a_cut_off_episode_is_truncated_and_not_terminated(self) -> None:
        env = build(steps=6)
        env.reset()
        for _ in range(5):
            assert not env.step(LEFT).done
        last = env.step(LEFT)
        assert last.truncated
        assert not last.terminated

    def test_it_draws_the_corridor(self) -> None:
        env = build()
        env.reset()
        assert env.render().startswith("A..|")
        env.step(RIGHT)
        assert env.render().startswith(".A.|")

    def test_it_says_where_it_is(self) -> None:
        env = build()
        env.reset()
        assert repr(env) == "AliasedCorridor(at cell 0)"


class TestTheRegistry:
    def test_it_is_there_under_a_name(self) -> None:
        env = ENVIRONMENTS.make("aliased", Rng(1).stream("env"))
        assert isinstance(env, AliasedCorridor)

    def test_it_carries_the_tag_that_says_what_it_is(self) -> None:
        assert set(ENVIRONMENTS["aliased"].tags) == {"aliased"}

    def test_the_corridor_is_three_cells(self) -> None:
        assert CELLS == 3
