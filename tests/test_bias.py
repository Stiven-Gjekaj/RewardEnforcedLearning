"""Tests for the two state problem where a maximum over noise points the wrong
way.

The load bearing test is `TestTheAnswerIsNotInDoubt`. This environment exists
so that an agent going left can be called a mistake rather than a preference,
and that rests on the arithmetic of the model rather than on the name of the
example. Value iteration over the transitions has to say right, and the value
of left has to be the mean of B's reward exactly.
"""

from __future__ import annotations

import statistics

import pytest

from rel.agents.dp import value_iteration
from rel.envs import ENVIRONMENTS
from rel.envs.bias import END, LEFT, MIDDLE, START, MaximisationBias
from rel.rng import Rng


def build(**extra: float | int) -> MaximisationBias:
    return MaximisationBias(Rng(1).stream("env"), **extra)  # type: ignore[arg-type]


class TestTheAnswerIsNotInDoubt:
    def test_value_iteration_goes_right_from_the_start(self) -> None:
        env = build()
        answer = value_iteration(env, discount=1.0)
        assert answer.policy[START] != LEFT

    def test_the_best_possible_return_is_nothing(self) -> None:
        answer = value_iteration(build(), discount=1.0)
        assert answer.values[START] == pytest.approx(0.0)

    def test_going_left_is_worth_the_mean_of_the_gamble(self) -> None:
        env = build()
        answer = value_iteration(env, discount=1.0)
        assert answer.values[MIDDLE] == pytest.approx(env.mean)

    def test_the_gap_is_the_mean_and_nothing_else(self) -> None:
        # Which is what makes this a clean example: an agent that goes left is
        # wrong by exactly a tenth, and by nothing to do with the horizon.
        env = build()
        answer = value_iteration(env, discount=1.0)
        assert answer.values[START] - answer.values[MIDDLE] == pytest.approx(0.1)


class TestTheGamble:
    def test_it_has_the_mean_the_book_gives(self) -> None:
        env = build()
        branches = env.transitions(MIDDLE, 0)
        mean = sum(one.probability * one.reward for one in branches)
        assert mean == pytest.approx(-0.1)

    def test_it_has_the_spread_the_book_gives(self) -> None:
        # Two points rather than a normal, and the same standard deviation.
        env = build()
        branches = env.transitions(MIDDLE, 0)
        mean = sum(one.probability * one.reward for one in branches)
        variance = sum(one.probability * (one.reward - mean) ** 2 for one in branches)
        assert variance**0.5 == pytest.approx(1.0)

    def test_every_action_at_the_middle_is_the_same_gamble(self) -> None:
        # The ten of them differ in the agent's estimates and in nothing else.
        env = build()
        first = env.transitions(MIDDLE, 0)
        for action in env.action_space:
            assert env.transitions(MIDDLE, action) == first

    def test_stepping_pays_one_of_the_two_and_averages_to_the_mean(self) -> None:
        env = build()
        paid = []
        for _ in range(4000):
            env.reset()
            env.step(LEFT)
            paid.append(env.step(0).reward)
        assert set(paid) == {-1.1, 0.9}
        assert statistics.mean(paid) == pytest.approx(-0.1, abs=0.05)


class TestTheShape:
    def test_left_goes_to_the_middle_without_paying(self) -> None:
        env = build()
        assert env.reset() == START
        landed = env.step(LEFT)
        assert landed.observation == MIDDLE
        assert landed.reward == 0.0
        assert not landed.terminated

    def test_every_other_action_ends_it_with_nothing(self) -> None:
        env = build()
        for action in range(1, env.action_space.n):
            env.reset()
            landed = env.step(action)
            assert landed.observation == END
            assert landed.reward == 0.0
            assert landed.terminated

    def test_the_middle_always_ends_the_episode(self) -> None:
        env = build()
        for action in env.action_space:
            env.reset()
            env.step(LEFT)
            assert env.step(action).terminated

    def test_the_end_is_the_only_terminal_state(self) -> None:
        assert build().terminal_states() == frozenset({END})

    def test_the_model_matches_what_stepping_does(self) -> None:
        # Every branch the model names has to be a branch the stepping takes,
        # and the shares have to match. A model that drifts from the code is
        # worse than no model.
        env = build()
        counted: dict[float, int] = {}
        for _ in range(6000):
            env.reset()
            env.step(LEFT)
            reward = env.step(0).reward
            counted[reward] = counted.get(reward, 0) + 1

        for branch in env.transitions(MIDDLE, 0):
            share = counted.get(branch.reward, 0) / 6000
            assert share == pytest.approx(branch.probability, abs=0.03)


class TestWhatItRefuses:
    def test_a_start_with_no_right_turn_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one right"):
            MaximisationBias(Rng(1), actions=1)

    def test_a_gamble_with_no_noise_is_refused(self) -> None:
        # Without noise there is nothing for a maximum to be biased about, so
        # the environment would be an example of nothing.
        with pytest.raises(ValueError, match="no noise"):
            MaximisationBias(Rng(1), spread=0.0)

    def test_a_left_turn_that_pays_is_refused(self) -> None:
        with pytest.raises(ValueError, match="wrong answer"):
            MaximisationBias(Rng(1), mean=0.1)


class TestWhatItReports:
    def test_the_audit_says_whether_the_episode_went_left(self) -> None:
        env = build()
        env.reset()
        env.step(LEFT)
        env.step(0)
        assert env.audit() == {"went_left": 1.0}

    def test_it_says_nothing_before_an_episode_has_run(self) -> None:
        assert build().audit() == {}

    def test_going_right_reports_a_nothing(self) -> None:
        env = build()
        env.reset()
        env.step(1)
        assert env.audit() == {"went_left": 0.0}

    def test_the_audit_is_reset_with_the_episode(self) -> None:
        env = build()
        env.reset()
        env.step(LEFT)
        env.step(0)
        env.reset()
        env.step(1)
        assert env.audit() == {"went_left": 0.0}

    def test_it_names_the_left_turn_and_numbers_the_rest(self) -> None:
        env = build(actions=3)
        assert env.action_names == ("left", "right 1", "right 2")

    def test_it_says_where_it_is(self) -> None:
        env = build()
        env.reset()
        assert env.render() == "at A"
        env.step(LEFT)
        assert env.render() == "at B, went left"

    def test_it_says_what_it_was_built_with(self) -> None:
        assert repr(build(actions=4)) == (
            "MaximisationBias(actions=4, mean=-0.1, spread=1)"
        )


class TestTheRegistry:
    def test_it_is_there_under_a_name(self) -> None:
        env = ENVIRONMENTS.make("bias", Rng(1).stream("env"))
        assert isinstance(env, MaximisationBias)

    def test_it_is_tabular_and_it_ends(self) -> None:
        assert set(ENVIRONMENTS["bias"].tags) == {"tabular"}
        assert ENVIRONMENTS.make("bias", Rng(1).stream("env")).spec.ends
