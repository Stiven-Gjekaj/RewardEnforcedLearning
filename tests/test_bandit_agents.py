"""Tests for the agents that pull levers."""

from __future__ import annotations

import pytest

from rel.agents.bandit import (
    EpsilonGreedyBandit,
    GradientBandit,
    OptimisticBandit,
    UpperConfidenceBandit,
)
from rel.agents.base import Transition
from rel.envs.bandit import drifting_bandit, stationary_bandit
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

THREE = Discrete(3)


def pull(action: int, reward: float) -> Transition[int]:
    return Transition(0, action, reward, 0, False, False)


class TestEstimates:
    def test_the_average_is_the_average(self) -> None:
        agent = EpsilonGreedyBandit(Rng(1), THREE)
        agent.observe(pull(0, 4.0))
        assert agent.estimates[0] == 4.0
        agent.observe(pull(0, 6.0))
        assert agent.estimates[0] == 5.0
        agent.observe(pull(0, 8.0))
        assert agent.estimates[0] == 6.0

    def test_a_step_size_keeps_a_fixed_weight_on_the_newest_reward(self) -> None:
        agent = EpsilonGreedyBandit(Rng(1), THREE, step_size=0.5)
        agent.observe(pull(0, 4.0))
        assert agent.estimates[0] == 2.0
        agent.observe(pull(0, 4.0))
        assert agent.estimates[0] == 3.0

    def test_only_the_lever_pulled_moves(self) -> None:
        agent = EpsilonGreedyBandit(Rng(1), THREE)
        agent.observe(pull(1, 9.0))
        assert agent.estimates == [0.0, 9.0, 0.0]

    def test_the_counts_follow_the_pulls(self) -> None:
        agent = EpsilonGreedyBandit(Rng(1), THREE)
        for action in (0, 1, 1, 2, 1):
            agent.observe(pull(action, 0.0))
        assert agent.counts == [1, 3, 1]
        assert agent.pulls == 5


class TestANewProblemEachEpisode:
    def test_starting_an_episode_clears_what_was_learned(self) -> None:
        # Every episode of the bandit environment is a new set of levers. An
        # agent that carried its estimates over would be answering a question
        # about levers that no longer exist, and the curve would look far
        # better than the agent is.
        agent = EpsilonGreedyBandit(Rng(1), THREE)
        agent.observe(pull(0, 100.0))
        agent.start_episode()
        assert agent.estimates == [0.0, 0.0, 0.0]
        assert agent.counts == [0, 0, 0]

    def test_an_optimistic_agent_starts_optimistic_again(self) -> None:
        agent = OptimisticBandit(Rng(1), THREE, optimism=5.0)
        agent.observe(pull(0, 0.0))
        agent.start_episode()
        assert agent.estimates == [5.0, 5.0, 5.0]


class TestExploration:
    def test_epsilon_zero_never_wanders(self) -> None:
        agent = EpsilonGreedyBandit(Rng(1), THREE, epsilon=0.0)
        agent.estimates = [0.0, 9.0, 0.0]
        assert {agent.act(0) for _ in range(200)} == {1}

    def test_epsilon_one_always_wanders(self) -> None:
        agent = EpsilonGreedyBandit(Rng(1), THREE, epsilon=1.0)
        agent.estimates = [0.0, 9.0, 0.0]
        assert {agent.act(0) for _ in range(200)} == {0, 1, 2}

    def test_an_optimistic_agent_tries_everything_without_wandering(self) -> None:
        # No random exploration at all. Every lever starts above anything it
        # can really pay, so pulling one lowers it below the untried ones.
        agent = OptimisticBandit(Rng(1), THREE, optimism=5.0, step_size=1.0)
        agent.start_episode()
        tried = set()
        for _ in range(3):
            action = agent.act(0)
            tried.add(action)
            agent.observe(pull(action, 0.0))
        assert tried == {0, 1, 2}


class TestUpperConfidence:
    def test_it_tries_every_lever_before_it_repeats_one(self) -> None:
        agent = UpperConfidenceBandit(Rng(1), THREE)
        agent.start_episode()
        tried = []
        for _ in range(3):
            action = agent.act(0)
            tried.append(action)
            agent.observe(pull(action, 0.0))
        assert sorted(tried) == [0, 1, 2]

    def test_a_lever_tried_less_often_gets_a_larger_score(self) -> None:
        agent = UpperConfidenceBandit(Rng(1), THREE, confidence=2.0)
        agent.estimates = [1.0, 1.0, 1.0]
        agent.counts = [100, 100, 2]
        agent.pulls = 202
        assert agent.act(0) == 2

    def test_a_confidence_of_zero_is_plain_greedy(self) -> None:
        agent = UpperConfidenceBandit(Rng(1), THREE, confidence=0.0)
        agent.estimates = [1.0, 5.0, 1.0]
        agent.counts = [1, 1, 1]
        agent.pulls = 3
        assert {agent.act(0) for _ in range(50)} == {1}


class TestGradient:
    def test_the_probabilities_add_up_to_one(self) -> None:
        agent = GradientBandit(Rng(1), THREE)
        agent.estimates = [1.0, -2.0, 0.5]
        assert sum(agent.probabilities()) == pytest.approx(1.0)

    def test_a_larger_preference_is_more_likely(self) -> None:
        agent = GradientBandit(Rng(1), THREE)
        agent.estimates = [0.0, 3.0, -3.0]
        shares = agent.probabilities()
        assert shares[1] > shares[0] > shares[2]

    def test_adding_the_same_number_to_every_preference_changes_nothing(self) -> None:
        # A preference is not a value. Reading one as an estimate of what a
        # lever pays is the mistake this test names.
        agent = GradientBandit(Rng(1), THREE)
        agent.estimates = [1.0, -2.0, 0.5]
        before = agent.probabilities()
        agent.estimates = [value + 100.0 for value in agent.estimates]
        after = agent.probabilities()
        assert all(a == pytest.approx(b) for a, b in zip(before, after, strict=True))

    def test_a_huge_preference_does_not_overflow(self) -> None:
        # `exp` overflows above about seven hundred. Taking the largest
        # preference off first changes nothing and stops it.
        agent = GradientBandit(Rng(1), THREE)
        agent.estimates = [5000.0, 0.0, -5000.0]
        shares = agent.probabilities()
        assert shares[0] == pytest.approx(1.0)
        assert sum(shares) == pytest.approx(1.0)

    def test_a_reward_above_the_baseline_raises_the_preference(self) -> None:
        agent = GradientBandit(Rng(1), THREE, step_size=0.1)
        agent.start_episode()
        agent.observe(pull(0, 10.0))
        assert agent.estimates[0] > 0.0
        assert agent.estimates[1] < 0.0

    def test_a_reward_below_the_baseline_lowers_it(self) -> None:
        agent = GradientBandit(Rng(1), THREE, step_size=0.1)
        agent.start_episode()
        agent.observe(pull(0, 10.0))
        agent.observe(pull(0, -10.0))
        assert agent.estimates[0] < 0.0


class TestOnTheTestbed:
    """Numbers from running the agents, not from remembering a figure.

    Every one of these is 200 problems of 1000 pulls on seed 19, which is the
    command in docs/algorithms.md. The bounds are wide, because the point is
    the order the agents come in rather than the third decimal place.
    """

    @staticmethod
    def _share(builder: type, **options: object) -> float:
        rng = Rng(19)
        env = stationary_bandit(rng.stream("env"))
        agent = builder(rng.stream("agent"), env.action_space, **options)
        record = train(env, agent, 200)
        return record.final_audit(200)["best_arm_share"]

    def test_wandering_a_little_beats_never_wandering(self) -> None:
        assert self._share(EpsilonGreedyBandit, epsilon=0.1) > self._share(
            EpsilonGreedyBandit, epsilon=0.0
        )

    def test_upper_confidence_finds_the_best_lever_most_often(self) -> None:
        best = self._share(UpperConfidenceBandit)
        assert best > self._share(EpsilonGreedyBandit, epsilon=0.1)
        assert best > 0.7

    def test_optimism_alone_explores_enough(self) -> None:
        assert self._share(OptimisticBandit) > 0.65

    def test_the_gradient_agent_finds_it_too(self) -> None:
        assert self._share(GradientBandit, step_size=0.1) > 0.65


class TestFollowingAMovingTarget:
    def test_an_average_cannot_follow_and_a_step_size_can(self) -> None:
        # Exercise 2.5. Every lever starts equal and all of them wander. An
        # agent that averages every reward a lever ever paid is outvoted by
        # rewards from a lever that no longer exists.
        def share(step_size: float | None) -> float:
            rng = Rng(19)
            env = drifting_bandit(rng.stream("env"))
            agent = EpsilonGreedyBandit(
                rng.stream("agent"), env.action_space, epsilon=0.1, step_size=step_size
            )
            return train(env, agent, 20).final_audit(20)["best_arm_share"]

        assert share(0.1) > share(None) + 0.15


class TestTheBaselineInTheGradientAgent:
    def test_it_matters_when_every_reward_is_large(self) -> None:
        # Figure 2.5. The same problem with every true value moved up by four.
        # Nothing about which lever is best has changed, and an agent that
        # reasons about the size of a reward rather than the difference
        # between two of them is hurt by it.
        def share(centre: float, baseline: bool) -> float:
            rng = Rng(19)
            env = stationary_bandit(rng.stream("env"), centre=centre)
            agent = GradientBandit(
                rng.stream("agent"), env.action_space, step_size=0.1, baseline=baseline
            )
            return train(env, agent, 200).final_audit(200)["best_arm_share"]

        assert abs(share(0.0, True) - share(0.0, False)) < 0.05
        assert share(4.0, True) > share(4.0, False) + 0.15
