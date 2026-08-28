"""Tests for the ladder that varies how hard an agent optimises.

The claim this module exists to support is that the gap between what a reward
pays and what was wanted widens as an agent optimises harder. That claim rests
on the ladder being a real ordering of optimisation pressure, so most of what
is here holds the ladder to that rather than checking a number.
"""

from __future__ import annotations

import pytest

from rel.agents.dp import value_iteration
from rel.envs.classic import cliff_walk
from rel.envs.gaming import BoatRace, VaseRoom
from rel.pressure import LADDER, PressuredAgent, Rung, ladder, share
from rel.rng import Rng


class TestThePressuredAgent:
    def test_epsilon_zero_is_the_policy_it_was_given(self) -> None:
        env = cliff_walk(Rng(1).stream("env"))
        best = value_iteration(env, discount=1.0)
        agent = PressuredAgent(
            Rng(2).stream("agent"), env.action_space, best.policy, 0.0
        )
        for state in range(env.observation_space.n):
            assert agent.act(state) == best.policy[state]

    def test_epsilon_one_never_consults_the_policy(self) -> None:
        # Every action must appear, which a policy of a single action cannot
        # produce on its own.
        env = cliff_walk(Rng(1).stream("env"))
        one_action = [0] * env.observation_space.n
        agent = PressuredAgent(
            Rng(2).stream("agent"), env.action_space, one_action, 1.0
        )
        seen = {agent.act(0) for _ in range(200)}
        assert seen == set(range(env.action_space.n))

    def test_the_measured_policy_keeps_its_noise(self) -> None:
        # `greedy` is what an evaluation run calls. If it dropped the epsilon,
        # every rung of the ladder would measure the optimum and the ladder
        # would be flat by construction.
        env = cliff_walk(Rng(1).stream("env"))
        one_action = [0] * env.observation_space.n
        agent = PressuredAgent(
            Rng(2).stream("agent"), env.action_space, one_action, 1.0
        )
        seen = {agent.greedy(0) for _ in range(200)}
        assert len(seen) > 1

    def test_an_epsilon_outside_zero_to_one_is_refused(self) -> None:
        env = cliff_walk(Rng(1).stream("env"))
        policy = [0] * env.observation_space.n
        with pytest.raises(ValueError, match="probability"):
            PressuredAgent(Rng(2).stream("agent"), env.action_space, policy, 1.5)

    def test_more_epsilon_follows_the_policy_less_often(self) -> None:
        env = cliff_walk(Rng(1).stream("env"))
        one_action = [0] * env.observation_space.n

        def agreement(epsilon: float) -> float:
            agent = PressuredAgent(
                Rng(7).stream("agent"), env.action_space, one_action, epsilon
            )
            taken = [agent.act(0) for _ in range(600)]
            return taken.count(0) / len(taken)

        # Every rung follows the policy at least as often as the next one, and
        # the two ends are far enough apart that the ordering is not noise.
        shares = [agreement(epsilon) for epsilon in LADDER]
        assert shares == sorted(shares)
        assert shares[-1] - shares[0] > 0.5


class TestTheLadder:
    def test_the_ladder_runs_from_no_pressure_to_all_of_it(self) -> None:
        assert LADDER[0] == 1.0
        assert LADDER[-1] == 0.0

    def test_pressure_is_one_minus_epsilon(self) -> None:
        assert Rung(1.0, 0.0, {}).pressure == 0.0
        assert Rung(0.0, 0.0, {}).pressure == 1.0

    def test_every_rung_reports_the_reward_and_the_audit(self) -> None:
        rungs = ladder(
            lambda rng: VaseRoom(rng), 0.99, epsilons=(1.0, 0.0), episodes=3, seed=4
        )
        assert len(rungs) == 2
        for rung in rungs:
            assert isinstance(rung.paid, float)
            assert "vase_broken" in rung.audit

    def test_the_reward_rises_with_the_pressure(self) -> None:
        # This is the ladder doing its one job. The reward paid is what the
        # policy was solved for, so an agent that follows it more of the time
        # must collect more of it.
        rungs = ladder(
            lambda rng: BoatRace(rng),
            0.99,
            epsilons=(1.0, 0.5, 0.0),
            episodes=8,
            seed=5,
        )
        paid = [rung.paid for rung in rungs]
        assert paid == sorted(paid)
        assert paid[-1] > paid[0]

    def test_the_rungs_come_back_in_the_order_they_were_asked_for(self) -> None:
        asked = (0.9, 0.4, 0.0)
        rungs = ladder(
            lambda rng: VaseRoom(rng), 0.99, epsilons=asked, episodes=2, seed=6
        )
        assert tuple(rung.epsilon for rung in rungs) == asked

    def test_the_same_seed_gives_the_same_ladder(self) -> None:
        settings = {"epsilons": (0.5,), "episodes": 4, "seed": 9}
        first = ladder(lambda rng: VaseRoom(rng), 0.99, **settings)  # type: ignore[arg-type]
        again = ladder(lambda rng: VaseRoom(rng), 0.99, **settings)  # type: ignore[arg-type]
        assert [rung.paid for rung in first] == [rung.paid for rung in again]


class TestTheShare:
    def test_the_worst_is_zero_and_the_best_is_one(self) -> None:
        assert share(0.0, 0.0, 10.0) == 0.0
        assert share(10.0, 0.0, 10.0) == 1.0

    def test_a_value_between_them_is_between_them(self) -> None:
        assert share(2.5, 0.0, 10.0) == pytest.approx(0.25)

    def test_it_works_when_the_best_is_the_smaller_number(self) -> None:
        # The vase room pays a negative return, so the best number is the one
        # closest to zero and the arithmetic has to hold either way round.
        assert share(-4.0, -20.0, -4.0) == 1.0
        assert share(-20.0, -20.0, -4.0) == 0.0
        assert share(-12.0, -20.0, -4.0) == pytest.approx(0.5)

    def test_a_value_outside_the_two_ends_is_clamped(self) -> None:
        # A noisy policy can beat the worst case. Left alone that draws a chart
        # that leaves its own axis.
        assert share(-1.0, 0.0, 10.0) == 0.0
        assert share(12.0, 0.0, 10.0) == 1.0

    def test_two_ends_that_are_the_same_number_are_not_divided_by(self) -> None:
        assert share(5.0, 5.0, 5.0) == 0.0
