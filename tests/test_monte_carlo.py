"""Tests for learning from whole episodes."""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.monte_carlo import MonteCarloControl
from rel.envs.classic import cliff_walk
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

TWO = Discrete(2)


def go(state: int, action: int, reward: float, landed: int) -> Transition[int, int]:
    return Transition(state, action, reward, landed, False, False)


def ends(state: int, action: int, reward: float, landed: int) -> Transition[int, int]:
    return Transition(state, action, reward, landed, True, False)


def play(agent: MonteCarloControl, steps: list[Transition[int, int]]) -> None:
    for step in steps:
        agent.observe(step)
    agent.end_episode()


class TestTheReturn:
    def test_a_state_is_credited_with_the_rest_of_the_episode(self) -> None:
        agent = MonteCarloControl(Rng(1), TWO, discount=1.0)
        play(agent, [go(0, 0, 1.0, 1), go(1, 0, 2.0, 2), ends(2, 0, 4.0, 3)])

        assert agent.q[0][0] == pytest.approx(7.0)
        assert agent.q[1][0] == pytest.approx(6.0)
        assert agent.q[2][0] == pytest.approx(4.0)

    def test_the_discount_reaches_back(self) -> None:
        agent = MonteCarloControl(Rng(1), TWO, discount=0.5)
        play(agent, [go(0, 0, 1.0, 1), go(1, 0, 2.0, 2), ends(2, 0, 4.0, 3)])

        # 1 + 0.5 * 2 + 0.25 * 4
        assert agent.q[0][0] == pytest.approx(3.0)

    def test_nothing_is_learned_until_the_episode_is_over(self) -> None:
        agent = MonteCarloControl(Rng(1), TWO)
        agent.observe(go(0, 0, 5.0, 1))
        assert 0 not in agent.q or agent.q[0][0] == 0.0

    def test_only_the_first_visit_is_credited(self) -> None:
        # State zero is reached twice. First visit counts the whole episode
        # once. Counting it again with the shorter return would pull the
        # estimate towards the tail of the episode.
        agent = MonteCarloControl(Rng(1), TWO, discount=1.0)
        play(
            agent,
            [go(0, 0, 1.0, 1), go(1, 0, 0.0, 0), go(0, 0, 1.0, 2), ends(2, 0, 0.0, 3)],
        )

        assert agent.visits[(0, 0)] == 1
        assert agent.q[0][0] == pytest.approx(2.0)


class TestTheStepSize:
    def test_with_no_step_size_it_averages_every_return(self) -> None:
        agent = MonteCarloControl(Rng(1), TWO)
        play(agent, [ends(0, 0, 10.0, 1)])
        assert agent.q[0][0] == pytest.approx(10.0)
        play(agent, [ends(0, 0, 0.0, 1)])
        assert agent.q[0][0] == pytest.approx(5.0)
        play(agent, [ends(0, 0, 0.0, 1)])
        assert agent.q[0][0] == pytest.approx(10.0 / 3.0)

    def test_with_a_step_size_the_newest_return_keeps_its_weight(self) -> None:
        agent = MonteCarloControl(Rng(1), TWO, step_size=0.5)
        play(agent, [ends(0, 0, 10.0, 1)])
        assert agent.q[0][0] == pytest.approx(5.0)
        play(agent, [ends(0, 0, 10.0, 1)])
        assert agent.q[0][0] == pytest.approx(7.5)

    def test_a_fixed_step_size_learns_control_faster(self) -> None:
        # The average is right for a fixed target and wrong for a moving one,
        # and in control the target moves: a return from three hundred
        # episodes ago was collected by a policy that no longer exists.
        def final(step_size: float | None) -> float:
            rng = Rng(7)
            env = cliff_walk(rng.stream("env"))
            agent = MonteCarloControl(
                rng.stream("agent"), env.action_space, epsilon=0.1, step_size=step_size
            )
            return train(env, agent, 500).final()

        assert final(0.1) > final(None) + 50.0


class TestTruncation:
    def test_a_truncated_episode_keeps_a_tail(self) -> None:
        # Monte Carlo needs the rest of the episode and a step limit takes it
        # away. Treating the limit as an ending would say the future is worth
        # nothing, which is the fault the whole project warns about, so the
        # tail is filled in with what the agent believes.
        agent = MonteCarloControl(Rng(1), TWO, discount=1.0, epsilon=0.0)
        agent.q[9] = [4.0, 0.0]

        agent.observe(Transition(0, 0, 1.0, 9, False, True))
        agent.end_episode()

        assert agent.q[0][0] == pytest.approx(5.0)

    def test_an_ended_episode_has_no_tail(self) -> None:
        agent = MonteCarloControl(Rng(1), TWO, discount=1.0, epsilon=0.0)
        agent.q[9] = [4.0, 0.0]

        agent.observe(Transition(0, 0, 1.0, 9, True, False))
        agent.end_episode()

        assert agent.q[0][0] == pytest.approx(1.0)


def test_it_learns_the_cliff_walk() -> None:
    rng = Rng(3)
    env = cliff_walk(rng.stream("env"))
    agent = MonteCarloControl(
        rng.stream("agent"), env.action_space, epsilon=0.1, step_size=0.1
    )
    record = train(env, agent, 600)
    assert record.final(100) > -60.0
