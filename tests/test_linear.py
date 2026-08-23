"""Tests for the agents that keep a weight for each switch."""

from __future__ import annotations

import pytest

from rel.agents.base import RandomAgent, Transition
from rel.agents.linear import SemiGradientQ, SemiGradientSarsa
from rel.agents.tiles import TileCoder
from rel.envs.control import CartPole, MountainCar
from rel.rng import Rng
from rel.spaces import Box, Discrete
from rel.training import evaluate, train

UNIT = Box([0.0], [1.0])
TWO = Discrete(2)


def a_coder(grids: int = 4) -> TileCoder:
    return TileCoder(UNIT, bins=4, grids=grids)


def go(
    here: float, action: int, reward: float, there: float
) -> Transition[tuple[float, ...]]:
    return Transition((here,), action, reward, (there,), False, False)


def ends(
    here: float, action: int, reward: float, there: float
) -> Transition[tuple[float, ...]]:
    return Transition((here,), action, reward, (there,), True, False)


def cut(
    here: float, action: int, reward: float, there: float
) -> Transition[tuple[float, ...]]:
    return Transition((here,), action, reward, (there,), False, True)


class TestTheValue:
    def test_an_unseen_state_is_worth_nothing(self) -> None:
        agent = SemiGradientQ(Rng(1), TWO, a_coder())
        assert agent.action_values((0.5,)) == [0.0, 0.0]

    def test_an_optimistic_start_is_the_value_asked_for(self) -> None:
        # The value is a sum over one weight per grid, so the optimistic value
        # has to be shared out between them. Putting the whole number in each
        # weight would make the starting value the number times the number of
        # grids, which is a different setting than the one asked for.
        agent = SemiGradientQ(Rng(1), TWO, a_coder(grids=8), optimism=3.0)
        assert agent.action_values((0.5,)) == [3.0, 3.0]

    def test_the_value_is_the_sum_over_the_active_switches(self) -> None:
        coder = a_coder(grids=4)
        agent = SemiGradientQ(Rng(1), TWO, coder)
        for switch in coder.active((0.5,)):
            agent.weights[0][switch] = 2.0
        assert agent.action_values((0.5,))[0] == 8.0


class TestTheStepSize:
    def test_the_value_moves_by_the_share_of_the_error_asked_for(self) -> None:
        # A step size of 0.5 has to move the value halfway to the target,
        # whatever the number of grids. It is divided by the number of grids
        # inside, because the value is a sum over that many weights.
        for grids in (1, 4, 16):
            agent = SemiGradientQ(
                Rng(1), TWO, a_coder(grids=grids), step_size=0.5, discount=1.0
            )
            agent.observe(ends(0.5, 0, 10.0, 0.7))
            assert agent.action_values((0.5,))[0] == pytest.approx(5.0), grids

    def test_two_updates_close_in_on_the_target(self) -> None:
        agent = SemiGradientQ(Rng(1), TWO, a_coder(), step_size=0.5, discount=1.0)
        agent.observe(ends(0.5, 0, 10.0, 0.7))
        agent.observe(ends(0.5, 0, 10.0, 0.7))
        assert agent.action_values((0.5,))[0] == pytest.approx(7.5)

    def test_only_the_action_that_was_taken_moves(self) -> None:
        agent = SemiGradientQ(Rng(1), TWO, a_coder(), step_size=0.5)
        agent.observe(ends(0.5, 1, 10.0, 0.7))
        assert agent.action_values((0.5,))[0] == 0.0
        assert agent.action_values((0.5,))[1] == pytest.approx(5.0)


class TestBootstrapping:
    def test_q_learning_takes_the_best_next_value(self) -> None:
        agent = SemiGradientQ(Rng(1), TWO, a_coder(), step_size=1.0, discount=1.0)
        for switch in agent.coder.active((0.9,)):
            agent.weights[1][switch] = 10.0 / agent.coder.grids

        agent.observe(go(0.1, 0, 0.0, 0.9))
        assert agent.action_values((0.1,))[0] == pytest.approx(10.0)

    def test_a_terminated_step_drops_the_future(self) -> None:
        agent = SemiGradientQ(Rng(1), TWO, a_coder(), step_size=1.0, discount=1.0)
        for switch in agent.coder.active((0.9,)):
            agent.weights[1][switch] = 99.0
        agent.observe(ends(0.1, 0, 5.0, 0.9))
        assert agent.action_values((0.1,))[0] == pytest.approx(5.0)

    def test_a_truncated_step_keeps_the_future(self) -> None:
        # On the cart pole the step limit is the best outcome the environment
        # has. Reporting it as an ending teaches the agent that the state it
        # was stopped in is worthless.
        agent = SemiGradientQ(Rng(1), TWO, a_coder(), step_size=1.0, discount=1.0)
        for switch in agent.coder.active((0.9,)):
            agent.weights[1][switch] = 10.0 / agent.coder.grids
        agent.observe(cut(0.1, 0, 5.0, 0.9))
        assert agent.action_values((0.1,))[0] == pytest.approx(15.0)


class TestSarsaWaits:
    def test_nothing_is_learned_until_the_next_action_is_known(self) -> None:
        agent = SemiGradientSarsa(Rng(1), TWO, a_coder(), step_size=1.0)
        agent.observe(go(0.1, 0, 5.0, 0.9))
        assert agent.action_values((0.1,))[0] == 0.0

    def test_it_uses_the_action_taken_and_not_the_one_it_would_choose(self) -> None:
        agent = SemiGradientSarsa(
            Rng(1), TWO, a_coder(), step_size=1.0, discount=1.0, epsilon=0.0
        )
        for switch in agent.coder.active((0.9,)):
            agent.weights[0][switch] = 100.0 / agent.coder.grids
            agent.weights[1][switch] = -100.0 / agent.coder.grids

        agent.observe(go(0.1, 0, 0.0, 0.9))
        agent.observe(go(0.9, 1, 0.0, 0.5))  # action 1, which greedy would not pick

        assert agent.action_values((0.1,))[0] == pytest.approx(-100.0)

    def test_the_last_step_is_not_left_unlearned(self) -> None:
        agent = SemiGradientSarsa(Rng(1), TWO, a_coder(), step_size=1.0)
        agent.observe(go(0.1, 0, -4.0, 0.9))
        agent.end_episode()
        assert agent.action_values((0.1,))[0] == pytest.approx(-4.0)


class TestThePolicy:
    def test_ties_are_broken_at_random(self) -> None:
        agent = SemiGradientQ(Rng(2), Discrete(3), a_coder())
        assert len({agent.greedy((0.5,)) for _ in range(200)}) == 3

    def test_a_clear_best_is_always_chosen(self) -> None:
        agent = SemiGradientQ(Rng(2), Discrete(3), a_coder())
        for switch in agent.coder.active((0.5,)):
            agent.weights[2][switch] = 1.0
        assert {agent.greedy((0.5,)) for _ in range(50)} == {2}

    def test_exploration_is_the_share_asked_for(self) -> None:
        agent = SemiGradientQ(Rng(3), Discrete(4), a_coder(), epsilon=0.5)
        for switch in agent.coder.active((0.5,)):
            agent.weights[0][switch] = 1.0
        draws = 4000
        chose_best = sum(1 for _ in range(draws) if agent.act((0.5,)) == 0)
        # Half greedy, plus a quarter of the other half.
        assert abs(chose_best / draws - 0.625) < 0.03


class TestItReallyLearns:
    def test_the_mountain_car_gets_out(self) -> None:
        # The random policy never leaves the valley in a thousand steps, so
        # this is the clearest separation in the project between an agent that
        # learned and one that did not.
        rng = Rng(11)
        env = MountainCar(rng.stream("env"))
        nothing = RandomAgent(rng.stream("agent"), env.action_space)
        baseline = train(env, nothing, 20)
        assert baseline.final(20) == -1000.0

        rng = Rng(11)
        env = MountainCar(rng.stream("env"))
        agent = SemiGradientSarsa(
            rng.stream("agent"),
            env.action_space,
            TileCoder(env.tiling_space, bins=8, grids=8),
            step_size=0.5,
            epsilon=0.0,
            discount=1.0,
        )
        record = train(env, agent, 200)
        assert record.final(50) > -250.0
        assert all(record.terminated[-20:])

    @pytest.mark.slow
    def test_the_cart_pole_stays_up(self) -> None:
        rng = Rng(11)
        env = CartPole(rng.stream("env"))
        agent = SemiGradientSarsa(
            rng.stream("agent"),
            env.action_space,
            TileCoder(env.tiling_space, bins=8, grids=8),
            step_size=0.5,
            epsilon=0.02,
            discount=1.0,
        )
        train(env, agent, 400)

        watched = evaluate(CartPole(Rng(12).stream("env")), agent, 20)
        assert watched.final() > env.spec.solved_return or 0.0
