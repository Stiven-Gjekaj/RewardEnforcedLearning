"""Tests for the agents that keep a weight for each feature.

`TestTheCoderIsInterchangeable` is the load bearing one. The agent used to
name `TileCoder` and read `coder.grids`, and generalising it to take a radial
basis as well is only safe if every run over a tile coder still gives the
numbers it gave before. Those tests say why it does and check that it does.
"""

from __future__ import annotations

import pytest

from rel.agents import AGENTS
from rel.agents.base import RandomAgent, Transition
from rel.agents.basis import RadialBasis
from rel.agents.linear import SemiGradientQ, SemiGradientSarsa
from rel.agents.tiles import TileCoder
from rel.envs import ENVIRONMENTS
from rel.envs.control import CartPole, MountainCar
from rel.rng import Rng
from rel.spaces import Box, Discrete
from rel.training import digest_of, evaluate, train

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


class TestTheCoderIsInterchangeable:
    """The agent asks a coder four things and nothing else.

    None of them names a tile. That is what lets a radial basis run under the
    same agent, and the tests below are the reason to believe the change cost
    the tile coder nothing.
    """

    def test_a_tile_coder_reports_a_one_for_every_switch(self) -> None:
        coder = a_coder(grids=4)
        indices, values = coder.encode((0.5,))
        assert indices == coder.active((0.5,))
        assert values == (1.0, 1.0, 1.0, 1.0)

    def test_multiplying_by_one_changes_no_float(self) -> None:
        """Why the generalisation is exact and not merely close.

        The value used to be a sum of weights and is now a sum of weights
        each times its own value. Every value is one here, and a float times
        one is that float, so the terms and their order are what they were.
        """
        coder = a_coder(grids=8)
        agent = SemiGradientQ(Rng(1), TWO, coder)
        rng = Rng(7).stream("weights")
        agent.weights[0] = [rng.normal() for _ in range(coder.features)]

        row = agent.weights[0]
        direct = sum(row[switch] for switch in coder.active((0.31,)))
        assert agent.action_values((0.31,))[0] == direct

    def test_a_step_size_divided_by_the_grids_is_the_same_float(self) -> None:
        # It used to divide by `coder.grids`, an integer attribute. It now
        # divides by `coder.squared_length`, which for a tile coder is the
        # count of its ones. Dividing by 8 and by 8.0 is the same operation.
        coder = a_coder(grids=8)
        agent = SemiGradientQ(Rng(1), TWO, coder, step_size=0.3)
        _, values = coder.encode((0.5,))
        assert coder.squared_length(values) == 8.0
        assert agent.current_step_size(values) == 0.3 / 8

    def test_a_whole_run_over_a_tile_coder_is_unchanged(self) -> None:
        """The digests from before the agent took a `Coder` at all.

        Written down rather than compared against a second run, because a
        second run of the same code agrees with itself whatever either of
        them does. These four numbers were produced by the version that named
        `TileCoder` in its signature.
        """
        for name, path, learned in [
            ("mountaincar", "22bdc32f14a68951", "c721fa20534a5f44"),
            ("cartpole", "979877443e6c00a3", "54f776207a734d90"),
        ]:
            root = Rng(1)
            env = ENVIRONMENTS.make(name, root.stream("env"))
            agent = AGENTS.make("tile-sarsa", root.stream("agent"), env)
            record = train(env, agent, 20, discount=env.spec.suggested_discount)
            assert record.digest.hexdigest() == path, name
            assert digest_of(agent) == learned, name

    def test_a_radial_basis_runs_under_the_same_agent(self) -> None:
        basis = RadialBasis(UNIT, bins=4)
        agent = SemiGradientQ(Rng(1), TWO, basis)
        assert len(agent.weights[0]) == basis.features
        agent.observe(go(0.2, 0, 1.0, 0.8))
        assert agent.action_values((0.2,))[0] > 0.0

    def test_an_optimistic_start_is_the_value_asked_for_either_way(self) -> None:
        # The share of the optimism one feature carries is the coder's answer
        # to give: an eighth of it for eight grids, all of it for values that
        # add to one. Both come out at the number that was asked for.
        for coder in (a_coder(grids=8), RadialBasis(UNIT, bins=4)):
            agent = SemiGradientQ(Rng(1), TWO, coder, optimism=3.0)
            assert agent.action_values((0.5,)) == pytest.approx([3.0, 3.0])

    def test_the_value_still_moves_by_the_share_of_the_error_asked_for(self) -> None:
        # The promise the divisor exists to keep, asked of a coder whose
        # features are not all one. Nothing about it is special to tiles.
        agent = SemiGradientQ(Rng(1), TWO, RadialBasis(UNIT, bins=4), step_size=0.25)
        before = agent.action_values((0.3,))[0]
        agent.observe(ends(0.3, 0, 4.0, 0.9))
        after = agent.action_values((0.3,))[0]
        assert after - before == pytest.approx(0.25 * (4.0 - before))
