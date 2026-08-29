"""Tests for learning with no discount at all.

The one that matters is the last class. On a task where two policies collect 1
and 2 reward per step, a discounted agent below 0.7394 takes the first and this
one takes the second at every setting tried, because it has no discount to get
wrong. That is the whole reason the agent exists.

The rest hold the update to its arithmetic. A differential update is four terms
and every one of them can be dropped without the agent looking broken, so each
is checked by hand on a single step.
"""

from __future__ import annotations

import pytest

from rel.agents.average import DifferentialQ
from rel.agents.base import Transition
from rel.agents.dp import average_reward, value_iteration
from rel.envs.continuing import LONG, SHORT, two_loops
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import digest_of, train

TWO = Discrete(2)


def go(state: int, action: int, reward: float, landed: int) -> Transition[int]:
    return Transition(state, action, reward, landed, False, False)


def stop(state: int, action: int, reward: float, landed: int) -> Transition[int]:
    return Transition(state, action, reward, landed, True, False)


class TestTheSettings:
    def test_the_rate_of_the_average_is_not_below_zero(self) -> None:
        with pytest.raises(ValueError, match="not be below zero"):
            DifferentialQ(Rng(1), TWO, average_step=-0.1)

    def test_there_is_no_discount_to_give(self) -> None:
        """The point of the agent, as a signature.

        A discount that did nothing would be a setting that quietly means
        nothing, and somebody would set it and believe they had changed
        something.
        """
        with pytest.raises(TypeError, match="discount"):
            DifferentialQ(Rng(1), TWO, discount=0.9)  # type: ignore[call-arg]

    def test_the_rate_starts_at_nothing(self) -> None:
        assert DifferentialQ(Rng(1), TWO).average == 0.0


class TestOneUpdateByHand:
    """Four terms, worked out on paper, one step at a time."""

    def test_the_first_step_moves_the_cell_by_the_reward(self) -> None:
        # error = 2 - 0 + 0 - 0 = 2. The cell moves by 0.5 * 2 = 1.
        agent = DifferentialQ(Rng(1), TWO, step_size=0.5, average_step=0.0)
        agent.observe(go(0, 0, 2.0, 1))
        assert agent.peek(0)[0] == pytest.approx(1.0)

    def test_the_rate_moves_by_its_share_of_the_same_error(self) -> None:
        # error = 2, step 0.5, average_step 0.25, so the rate moves by 0.25.
        agent = DifferentialQ(Rng(1), TWO, step_size=0.5, average_step=0.25)
        agent.observe(go(0, 0, 2.0, 1))
        assert agent.average == pytest.approx(0.25)

    def test_the_rate_is_subtracted_from_the_reward(self) -> None:
        # With the rate already at 2, a reward of 2 into an empty row is an
        # error of zero and nothing moves at all.
        agent = DifferentialQ(Rng(1), TWO, step_size=0.5, average_step=0.25)
        agent.average = 2.0
        agent.observe(go(0, 0, 2.0, 1))
        assert agent.peek(0)[0] == pytest.approx(0.0)
        assert agent.average == pytest.approx(2.0)

    def test_the_best_action_ahead_is_the_one_it_bootstraps_on(self) -> None:
        # Off-policy: the larger of the two cells ahead, not the one taken.
        agent = DifferentialQ(Rng(1), TWO, step_size=1.0, average_step=0.0)
        agent.values(1)[0] = 3.0
        agent.values(1)[1] = 7.0
        agent.observe(go(0, 0, 0.0, 1))
        assert agent.peek(0)[0] == pytest.approx(7.0)

    def test_an_ending_drops_what_is_ahead(self) -> None:
        agent = DifferentialQ(Rng(1), TWO, step_size=1.0, average_step=0.0)
        agent.values(1)[0] = 7.0
        agent.observe(stop(0, 0, 0.0, 1))
        assert agent.peek(0)[0] == pytest.approx(0.0)

    def test_reading_ahead_makes_no_row(self) -> None:
        """The fault this project has had twice. `best_value` reads through
        `peek` and a landing cell must not gain a row from being landed in."""
        agent = DifferentialQ(Rng(1), TWO, step_size=0.5)
        agent.observe(go(0, 0, 1.0, 1))
        assert agent.knows(0)
        assert not agent.knows(1)


class TestWhatItReports:
    def test_the_rate_is_in_the_digest(self) -> None:
        """Two agents with the same table and different rates make different
        updates on the very next step, so they have not agreed."""
        one = DifferentialQ(Rng(1), TWO)
        other = DifferentialQ(Rng(1), TWO)
        one.values(0)[0] = 1.0
        other.values(0)[0] = 1.0
        other.average = 0.5
        assert digest_of(one) != digest_of(other)

    def test_it_says_what_it_is(self) -> None:
        assert "average=" in repr(DifferentialQ(Rng(1), TWO))


class TestNoDiscountMeansNoWrongDiscount:
    """The finding, on the environment built to hold it.

    Two policies collect 1 and 2 reward per step. A discounted agent picks the
    first below 0.7394 and the second above it, and that is the correct answer
    to the question the discount asked. This agent has no discount, so there is
    nothing to get wrong, and it takes the better loop at every setting.
    """

    SETTINGS = ((0.1, 0.01), (0.1, 0.5), (0.3, 0.1), (0.5, 0.05))

    @pytest.mark.parametrize(("step_size", "average_step"), SETTINGS)
    def test_it_takes_the_loop_that_pays_more_per_step(
        self, step_size: float, average_step: float
    ) -> None:
        root = Rng(1)
        env = two_loops(root.stream("env"))
        agent = DifferentialQ(
            root.stream("agent"),
            env.action_space,
            step_size=step_size,
            average_step=average_step,
        )
        train(env, agent, 20, discount=1.0)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        assert policy[0] == LONG
        assert average_reward(env, policy) == pytest.approx(env.per_step(LONG))

    @pytest.mark.parametrize(("step_size", "average_step"), SETTINGS)
    def test_the_rate_it_learns_is_the_rate_it_collects(
        self, step_size: float, average_step: float
    ) -> None:
        """Not just a bias term that happens to work.

        The number this agent subtracts is an estimate of something real, and
        the environment says exactly what that something is.
        """
        root = Rng(1)
        env = two_loops(root.stream("env"))
        agent = DifferentialQ(
            root.stream("agent"),
            env.action_space,
            step_size=step_size,
            average_step=average_step,
        )
        train(env, agent, 20, discount=1.0)
        assert agent.average == pytest.approx(env.per_step(LONG), abs=0.1)

    def test_a_discounted_agent_below_the_crossover_takes_the_other_one(
        self,
    ) -> None:
        """The other half of the comparison, from the model rather than a run.

        Nothing about learning is involved: this is the exactly optimal policy
        under that discount, and it collects half as much per step.
        """
        env = two_loops(Rng(1))
        chosen = list(value_iteration(env, discount=0.7).policy)
        assert chosen[0] == SHORT
        assert average_reward(env, chosen) == pytest.approx(env.per_step(SHORT))
