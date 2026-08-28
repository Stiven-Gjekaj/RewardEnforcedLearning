"""Tests for n-step tree backup.

The strongest tests here are the two collapses at n of one, because they are
exact rather than approximate. With a greedy target the method has to be
Q-learning cell for cell, and with the exploring policy as its target it has to
be expected SARSA. A method that did not collapse to its one step form would be
a different algorithm wearing the name.

The property the method exists for is that it needs no importance ratio, and
that is structural rather than numerical: there is no division anywhere in the
update. A test reads the source and says so.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from rel.agents.base import Transition
from rel.agents.dp import evaluate_policy
from rel.agents.td import ExpectedSarsa, NStepSarsa, QLearning
from rel.agents.tree import TARGETS, TreeBackup
from rel.envs.classic import cliff_walk
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

WALK = (
    Transition(0, 0, -1.0, 1, terminated=False, truncated=False),
    Transition(1, 1, -1.0, 2, terminated=False, truncated=False),
    Transition(2, 0, -1.0, 3, terminated=False, truncated=False),
    Transition(3, 1, 10.0, 4, terminated=True, truncated=False),
)


def feed(agent) -> None:  # type: ignore[no-untyped-def]
    agent.start_episode()
    for step in WALK:
        agent.observe(step)
    agent.end_episode()


class TestTheCollapseAtOneStep:
    def test_a_greedy_target_at_n_of_one_is_q_learning(self) -> None:
        plain: QLearning[int] = QLearning(
            Rng(1).stream("a"), Discrete(2), step_size=0.5, discount=0.9, epsilon=0.1
        )
        tree: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"),
            Discrete(2),
            n=1,
            target="greedy",
            step_size=0.5,
            discount=0.9,
            epsilon=0.1,
        )
        feed(plain)
        feed(tree)
        assert tree.q == plain.q

    def test_the_exploring_policy_as_target_at_n_of_one_is_expected_sarsa(self) -> None:
        plain: ExpectedSarsa[int] = ExpectedSarsa(
            Rng(1).stream("a"), Discrete(2), step_size=0.5, discount=0.9, epsilon=0.1
        )
        tree: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"),
            Discrete(2),
            n=1,
            target="policy",
            step_size=0.5,
            discount=0.9,
            epsilon=0.1,
        )
        feed(plain)
        feed(tree)
        assert tree.q == plain.q

    def test_more_steps_reach_further_back(self) -> None:
        # The first cell of the walk is three steps from the reward, so one
        # step of anything cannot have moved it after a single episode.
        short: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"), Discrete(2), n=1, step_size=0.5, epsilon=0.0
        )
        long: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"), Discrete(2), n=4, step_size=0.5, epsilon=0.0
        )
        feed(short)
        feed(long)
        assert long.q[0][0] != short.q[0][0]


class TestTheRecursion:
    """The body of the loop, pinned by a target worked out by hand.

    The two collapses at n of one do not reach this. At one step the window is
    a single step and the recursion never runs, so a fault inside it passes
    both of them and is caught only by an agent failing to learn.
    """

    def test_a_two_step_target_is_what_it_should_be(self) -> None:
        # Two steps, a discount of a half and a step size of one, so the cell
        # ends up holding the target itself.
        #
        # At state 1 the greedy action is 1 and the agent takes 0, so the
        # target policy gives what really happened next a probability of zero.
        # The value of the action it would have taken carries the return
        # instead:
        #
        #     1 + 0.5 * (1.0 * 8.0 + 0.0 * 99.0) = 5.0
        #
        # The reward of 99 is collected and contributes nothing, which is the
        # whole mechanism in one number.
        agent: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"),
            Discrete(2),
            n=2,
            target="greedy",
            step_size=1.0,
            discount=0.5,
            epsilon=0.0,
        )
        agent.values(1)[:] = [2.0, 8.0]

        agent.start_episode()
        agent.observe(Transition(0, 0, 1.0, 1, terminated=False, truncated=False))
        agent.observe(Transition(1, 0, 99.0, 2, terminated=True, truncated=False))
        agent.end_episode()

        assert agent.q[0][0] == pytest.approx(5.0)
        # The step that explored is credited with its own reward in full.
        assert agent.q[1][0] == pytest.approx(99.0)


class TestItNeedsNoRatio:
    def test_the_update_divides_by_nothing(self) -> None:
        # The whole point of the method. An importance ratio is a division, and
        # the variance it is known for comes from dividing by a probability
        # that can be small. There is no division in this update at all, and
        # this reads the source rather than trusting the docstring.
        source = inspect.getsource(TreeBackup._update)
        tree = ast.parse(source.strip())
        divisions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div | ast.FloorDiv)
        ]
        assert divisions == []

    def test_it_learns_about_a_policy_it_is_not_following(self) -> None:
        # The behaviour policy explores and the target policy does not, so the
        # two differ at every state where exploring is possible.
        agent: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"), Discrete(3), target="greedy", epsilon=0.3
        )
        agent.values(0)[:] = [1.0, 5.0, 2.0]
        assert agent.target_probabilities(0) == [0.0, 1.0, 0.0]
        assert min(agent.policy_probabilities(0)) > 0.0

    def test_a_target_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ValueError, match="target is one of"):
            TreeBackup(Rng(1).stream("a"), Discrete(2), target="hopeful")  # type: ignore[arg-type]


class TestTheEndings:
    def test_a_terminated_episode_bootstraps_from_nothing(self) -> None:
        # The last reward of the walk is 10 and the episode ends there, so the
        # cell it was collected from is worth exactly that much of a step size.
        agent: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"), Discrete(2), n=1, step_size=1.0, discount=0.9
        )
        agent.start_episode()
        agent.observe(Transition(3, 1, 10.0, 4, terminated=True, truncated=False))
        agent.end_episode()
        assert agent.q[3][1] == pytest.approx(10.0)

    def test_a_cut_off_episode_keeps_its_future(self) -> None:
        # The step limit stops the episode and the state it stopped in still
        # has a value, so the target is more than the reward alone.
        agent: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"), Discrete(2), n=1, step_size=1.0, discount=0.9
        )
        agent.values(4)[:] = [0.0, 20.0]
        agent.start_episode()
        agent.observe(Transition(3, 1, 10.0, 4, terminated=False, truncated=True))
        agent.end_episode()
        assert agent.q[3][1] > 10.0


class TestItLearns:
    @pytest.mark.parametrize("target", list(TARGETS))
    def test_it_reaches_the_optimal_cliff_walk_policy(self, target: str) -> None:
        rng = Rng(3)
        env = cliff_walk(rng.stream("env"))
        agent = TreeBackup(
            rng.stream("agent"),
            env.action_space,
            n=3,
            target=target,  # type: ignore[arg-type]
            step_size=0.2,
            epsilon=0.1,
        )
        train(env, agent, 500, discount=1.0)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=1.0)
        assert report.reaches_end
        assert report.start_value >= -20.0

    def test_it_keeps_no_row_for_a_cell_it_never_stands_in(self) -> None:
        rng = Rng(3)
        env = cliff_walk(rng.stream("env"))
        agent = TreeBackup(rng.stream("agent"), env.action_space, n=3)

        stood_in: set[int] = set()
        watched = agent.observe

        def observe(transition: Transition[int]) -> None:
            stood_in.add(transition.observation)
            watched(transition)

        agent.observe = observe  # type: ignore[method-assign]
        train(env, agent, 60, discount=1.0)
        assert set(agent.q) - stood_in == set()


class TestAgainstTheNeighbour:
    def test_it_is_not_n_step_sarsa(self) -> None:
        # They share the buffering and differ in the target, so a walk that
        # both are fed has to leave them holding different tables.
        sarsa: NStepSarsa[int] = NStepSarsa(
            Rng(1).stream("a"), Discrete(2), n=3, step_size=0.5, epsilon=0.1
        )
        tree: TreeBackup[int] = TreeBackup(
            Rng(1).stream("a"), Discrete(2), n=3, step_size=0.5, epsilon=0.1
        )
        feed(sarsa)
        feed(tree)
        assert tree.q != sarsa.q
