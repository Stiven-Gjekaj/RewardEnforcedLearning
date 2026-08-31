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
from rel.agents.tree import TARGETS, QSigma, TreeBackup
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

        def observe(transition: Transition[int, int]) -> None:
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


def a_sigma(sigma: float, target: str = "greedy", n: int = 3) -> QSigma[int]:
    return QSigma(
        Rng(1).stream("a"),
        Discrete(2),
        n=n,
        sigma=sigma,
        target=target,  # type: ignore[arg-type]
        step_size=0.5,
        discount=0.9,
        epsilon=0.1,
    )


def a_tree(target: str = "greedy", n: int = 3) -> TreeBackup[int]:
    return TreeBackup(
        Rng(1).stream("a"),
        Discrete(2),
        n=n,
        target=target,  # type: ignore[arg-type]
        step_size=0.5,
        discount=0.9,
        epsilon=0.1,
    )


def cells(agent) -> list[float]:  # type: ignore[no-untyped-def]
    return [value for state in range(5) for value in agent.peek(state)]


class TestSigmaOfNothingIsTreeBackup:
    """The end of the family that this project already had.

    A method that did not reach the agent it says it contains would be a
    different algorithm wearing the name, so this is checked cell for cell
    rather than approximately.
    """

    def test_a_greedy_target_agrees_bit_for_bit(self) -> None:
        # Exactly, because a greedy target has shares of one and nothing, so
        # nothing is rearranged between the two ways of writing the step.
        tree, sigma = a_tree("greedy"), a_sigma(0.0, "greedy")
        feed(tree)
        feed(sigma)
        assert cells(tree) == cells(sigma)

    def test_an_averaged_target_agrees_too(self) -> None:
        tree, sigma = a_tree("policy"), a_sigma(0.0, "policy")
        feed(tree)
        feed(sigma)
        assert cells(tree) == cells(sigma)

    @pytest.mark.parametrize("target", TARGETS)
    def test_it_holds_over_a_whole_run(self, target: str) -> None:
        """One episode of four steps could agree by accident.

        This is sixty episodes of the cliff walk, where the two agents also
        have to choose the same action every step of the way, and a
        difference in the last bits of a value can flip which action is best.
        """
        got = []
        shared = {
            "n": 3,
            "target": target,
            "step_size": 0.2,
            "discount": 1.0,
            "epsilon": 0.1,
        }
        for sigma in (None, 0.0):
            env = cliff_walk(Rng(1).stream("env"))
            agent = (
                TreeBackup(Rng(1).stream("a"), env.action_space, **shared)
                if sigma is None
                else QSigma(Rng(1).stream("a"), env.action_space, sigma=sigma, **shared)
            )
            train(env, agent, 60, discount=1.0)
            got.append([value for state in range(48) for value in agent.peek(state)])

        worst = max(abs(one - other) for one, other in zip(*got, strict=True))
        # Not always zero for an averaged target. Tree backup sums the actions
        # that were not taken and adds the taken one separately; this sums all
        # of them and subtracts the taken one back out. Float addition is not
        # associative, so the two can land a few bits apart and stay there.
        assert worst < 1e-12, target


class TestSigmaChangesTheAnswer:
    def test_the_two_ends_do_not_agree(self) -> None:
        # Without this the class above would pass on a sigma that was ignored.
        nothing, everything = a_sigma(0.0), a_sigma(1.0)
        feed(nothing)
        feed(everything)
        assert cells(nothing) != cells(everything)

    def test_the_middle_is_neither_end(self) -> None:
        middle = a_sigma(0.5)
        nothing, everything = a_sigma(0.0), a_sigma(1.0)
        for agent in (middle, nothing, everything):
            feed(agent)
        assert cells(middle) != cells(nothing)
        assert cells(middle) != cells(everything)

    def test_the_middle_lies_between_the_two_ends(self) -> None:
        """Which is what makes it an interpolation rather than a third thing.

        The coefficient is a straight line in sigma, so at a half it is the
        average of the two ends. That is a claim about one update, so it is
        checked on the first update of a fresh agent rather than after a run,
        where the ends would have moved apart.
        """
        got = {}
        for sigma in (0.0, 0.5, 1.0):
            agent = a_sigma(sigma)
            agent.start_episode()
            for step in WALK:
                agent.observe(step)
            got[sigma] = agent.peek(0)[0]

        assert got[0.5] == pytest.approx((got[0.0] + got[1.0]) / 2.0)


class TestTheCoefficient:
    def test_at_sigma_of_nothing_it_is_the_share(self) -> None:
        agent = a_sigma(0.0)
        assert agent._coefficient(0, 0, 0.25) == 0.25

    def test_at_sigma_of_everything_it_is_the_ratio(self) -> None:
        """The share divided by how often the behaviour policy takes it.

        With one action clearly ahead, an epsilon of 0.1 over two actions
        takes it 0.95 of the time, so a target share of one is a ratio of one
        over 0.95. On a fresh table the two actions are tied and the answer
        would be two, which is a fact about ties rather than about the ratio.
        """
        agent = a_sigma(1.0)
        agent.values(0)[1] = 5.0
        assert agent.greedy(0) == 1
        assert agent._coefficient(0, 1, 1.0) == pytest.approx(1.0 / 0.95)

    def test_it_is_a_straight_line_between_them(self) -> None:
        agents = [a_sigma(sigma) for sigma in (0.0, 0.5, 1.0)]
        for agent in agents:
            agent.values(0)[1] = 5.0
        low, middle, high = (agent._coefficient(0, 1, 1.0) for agent in agents)
        assert middle == pytest.approx((low + high) / 2.0)

    def test_an_action_the_behaviour_would_never_take_has_no_ratio(self) -> None:
        # A division by nothing, answered rather than raised. It cannot happen
        # on an action the behaviour policy really took.
        agent = a_sigma(1.0, "greedy")
        agent.explore.epsilon = lambda episodes: 0.0  # type: ignore[assignment]
        worst = 1 - agent.greedy(0)
        assert agent._coefficient(0, worst, 0.0) == 0.0


class TestSigmaCanChange:
    def test_a_schedule_is_read_on_every_step(self) -> None:
        # The book suggests starting at one and falling towards nothing.
        agent = a_sigma(0.0)
        agent.sigma = lambda steps: 1.0 if steps < 2 else 0.0  # type: ignore[assignment]
        agent.steps = 0
        assert agent.current_sigma() == 1.0
        agent.steps = 5
        assert agent.current_sigma() == 0.0

    def test_a_sigma_outside_nothing_to_one_is_refused(self) -> None:
        agent = a_sigma(0.0)
        agent.sigma = lambda steps: 1.5  # type: ignore[assignment]
        with pytest.raises(ValueError, match="share of a sample"):
            agent.current_sigma()


class TestItLearnsTheCliffWalk:
    @pytest.mark.parametrize("sigma", [0.0, 0.5, 1.0])
    def test_the_greedy_policy_reaches_the_goal(self, sigma: float) -> None:
        env = cliff_walk(Rng(1).stream("env"))
        agent = QSigma(
            Rng(1).stream("agent"),
            env.action_space,
            n=3,
            sigma=sigma,
            step_size=0.2,
            discount=1.0,
            epsilon=0.1,
        )
        train(env, agent, 400, discount=1.0)
        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=1.0)
        assert report.reaches_end, sigma
        assert report.start_value >= -20.0, sigma


class TestTheRegistryEntry:
    def test_it_is_registered_off_policy(self) -> None:
        from rel.agents import AGENTS

        assert set(AGENTS["q-sigma"].tags) == {"tabular", "off-policy"}

    def test_the_settings_can_be_reached_from_the_command_line(self) -> None:
        from rel.agents import AGENTS

        env = cliff_walk(Rng(1).stream("env"))
        agent = AGENTS.make(
            "q-sigma", Rng(1).stream("agent"), env, n=5, sigma=0.25, target="policy"
        )
        assert isinstance(agent, QSigma)
        assert (agent.n, agent.current_sigma(), agent.target) == (5, 0.25, "policy")

    def test_it_says_its_sigma(self) -> None:
        assert "sigma=0.5" in repr(a_sigma(0.5))
