"""Tests for solving an environment exactly.

These numbers are the reference that every other result in the project is
measured against, so they are checked three ways: against arithmetic done by
hand, against a second method that reaches the same answer differently, and
against the environment itself by running the policy that comes out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from rel.agents.dp import (
    DidNotSettleError,
    FixedPolicyAgent,
    average_reward,
    evaluate_policy,
    policy_iteration,
    reaches_end,
    value_iteration,
)
from rel.core import EnvSpec, Outcome, Step, TabularEnv
from rel.envs.classic import (
    cliff_walk,
    dyna_maze,
    four_rooms,
    frozen_lake,
    windy_grid,
)
from rel.envs.continuing import LONG, SHORT, two_loops
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import evaluate

# name, builder, discount, and how many episodes are needed to see the value.
GRIDS = [
    ("cliff", cliff_walk, 1.0),
    ("windy", windy_grid, 1.0),
    ("rooms", four_rooms, 1.0),
    ("maze", dyna_maze, 0.95),
    ("lake4", frozen_lake, 1.0),
]


class TestKnownAnswers:
    """Numbers that can be worked out without running anything.

    Each one is derived in the comment above it. A test that only compared two
    of this project's own functions would agree with itself while both were
    wrong.
    """

    def test_the_cliff_walk_is_worth_minus_thirteen(self) -> None:
        # One step up out of the start, eleven steps right along row two, one
        # step down onto the goal. Thirteen steps at -1 each.
        solved = value_iteration(cliff_walk(Rng(1)))
        assert solved.start_value == pytest.approx(-13.0)

    def test_the_windy_grid_is_worth_minus_fifteen(self) -> None:
        # Fifteen steps is the answer in Sutton and Barto, example 6.5.
        solved = value_iteration(windy_grid(Rng(1)))
        assert solved.start_value == pytest.approx(-15.0)

    def test_the_frozen_lake_is_worth_fourteen_seventeenths(self) -> None:
        # The published optimal value of the four by four slippery lake. Every
        # step pays nothing and the goal pays one, so this is the chance that
        # the best possible policy arrives at all.
        solved = value_iteration(frozen_lake(Rng(1)))
        assert solved.start_value == pytest.approx(14.0 / 17.0, abs=1e-6)

    def test_the_dyna_maze_is_the_discount_over_the_shortest_path(self) -> None:
        # Fourteen steps to the goal, nothing paid until the last one, and a
        # discount of 0.95. That is 0.95 to the thirteenth.
        solved = value_iteration(dyna_maze(Rng(1)), discount=0.95)
        assert solved.start_value == pytest.approx(0.95**13)

    def test_a_chain_of_three_matches_arithmetic(self) -> None:
        class Chain(TabularEnv):
            """Three states in a line. Right moves on, left stays."""

            def __init__(self, rng: Rng) -> None:
                super().__init__(rng)
                from rel.core import EnvSpec
                from rel.spaces import Discrete

                self.observation_space = Discrete(3)
                self.action_space = Discrete(2)
                self.spec = EnvSpec("chain", "Three in a line.")
                self.at = 0

            def _reset(self) -> int:
                self.at = 0
                return 0

            def _step(self, action: int) -> Step[int]:
                branch = self.transitions(self.at, action)[0]
                self.at = branch.observation
                return Step(branch.observation, branch.reward, branch.terminated, False)

            def transitions(self, state: int, action: int) -> Sequence[Outcome]:
                if state == 2:
                    return [Outcome(1.0, 2, 0.0, terminated=True)]
                if action == 1:
                    landed = state + 1
                    return [Outcome(1.0, landed, 1.0, landed == 2)]
                return [Outcome(1.0, state, -1.0, terminated=False)]

            def start_states(self) -> Sequence[tuple[float, int]]:
                return [(1.0, 0)]

        # Right twice: 1 + 0.5 * 1 = 1.5 from state 0, and 1 from state 1.
        solved = value_iteration(Chain(Rng(1)), discount=0.5)
        assert solved.values[0] == pytest.approx(1.5)
        assert solved.values[1] == pytest.approx(1.0)
        assert solved.values[2] == 0.0
        assert solved.policy[0] == 1


class TestTerminationInsideTheModel:
    """A branch marked as an ending pays its reward and nothing after it.

    In every grid here a terminal state also has a value of zero, so
    bootstrapping through one adds zero and the fault is invisible. This
    environment separates the two: entering state one from state zero ends the
    episode, and state one is an ordinary state worth five when it is reached
    any other way.

    The guard in `_backup` was written before this test and survived a run that
    removed it. That is the whole reason the test exists.
    """

    class Doorway(TabularEnv):
        def __init__(self, rng: Rng) -> None:
            super().__init__(rng)
            from rel.core import EnvSpec
            from rel.spaces import Discrete

            self.observation_space = Discrete(3)
            self.action_space = Discrete(2)
            self.spec = EnvSpec("doorway", "Entering state one can end it.")
            self.at = 0

        def _reset(self) -> int:
            self.at = 0
            return 0

        def _step(self, action: int) -> Step[int]:
            branch = self.transitions(self.at, action)[0]
            self.at = branch.observation
            return Step(branch.observation, branch.reward, branch.terminated, False)

        def transitions(self, state: int, action: int) -> Sequence[Outcome]:
            if state == 2:
                return [Outcome(1.0, 2, 0.0, terminated=True)]
            if state == 0:
                if action == 0:
                    # Ends the episode, standing in state one.
                    return [Outcome(1.0, 1, 0.0, terminated=True)]
                return [Outcome(1.0, 1, 0.0, terminated=False)]
            # State one is an ordinary state. Action zero leaves it and pays
            # five. Action one waits and pays nothing, which is what stops the
            # default rule from calling state one terminal.
            if action == 0:
                return [Outcome(1.0, 2, 5.0, terminated=True)]
            return [Outcome(1.0, 1, -1.0, terminated=False)]

        def start_states(self) -> Sequence[tuple[float, int]]:
            return [(1.0, 0)]

    def test_state_one_is_not_terminal(self) -> None:
        env = self.Doorway(Rng(1))
        assert env.terminal_states() == frozenset({2})

    def test_the_door_is_worth_nothing_and_the_long_way_is_worth_five(self) -> None:
        solved = value_iteration(self.Doorway(Rng(1)))
        assert solved.values[1] == pytest.approx(5.0)
        # Taking the door ends the episode at once, so it pays its own reward
        # of zero and not the five that standing in state one is worth.
        assert solved.policy[0] == 1
        assert solved.start_value == pytest.approx(5.0)


@pytest.mark.parametrize(("name", "build", "discount"), GRIDS)
class TestTheTwoMethodsAgree:
    def test_the_values_are_the_same(
        self, name: str, build: object, discount: float
    ) -> None:
        env = build(Rng(1))  # type: ignore[operator]
        first = value_iteration(env, discount=discount)
        second = policy_iteration(env, discount=discount)
        largest = max(
            abs(a - b) for a, b in zip(first.values, second.values, strict=True)
        )
        assert largest < 1e-6, f"{name}: the two methods differ by {largest:g}"

    def test_the_start_value_is_the_same(
        self, name: str, build: object, discount: float
    ) -> None:
        env = build(Rng(1))  # type: ignore[operator]
        first = value_iteration(env, discount=discount)
        second = policy_iteration(env, discount=discount)
        assert first.start_value == pytest.approx(second.start_value, abs=1e-6)


@pytest.mark.parametrize(("name", "build", "discount"), GRIDS)
class TestTheAnswerSurvivesContactWithTheEnvironment:
    """The last check, and the one that ties the model to the code.

    Dynamic programming reads the transitions an environment declares. Running
    the policy it produces asks the environment itself. If the two disagree,
    the model has drifted from the code, and every claim measured against the
    model is wrong by an amount nobody can see.
    """

    def test_running_the_best_policy_pays_what_it_promised(
        self, name: str, build: object, discount: float
    ) -> None:
        env = build(Rng(1))  # type: ignore[operator]
        solved = value_iteration(env, discount=discount)

        watched = build(Rng(2).stream("env"))  # type: ignore[operator]
        agent = FixedPolicyAgent(
            Rng(2).stream("agent"), env.action_space, solved.policy
        )
        record = evaluate(watched, agent, 3000, discount=discount)

        measured = sum(record.discounted) / len(record.discounted)
        # Four standard errors of the mean, plus room for the discounting.
        spread = (
            4.0
            * (
                sum((value - measured) ** 2 for value in record.discounted)
                / (len(record.discounted) - 1)
            )
            ** 0.5
            / len(record.discounted) ** 0.5
        )
        assert abs(measured - solved.start_value) < spread + 1e-9, (
            f"{name}: the model says {solved.start_value:.5f} and running it "
            f"gave {measured:.5f}"
        )

    def test_the_best_policy_always_finishes(
        self, name: str, build: object, discount: float
    ) -> None:
        env = build(Rng(1))  # type: ignore[operator]
        solved = value_iteration(env, discount=discount)
        assert reaches_end(env, solved.policy)


class TestProperPolicies:
    def test_a_policy_that_circles_is_reported(self) -> None:
        env = cliff_walk(Rng(1))
        circling = [0] * env.observation_space.n  # up everywhere
        assert not reaches_end(env, circling)

    def test_evaluating_a_circling_policy_says_so_rather_than_guessing(self) -> None:
        # The alternative is a very large negative number, which a reader takes
        # for a long path.
        env = cliff_walk(Rng(1))
        report = evaluate_policy(env, [0] * env.observation_space.n)
        assert not report.reaches_end
        assert report.start_value == -math.inf

    def test_evaluating_the_best_policy_gives_the_values_back(self) -> None:
        env = cliff_walk(Rng(1))
        solved = value_iteration(env)
        report = evaluate_policy(env, solved.policy)
        assert report.reaches_end
        assert report.start_value == pytest.approx(solved.start_value)

    def test_policy_iteration_starts_from_a_policy_that_finishes(self) -> None:
        # Policy iteration used to sit in its first evaluation for twenty
        # thousand sweeps, because the policy it began with walked into a wall
        # and stayed. The fix is a start built by a backward search.
        from rel.agents.dp import _proper_policy

        for _, build, _discount in GRIDS:
            env = build(Rng(1))  # type: ignore[operator]
            start = _proper_policy(env, env.terminal_states())
            assert reaches_end(env, start), env.spec.name


class TestFailingLoudly:
    def test_value_iteration_says_when_it_did_not_settle(self) -> None:
        with pytest.raises(DidNotSettleError, match="still moving"):
            value_iteration(frozen_lake(Rng(1)), max_sweeps=3)

    def test_policy_evaluation_says_when_it_did_not_settle(self) -> None:
        env = cliff_walk(Rng(1))
        solved = value_iteration(env)
        with pytest.raises(DidNotSettleError, match="still moving"):
            evaluate_policy(env, solved.policy, max_sweeps=1)


class TestFixedPolicyAgent:
    def test_it_plays_the_policy_it_was_given(self) -> None:
        env = cliff_walk(Rng(1))
        policy = value_iteration(env).policy
        agent = FixedPolicyAgent(Rng(1), env.action_space, policy)
        for state in range(env.observation_space.n):
            assert agent.act(state) == policy[state]
            assert agent.greedy(state) == policy[state]

    def test_it_never_explores(self) -> None:
        env = cliff_walk(Rng(1))
        agent = FixedPolicyAgent(Rng(1), env.action_space, [2] * 48)
        assert {agent.act(0) for _ in range(200)} == {2}


class TestTheAverageRewardOfAPolicy:
    """What a task with no ending is really scored by.

    A discounted value answers "what is the future worth from here, with later
    worth less", and on a task that never ends it has a different answer for
    every discount. This has one answer and no setting in it.
    """

    def test_each_loop_is_worth_what_it_pays_per_step(self) -> None:
        env = two_loops(Rng(1))
        states = env.observation_space.n
        assert average_reward(env, [SHORT] * states) == pytest.approx(1.0)
        assert average_reward(env, [LONG] * states) == pytest.approx(2.0)

    @pytest.mark.parametrize(("discount", "collected"), [(0.7, 1.0), (0.9, 2.0)])
    def test_it_scores_what_a_discount_chose(
        self, discount: float, collected: float
    ) -> None:
        """The finding of the two loops, said in the units that matter.

        At a discount of 0.7 the best policy under that discount collects half
        as much per step as the other one. Nothing went wrong: the discount
        was part of the question.
        """
        env = two_loops(Rng(1))
        chosen = list(value_iteration(env, discount=discount).policy)
        assert average_reward(env, chosen) == pytest.approx(collected)

    def test_the_steps_before_the_cycle_do_not_count(self) -> None:
        """A per step average over for ever does not see a finite prefix."""

        class LeadIn(TabularEnv):
            """A large payment once, then a cycle worth one a step."""

            def __init__(self, rng: Rng) -> None:
                super().__init__(rng)
                self.observation_space = Discrete(3)
                self.action_space = Discrete(1)
                self.spec = EnvSpec("leadin", "One payment, then a cycle.")
                self.at = 0

            def _reset(self) -> int:
                self.at = 0
                return 0

            def _step(self, action: int) -> Step[int]:
                branch = self.transitions(self.at, action)[0]
                self.at = branch.observation
                return Step(branch.observation, branch.reward, False, False)

            def transitions(self, state: int, action: int) -> Sequence[Outcome]:
                if state == 0:
                    return [Outcome(1.0, 1, 100.0, terminated=False)]
                if state == 1:
                    return [Outcome(1.0, 2, 0.0, terminated=False)]
                return [Outcome(1.0, 1, 2.0, terminated=False)]

            def start_states(self) -> Sequence[tuple[float, int]]:
                return [(1.0, 0)]

        assert average_reward(LeadIn(Rng(1)), [0, 0, 0]) == pytest.approx(1.0)

    def test_a_branching_step_has_no_answer_here(self) -> None:
        """The frozen lake slips, so a walk is not what it is worth.

        Averaging over a stationary distribution would answer it, and that
        needs a linear solve this project does not have. Saying `None` is
        better than walking one branch and calling it the answer.
        """
        env = frozen_lake(Rng(1))
        policy = [0] * env.observation_space.n
        assert average_reward(env, policy) is None
