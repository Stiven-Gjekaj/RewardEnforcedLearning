"""Tests for Dyna that replays the step that matters.

The strongest test here is the backwards sweep worked out by hand. Two steps
into a goal, a step size of one, and the cell two moves from the reward ends up
holding the discounted value of it after a single pass. Uniform replay reaches
that cell only by chance, and this reaches it because the change was followed.
"""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.dp import evaluate_policy, value_iteration
from rel.agents.dyna import DynaQ
from rel.agents.sweeping import CAP, PrioritisedSweeping
from rel.envs.classic import cliff_walk, dyna_maze
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

TWO = Discrete(2)


def go(state: int, action: int, reward: float, landed: int) -> Transition[int, int]:
    return Transition(state, action, reward, landed, False, False)


def stop(state: int, action: int, reward: float, landed: int) -> Transition[int, int]:
    return Transition(state, action, reward, landed, True, False)


def sweeper(**extra: float) -> PrioritisedSweeping[int]:
    settings: dict[str, float] = {
        "planning_steps": 5,
        "step_size": 1.0,
        "discount": 0.95,
        "epsilon": 0.0,
        **extra,
    }
    return PrioritisedSweeping(Rng(1), TWO, **settings)  # type: ignore[arg-type]


class TestThePredecessorTable:
    def test_a_step_records_what_leads_into_where_it_landed(self) -> None:
        agent = sweeper()
        agent.observe(go(0, 1, 0.0, 4))
        assert agent.leading_to[4] == {(0, 1)}

    def test_two_ways_into_one_cell_are_both_recorded(self) -> None:
        agent = sweeper()
        agent.observe(go(0, 1, 0.0, 4))
        agent.observe(go(3, 0, 0.0, 4))
        assert agent.leading_to[4] == {(0, 1), (3, 0)}

    def test_a_terminated_step_records_nothing(self) -> None:
        # There is no sweeping back from an ending. The state the episode
        # stopped in is never one whose value moves, so a predecessor of it
        # would never be looked up, and holding one would be a key for a state
        # the agent is never in.
        agent = sweeper()
        agent.observe(stop(0, 1, 1.0, 9))
        assert 9 not in agent.leading_to


class TestTheQueue:
    def test_the_change_is_how_far_a_replay_would_move_the_value(self) -> None:
        agent = sweeper()
        agent.observe(stop(0, 1, 3.0, 9))
        agent.q[0][1] = 0.5
        assert agent.change_from(0, 1) == pytest.approx(2.5)

    def test_a_step_the_model_has_never_seen_would_change_nothing(self) -> None:
        agent = sweeper()
        assert agent.change_from(7, 0) == 0.0

    def test_a_change_below_the_threshold_is_not_worth_a_slot(self) -> None:
        agent = sweeper(threshold=0.1)
        agent.push(0, 1, 0.05)
        assert agent.queue == {}

    def test_the_larger_of_two_claims_on_one_step_wins(self) -> None:
        agent = sweeper(threshold=0.0)
        agent.push(0, 1, 0.4)
        agent.push(0, 1, 0.9)
        agent.push(0, 1, 0.2)
        assert agent.queue[(0, 1)] == pytest.approx(0.9)

    def test_the_largest_change_comes_off_first(self) -> None:
        agent = sweeper(threshold=0.0)
        agent.push(0, 0, 0.2)
        agent.push(0, 1, 0.9)
        agent.push(1, 0, 0.5)
        assert agent.pop() == (0, 1)
        assert agent.pop() == (1, 0)
        assert agent.pop() == (0, 0)

    def test_an_empty_queue_has_nothing_to_offer(self) -> None:
        assert sweeper().pop() is None

    def test_a_full_queue_drops_the_smallest_change(self) -> None:
        # The bound is on the shipped constant rather than a patched one, so
        # a change to it that broke this would be seen here.
        agent = sweeper(threshold=0.0)
        for index in range(CAP):
            agent.push(index, 0, 1.0 + index)
        assert len(agent.queue) == CAP

        agent.push(-1, 0, 500.0)
        assert len(agent.queue) == CAP
        # The new entry is worth more than the smallest, so the smallest went.
        assert (-1, 0) in agent.queue
        assert (0, 0) not in agent.queue

    def test_a_negative_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="below zero"):
            sweeper(threshold=-1.0)


class TestEveryUpdateComesOffTheQueue:
    def test_no_planning_steps_is_refused(self) -> None:
        # `dyna-q` with no planning is plain Q-learning. This with no planning
        # is an agent that learns nothing, which is a trap rather than a
        # setting.
        with pytest.raises(ValueError, match="at least one planning step"):
            sweeper(planning_steps=0)

    def test_the_replays_are_the_only_thing_that_touches_the_table(self) -> None:
        # The real step is pushed rather than applied, so a step whose change
        # is under the threshold moves nothing at all.
        agent = sweeper(threshold=10.0)
        agent.observe(stop(0, 1, 3.0, 9))
        assert agent.replays == 0
        assert agent.q == {}

    def test_it_does_not_replay_more_than_the_queue_holds(self) -> None:
        agent = sweeper(planning_steps=50)
        agent.observe(stop(0, 1, 3.0, 9))
        # One real step, and nothing leads into the cell it came from, so the
        # quota of fifty buys exactly one replay.
        assert agent.replays == 1


class TestWorkFollowsTheChangeBackwards:
    def test_one_pass_teaches_the_cell_before_the_one_that_moved(self) -> None:
        # Two steps to a goal that pays one. The step size is one, so a cell
        # ends up holding its target exactly.
        #
        #   the goal step:   1.0
        #   the cell before: 0 + 0.95 * 1.0 = 0.95
        #
        # The second of those is the sweep. The change at state one was found,
        # and the table was asked what leads into state one.
        agent = sweeper(planning_steps=2)
        agent.observe(go(0, 0, 0.0, 1))
        agent.observe(stop(1, 0, 1.0, 2))

        assert agent.q[1][0] == pytest.approx(1.0)
        assert agent.q[0][0] == pytest.approx(0.95)

    def test_the_sweep_stops_where_the_quota_does(self) -> None:
        # The same walk with one planning step. The goal step is replayed and
        # the cell before it is queued, and there the pass ends.
        agent = sweeper(planning_steps=1)
        agent.observe(go(0, 0, 0.0, 1))
        agent.observe(stop(1, 0, 1.0, 2))

        assert agent.q[1][0] == pytest.approx(1.0)
        assert 0 not in agent.q
        assert agent.queue[(0, 0)] == pytest.approx(0.95)


class TestItLearns:
    def test_it_reaches_the_optimal_cliff_walk_policy(self) -> None:
        rng = Rng(3)
        env = cliff_walk(rng.stream("env"))
        agent = PrioritisedSweeping(
            rng.stream("agent"),
            env.action_space,
            planning_steps=5,
            step_size=0.5,
            discount=1.0,
            epsilon=0.1,
        )
        train(env, agent, 200, discount=1.0)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=1.0)
        assert report.reaches_end
        assert report.start_value >= -20.0

    def test_it_keeps_no_row_for_a_cell_it_never_stands_in(self) -> None:
        rng = Rng(3)
        env = dyna_maze(rng.stream("env"))
        agent = PrioritisedSweeping(
            rng.stream("agent"), env.action_space, planning_steps=5, step_size=0.5
        )

        stood_in: set[int] = set()
        watched = agent.observe

        def observe(transition: Transition[int, int]) -> None:
            stood_in.add(transition.observation)
            watched(transition)

        agent.observe = observe  # type: ignore[method-assign]
        train(env, agent, 20)
        assert set(agent.q) - stood_in == set()


class TestAgainstUniformReplay:
    """The claim the method exists for, counted in updates rather than episodes.

    An update is one application of the learning rule to one cell. `dyna-q`
    makes one for the real step and its full quota after it, every step. This
    makes one for every entry it takes off the queue, and none when the queue
    is empty.
    """

    @staticmethod
    def updates_to_solve(builder: type[DynaQ[int]], seed: int) -> int:
        planning = 5
        rng = Rng(seed)
        env = dyna_maze(rng.stream("env"))
        best = value_iteration(env, discount=0.95).start_value
        agent = builder(
            rng.stream("agent"),
            env.action_space,
            planning_steps=planning,
            step_size=0.5,
            discount=0.95,
            epsilon=0.1,
        )

        for _ in range(60):
            train(env, agent, 1)

            # `greedy` breaks a tie by drawing, and an untouched row is all
            # ties, so asking for the policy after every episode would change
            # the run being measured. The draws are put back.
            before = agent.rng.snapshot()
            policy = [agent.greedy(state) for state in range(env.observation_space.n)]
            agent.rng = Rng.restore(*before)

            report = evaluate_policy(env, policy, discount=0.95)
            if report.reaches_end and report.start_value >= best - 1e-9:
                if isinstance(agent, PrioritisedSweeping):
                    return agent.replays
                return agent.steps * (1 + planning)

        raise AssertionError(f"{builder.__name__} did not solve seed {seed}.")

    @pytest.mark.parametrize("seed", [3, 4, 8])
    def test_it_needs_fewer_updates_than_dyna_q_on_the_maze(self, seed: int) -> None:
        # Measured over ten seeds by `scripts/measure_sweeping.py`: a median of
        # 668 updates against 7626. The margin asserted here is three, which is
        # well inside the narrowest of the three seeds below.
        uniform = self.updates_to_solve(DynaQ, seed)
        ordered = self.updates_to_solve(PrioritisedSweeping, seed)
        assert ordered * 3 < uniform


class TestItStopsWhenThereIsNothingLeft:
    """The threshold is why the work ends, and why it can end too early.

    Once every step in the model would change by less than the threshold, the
    queue empties and stays empty. `dyna-q` in the same position keeps making
    its full quota of replays forever, all of them teaching nothing.

    The same mechanism is the weakness. The agent stops because it believes it
    is finished, and it can believe that while holding a policy that is not the
    best one.
    """

    @staticmethod
    def settled(seed: int) -> PrioritisedSweeping[int]:
        rng = Rng(seed)
        env = dyna_maze(rng.stream("env"))
        agent: PrioritisedSweeping[int] = PrioritisedSweeping(
            rng.stream("agent"),
            env.action_space,
            planning_steps=5,
            step_size=0.5,
            discount=0.95,
            epsilon=0.1,
        )
        train(env, agent, 200)
        return agent

    def test_a_settled_run_costs_almost_nothing(self) -> None:
        rng = Rng(1)
        env = dyna_maze(rng.stream("env"))
        agent = self.settled(1)
        assert agent.queue == {}

        before_replays, before_steps = agent.replays, agent.steps
        train(env, agent, 20)
        steps = agent.steps - before_steps

        # Dyna-Q would have made six updates for each of these steps.
        assert steps > 200
        assert agent.replays - before_replays <= 1

    def test_it_can_stop_while_holding_a_policy_that_is_not_the_best(self) -> None:
        # Seed 9 settles on a route two steps longer than the shortest one and
        # never moves off it. Nothing in the model changes any more, so nothing
        # is queued, and the only way back to the question is a real step that
        # surprises it. That is what the bonus of Dyna-Q+ is for, and this
        # agent does not have one.
        rng = Rng(9)
        env = dyna_maze(rng.stream("env"))
        best = value_iteration(env, discount=0.95).start_value
        agent = self.settled(9)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=0.95)
        assert report.reaches_end
        assert report.start_value < best

        # And it stays there. Two hundred more episodes change nothing.
        train(env, agent, 200)
        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        assert evaluate_policy(env, policy, discount=0.95).start_value < best
