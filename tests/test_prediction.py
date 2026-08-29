"""Tests for estimating what a fixed policy is worth.

The random walk is the anchor. Its values are arithmetic rather than another
computation, so a prediction can be scored against the answer, and the tests
that matter here either work an update out by hand or hold one method against
another it has to equal exactly.
"""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.prediction import (
    MonteCarloPrediction,
    NStepTD,
    Predictor,
    TDLambda,
    TemporalDifference,
)
from rel.envs.classic import random_walk
from rel.rng import Rng
from rel.schedules import exponential
from rel.spaces import Discrete
from rel.training import train

TWO = Discrete(2)

#: A step size that starts high enough to move and ends low enough to settle.
#: A schedule is a function, and a function kept on a class becomes a method
#: of it, so this lives here rather than beside the test that uses it.
SETTLING = exponential(0.2, 400, 0.005)

WALK = (
    Transition(0, 0, -1.0, 1, terminated=False, truncated=False),
    Transition(1, 1, -1.0, 2, terminated=False, truncated=False),
    Transition(2, 0, 10.0, 3, terminated=True, truncated=False),
)


def feed(agent: Predictor, steps: tuple[Transition[int], ...] = WALK) -> Predictor:
    agent.start_episode()
    for step in steps:
        agent.observe(step)
    return agent


class TestFollowingThePolicyItWasGiven:
    def test_it_takes_the_action_the_policy_names(self) -> None:
        agent = Predictor(Rng(1), TWO, {0: 1, 1: 0})
        assert agent.act(0) == 1
        assert agent.act(1) == 0

    def test_greedy_and_act_are_the_same(self) -> None:
        # A predictor is not choosing, so there is no exploration to leave out.
        agent = Predictor(Rng(1), TWO, {0: 1})
        assert agent.greedy(0) == agent.act(0)

    def test_no_policy_draws_uniformly(self) -> None:
        agent = Predictor(Rng(1), TWO)
        drawn = {agent.act(0) for _ in range(50)}
        assert drawn == {0, 1}

    def test_a_policy_silent_about_a_state_is_refused(self) -> None:
        # Quietly taking action zero would be a different policy, and the
        # value it estimated would be the value of that one.
        agent = Predictor(Rng(1), TWO, {0: 1})
        with pytest.raises(ValueError, match="says nothing about state 4"):
            agent.act(4)

    def test_a_discount_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            Predictor(Rng(1), TWO, discount=1.5)


class TestTheTable:
    def test_an_unseen_state_is_worth_what_it_started_at(self) -> None:
        assert Predictor(Rng(1), TWO, start_value=0.5).value(9) == 0.5

    def test_reading_a_state_does_not_add_it(self) -> None:
        agent = Predictor(Rng(1), TWO)
        agent.value(9)
        assert not agent.knows(9)

    def test_it_keeps_no_value_for_an_action(self) -> None:
        assert Predictor(Rng(1), TWO).action_values(0) is None

    def test_the_value_of_a_state_is_the_number_it_keeps(self) -> None:
        agent = Predictor(Rng(1), TWO)
        agent.v[3] = 0.75
        assert agent.state_value(3) == 0.75

    def test_what_it_learned_is_the_table(self) -> None:
        agent = Predictor(Rng(1), TWO)
        agent.v[2] = 0.5
        agent.v[1] = 0.25
        assert list(agent.learned()) == ["1|0.25", "2|0.5"]


class TestScoringAgainstTheTruth:
    def test_a_perfect_table_scores_zero(self) -> None:
        agent = Predictor(Rng(1), TWO)
        agent.v.update({1: 0.5, 2: 1.0})
        assert agent.error_against({1: 0.5, 2: 1.0}) == 0.0

    def test_it_is_the_root_mean_square(self) -> None:
        agent = Predictor(Rng(1), TWO)
        agent.v.update({1: 0.0, 2: 0.0})
        # Errors of 3 and 4, so the mean square is 12.5 and the root is 3.5355.
        assert agent.error_against({1: 3.0, 2: 4.0}) == pytest.approx(12.5**0.5)

    def test_a_state_never_seen_counts_at_what_it_started_at(self) -> None:
        # Leaving it out would report an agent that has seen one state as
        # perfect.
        agent = Predictor(Rng(1), TWO, start_value=0.0)
        assert agent.error_against({1: 1.0}) == 1.0

    def test_scoring_nothing_is_zero(self) -> None:
        assert Predictor(Rng(1), TWO).error_against({}) == 0.0

    def test_the_walk_leaves_its_endings_out(self) -> None:
        # An agent is never in one, so its entry there never moves off the
        # starting value and scoring it would measure the starting value.
        env = random_walk(Rng(1))
        assert set(env.values_to_score()) == {1, 2, 3, 4, 5}


class TestTemporalDifference:
    def test_one_step_is_the_reward_plus_what_it_believes_next(self) -> None:
        #   -1 + 0.9 * 10 = 8, and half of that from zero is 4.
        agent = TemporalDifference(Rng(1), TWO, step_size=0.5, discount=0.9)
        agent.v[1] = 10.0
        agent.observe(WALK[0])
        assert agent.v[0] == pytest.approx(4.0)

    def test_a_terminated_episode_bootstraps_from_nothing(self) -> None:
        agent = TemporalDifference(Rng(1), TWO, step_size=1.0, discount=0.9)
        agent.v[3] = 100.0
        agent.observe(WALK[2])
        assert agent.v[2] == pytest.approx(10.0)

    def test_a_cut_off_episode_keeps_its_future(self) -> None:
        agent = TemporalDifference(Rng(1), TWO, step_size=1.0, discount=0.9)
        agent.v[3] = 100.0
        agent.observe(Transition(2, 0, 10.0, 3, terminated=False, truncated=True))
        assert agent.v[2] == pytest.approx(100.0)


class TestNStepTD:
    def test_one_step_is_td_zero_cell_for_cell(self) -> None:
        one = NStepTD(Rng(1), TWO, n=1, step_size=0.5, discount=0.9)
        other = TemporalDifference(Rng(1), TWO, step_size=0.5, discount=0.9)
        assert feed(one).v == feed(other).v

    def test_a_two_step_target_is_what_it_should_be(self) -> None:
        # Two steps of -1, then the value of where the window stopped.
        #
        #   -1 + 0.5 * -1 + 0.25 * 8 = 0.5
        agent = NStepTD(Rng(1), TWO, n=2, step_size=1.0, discount=0.5)
        agent.v[2] = 8.0
        agent.start_episode()
        agent.observe(WALK[0])
        agent.observe(WALK[1])
        assert agent.v[0] == pytest.approx(0.5)

    def test_an_episode_shorter_than_n_still_credits_its_states(self) -> None:
        # The window never fills at all. Working out which states are still
        # owed an update from the length says none of them are, which is why
        # they are counted instead.
        agent = NStepTD(Rng(1), TWO, n=3, step_size=1.0, discount=1.0)
        agent.start_episode()
        agent.observe(Transition(0, 0, 7.0, 1, terminated=True, truncated=False))
        assert agent.v == {0: 7.0}

    def test_the_end_of_an_episode_credits_everything_left(self) -> None:
        agent = NStepTD(Rng(1), TWO, n=2, step_size=1.0, discount=1.0)
        feed(agent)
        assert set(agent.v) == {0, 1, 2}

    def test_a_cut_off_episode_keeps_its_future(self) -> None:
        agent = NStepTD(Rng(1), TWO, n=2, step_size=1.0, discount=1.0)
        agent.v[3] = 20.0
        agent.start_episode()
        agent.observe(Transition(2, 0, 10.0, 3, terminated=False, truncated=True))
        assert agent.v[2] == pytest.approx(30.0)

    def test_a_step_count_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            NStepTD(Rng(1), TWO, n=0)


class TestMonteCarloPrediction:
    def test_a_state_is_credited_with_the_rest_of_the_episode(self) -> None:
        #   -1 + 0.5 * -1 + 0.25 * 10 = 1.0
        agent = MonteCarloPrediction(Rng(1), TWO, step_size=1.0, discount=0.5)
        feed(agent)
        assert agent.v[0] == pytest.approx(1.0)

    def test_first_and_every_visit_differ_on_a_state_seen_twice(self) -> None:
        there_and_back = (
            Transition(0, 0, 1.0, 1, terminated=False, truncated=False),
            Transition(1, 0, 1.0, 0, terminated=False, truncated=False),
            Transition(0, 1, 1.0, 2, terminated=True, truncated=False),
        )
        first = MonteCarloPrediction(Rng(1), TWO, step_size=1.0, discount=1.0)
        every = MonteCarloPrediction(
            Rng(1), TWO, first_visit=False, step_size=1.0, discount=1.0
        )
        assert feed(first, there_and_back).v[0] != feed(every, there_and_back).v[0]

    def test_a_cut_off_episode_keeps_its_tail(self) -> None:
        # The step limit is not an ending. Treating it as one would teach the
        # agent that a long episode ends somewhere worthless.
        agent = MonteCarloPrediction(Rng(1), TWO, step_size=1.0, discount=1.0)
        agent.v[3] = 20.0
        agent.start_episode()
        agent.observe(Transition(2, 0, 10.0, 3, terminated=False, truncated=True))
        assert agent.v[2] == pytest.approx(30.0)

    def test_nothing_is_credited_until_the_episode_ends(self) -> None:
        agent = MonteCarloPrediction(Rng(1), TWO, step_size=1.0)
        agent.start_episode()
        agent.observe(WALK[0])
        assert agent.v == {}


class TestTDLambda:
    def test_a_decay_of_zero_is_td_zero_cell_for_cell(self) -> None:
        one = TDLambda(Rng(1), TWO, trace_decay=0.0, step_size=0.5, discount=0.9)
        other = TemporalDifference(Rng(1), TWO, step_size=0.5, discount=0.9)
        assert feed(one).v == feed(other).v

    def test_a_trace_does_not_reach_across_an_ending(self) -> None:
        agent = TDLambda(Rng(1), TWO, trace_decay=1.0, step_size=0.1)
        feed(agent)
        assert agent.e == {}

    def test_it_reaches_back_further_than_one_step(self) -> None:
        # The first state of the walk is two steps from the reward, so one
        # step of anything cannot have moved it after a single episode.
        short = TDLambda(Rng(1), TWO, trace_decay=0.0, step_size=0.5, discount=1.0)
        long = TDLambda(Rng(1), TWO, trace_decay=1.0, step_size=0.5, discount=1.0)
        assert feed(long).v[0] != feed(short).v[0]

    def test_the_three_kinds_differ_on_a_state_visited_twice(self) -> None:
        there_and_back = (
            Transition(0, 0, 1.0, 1, terminated=False, truncated=False),
            Transition(1, 0, 1.0, 0, terminated=False, truncated=False),
            Transition(0, 1, 1.0, 2, terminated=True, truncated=False),
        )
        seen = {
            kind: feed(
                TDLambda(
                    Rng(1),
                    TWO,
                    trace_decay=0.9,
                    traces=kind,  # type: ignore[arg-type]
                    step_size=0.2,
                ),
                there_and_back,
            ).v[0]
            for kind in ("accumulating", "replacing", "dutch")
        }
        assert len(set(seen.values())) == 3

    def test_a_decay_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            TDLambda(Rng(1), TWO, trace_decay=1.5)

    def test_a_kind_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ValueError, match="traces is one of"):
            TDLambda(Rng(1), TWO, traces="sticky")  # type: ignore[arg-type]


class TestTheyLearnTheWalk:
    """Every method gets close to the answer, with a step size that decays.

    A constant step size never converges. It tracks, and what it settles into
    is a band around the answer whose width is proportional to the step size,
    so a bound tight enough to mean something would be a bound on the step size
    rather than on the method. Decaying it is what turns tracking into
    converging.
    """

    @pytest.mark.parametrize(
        "builder",
        [
            lambda rng, env, step: TemporalDifference(
                rng, env.action_space, step_size=step, start_value=0.5
            ),
            lambda rng, env, step: NStepTD(
                rng, env.action_space, n=3, step_size=step, start_value=0.5
            ),
            lambda rng, env, step: TDLambda(
                rng, env.action_space, trace_decay=0.8, step_size=step, start_value=0.5
            ),
            lambda rng, env, step: MonteCarloPrediction(
                rng, env.action_space, step_size=step, start_value=0.5
            ),
        ],
        ids=["td", "n-step", "lambda", "monte-carlo"],
    )
    def test_it_gets_close_to_the_answer(self, builder) -> None:  # type: ignore[no-untyped-def]
        rng = Rng(3)
        env = random_walk(rng.stream("env"))
        agent = builder(rng.stream("agent"), env, SETTLING)
        train(env, agent, 2000)
        assert agent.error_against(env.values_to_score()) < 0.05

    def test_the_bound_is_one_an_untrained_table_fails(self) -> None:
        # Otherwise it would be a bound on the starting value rather than on
        # the learning. A table that is 0.5 everywhere scores 0.236.
        env = random_walk(Rng(3))
        agent = Predictor(Rng(3), env.action_space, start_value=0.5)
        assert agent.error_against(env.values_to_score()) == pytest.approx(
            0.2357, abs=1e-3
        )


class TestTDBeatsMonteCarloOnTheWalk:
    """The finding the random walk is here to make.

    Sutton and Barto, example 6.2. Both methods are estimating the same thing
    from the same episodes, and the one that uses what it already believes
    about where it ended up gets closer than the one that waits to find out.

    Twenty runs rather than fifty, and one step size rather than a ladder, so
    that this stays a test. `scripts/measure_prediction.py` is the measurement,
    and it says the same thing across the whole ladder.
    """

    STEP = 0.05
    RUNS = 20
    EPISODES = 100

    def _error(self, builder) -> float:  # type: ignore[no-untyped-def]
        errors = []
        for seed in range(1, self.RUNS + 1):
            root = Rng(seed)
            env = random_walk(root.stream("env"))
            agent = builder(root.stream("agent"), env)
            train(env, agent, self.EPISODES)
            errors.append(agent.error_against(env.values_to_score()))
        return sum(errors) / len(errors)

    def test_it_gets_closer_than_monte_carlo(self) -> None:
        td = self._error(
            lambda rng, env: TemporalDifference(
                rng, env.action_space, step_size=self.STEP, start_value=0.5
            )
        )
        monte_carlo = self._error(
            lambda rng, env: MonteCarloPrediction(
                rng, env.action_space, step_size=self.STEP, start_value=0.5
            )
        )
        assert td < monte_carlo

    def test_both_are_better_than_not_learning(self) -> None:
        # The comparison above is only worth making if both methods are
        # learning something. An untrained table scores 0.236.
        for builder in (
            lambda rng, env: TemporalDifference(
                rng, env.action_space, step_size=self.STEP, start_value=0.5
            ),
            lambda rng, env: MonteCarloPrediction(
                rng, env.action_space, step_size=self.STEP, start_value=0.5
            ),
        ):
            assert self._error(builder) < 0.15

    def test_every_visit_is_the_worst_of_them_at_this_step_size(self) -> None:
        # It credits a state once for every time the walk passed through it,
        # and the random walk doubles back constantly, so one episode can move
        # a cell several times in the same direction.
        every = self._error(
            lambda rng, env: MonteCarloPrediction(
                rng,
                env.action_space,
                first_visit=False,
                step_size=self.STEP,
                start_value=0.5,
            )
        )
        first = self._error(
            lambda rng, env: MonteCarloPrediction(
                rng, env.action_space, step_size=self.STEP, start_value=0.5
            )
        )
        assert every > first
