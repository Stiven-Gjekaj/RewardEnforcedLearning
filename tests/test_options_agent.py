"""Tests for Q-learning whose choices last several steps.

The two that carry the most are the collapse and the return. An agent holding
only primitive options has to be Q-learning cell for cell, because an option
that stops after one step is an action. And the return collected while a long
option runs has to be the discounted sum of what it collected, bootstrapped by
the discount raised to how long it took, which is worked out by hand here.
"""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.dp import evaluate_policy, value_iteration
from rel.agents.options import OptionsQ
from rel.agents.td import QLearning
from rel.envs.classic import four_rooms
from rel.options import Option, hallway_options, primitives
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

FOUR = Discrete(4)
STATES = range(4)

# Walk right along 0, 1, 2 and stop on arriving at 3.
CORRIDOR = Option(name="to the end", policy={0: 1, 1: 1, 2: 1}, stops=frozenset({3}))


def go(
    state: int, action: int, reward: float, landed: int, **flags: bool
) -> Transition[int]:
    return Transition(
        state,
        action,
        reward,
        landed,
        flags.get("terminated", False),
        flags.get("truncated", False),
    )


def an_agent(*extra: Option, **settings: float) -> OptionsQ:
    options = [*primitives(FOUR, STATES), *extra]
    return OptionsQ(
        Rng(1),
        FOUR,
        options,
        step_size=settings.get("step_size", 1.0),
        discount=settings.get("discount", 0.9),
        epsilon=settings.get("epsilon", 0.0),
    )


class TestAnOptionThatStopsAfterOneStepIsQLearning:
    def test_the_update_is_the_same_number(self) -> None:
        agent = an_agent(step_size=0.5, discount=0.9)
        agent.values(1)[2] = 10.0

        plain: QLearning[int] = QLearning(Rng(1), FOUR, step_size=0.5, discount=0.9)
        plain.values(1)[2] = 10.0

        # The option chosen is the one for action 1, made strictly best here.
        agent.values(0)[1] = 0.0
        agent.values(0)[0] = -1.0
        agent.values(0)[2] = -1.0
        agent.values(0)[3] = -1.0

        assert agent.act(0) == 1
        agent.observe(go(0, 1, -1.0, 1))
        plain.observe(go(0, 1, -1.0, 1))

        #   -1 + 0.9 * 10 = 8, and half of that from zero is 4.
        assert agent.q[0][1] == pytest.approx(4.0)
        assert agent.q[0][1] == pytest.approx(plain.q[0][1])

    def test_it_hands_control_back_after_every_step(self) -> None:
        agent = an_agent()
        agent.act(0)
        agent.observe(go(0, 1, 0.0, 1))
        assert agent.running is None


class TestTheReturnCollectedWhileAnOptionRuns:
    def test_it_is_discounted_and_bootstrapped_by_how_long_it_took(self) -> None:
        # Three steps at a discount of a half and a step size of one.
        #
        #   collected   1 + 0.5 * 2 + 0.25 * 4     = 3
        #   bootstrap   0.5 ** 3 * 8               = 1
        #   target                                 = 4
        agent = an_agent(CORRIDOR, discount=0.5, step_size=1.0)
        agent.values(3)[0] = 8.0
        agent.values(0)[4] = 1.0

        assert agent.act(0) == 1
        agent.observe(go(0, 1, 1.0, 1))
        agent.act(1)
        agent.observe(go(1, 1, 2.0, 2))
        agent.act(2)
        agent.observe(go(2, 1, 4.0, 3))

        assert agent.running is None
        assert agent.q[0][4] == pytest.approx(4.0)
        assert agent.length == 3

    def test_it_credits_the_state_the_option_started_in(self) -> None:
        # Not the state it stopped in, and not the ones in between. One
        # decision was made and it was made at the start, so exactly one row
        # exists afterwards. Reading the value of where it stopped adds no row
        # of its own, which is the rule the whole table is read under.
        agent = an_agent(CORRIDOR, discount=0.5)
        agent.values(0)[4] = 1.0
        for state, landed in ((0, 1), (1, 2), (2, 3)):
            agent.act(state)
            agent.observe(go(state, 1, 1.0, landed))

        assert set(agent.q) == {0}

    def test_a_terminated_episode_bootstraps_from_nothing(self) -> None:
        agent = an_agent(CORRIDOR, discount=0.5, step_size=1.0)
        agent.values(3)[0] = 8.0
        agent.values(0)[4] = 1.0

        agent.act(0)
        agent.observe(go(0, 1, 1.0, 1))
        agent.act(1)
        agent.observe(go(1, 1, 2.0, 3, terminated=True))

        # 1 + 0.5 * 2, and the 8 at state 3 is dropped.
        assert agent.q[0][4] == pytest.approx(2.0)

    def test_a_cut_off_episode_stops_the_option_and_keeps_its_future(self) -> None:
        # The step limit is not an ending. The option is stopped where it
        # stands, and the state it stopped in still has a value.
        agent = an_agent(CORRIDOR, discount=0.5, step_size=1.0)
        agent.values(2)[0] = 8.0
        agent.values(0)[4] = 1.0

        agent.act(0)
        agent.observe(go(0, 1, 1.0, 1))
        agent.act(1)
        agent.observe(go(1, 1, 2.0, 2, truncated=True))

        assert agent.running is None
        # 1 + 0.5 * 2 + 0.25 * 8 = 4.
        assert agent.q[0][4] == pytest.approx(4.0)


class TestRunningAnOption:
    def test_exploring_does_not_interrupt_one(self) -> None:
        # A decision that lasts several steps is the whole idea. An agent that
        # re-drew every step would have options in name only.
        agent = an_agent(CORRIDOR, epsilon=1.0)
        agent.values(0)[4] = 1.0
        while agent.running != 4:
            agent = an_agent(CORRIDOR, epsilon=1.0)
            agent.values(0)[4] = 1.0
            agent.act(0)

        agent.observe(go(0, 1, 0.0, 1))
        assert agent.running == 4
        assert agent.act(1) == CORRIDOR.act(1)

    def test_walking_out_of_its_own_states_stops_it(self) -> None:
        agent = an_agent(CORRIDOR)
        agent.values(0)[4] = 1.0
        agent.act(0)
        agent.observe(go(0, 1, 0.0, 9))
        assert agent.running is None

    def test_a_new_episode_drops_the_option_it_was_running(self) -> None:
        agent = an_agent(CORRIDOR)
        agent.values(0)[4] = 1.0
        agent.act(0)
        agent.observe(go(0, 1, 0.0, 1))
        agent.start_episode()
        assert agent.running is None

    def test_it_counts_what_it_ran(self) -> None:
        agent = an_agent(CORRIDOR, discount=0.5)
        agent.values(0)[4] = 1.0
        for state, landed in ((0, 1), (1, 2), (2, 3)):
            agent.act(state)
            agent.observe(go(state, 1, 1.0, landed))

        assert agent.finished == 1
        assert agent.steps_in_options == 3


class TestOnlyTheOptionsThatCanStartHereCount:
    def test_the_best_value_ignores_an_option_that_cannot_start(self) -> None:
        # The fault this prevents: an untouched entry for an option belonging
        # to another room is zero, and on a grid where every step costs
        # something zero is the largest number in the table.
        agent = an_agent(CORRIDOR)
        agent.values(3)[0] = -5.0
        agent.values(3)[1] = -6.0
        agent.values(3)[2] = -7.0
        agent.values(3)[3] = -8.0
        assert agent.best_value(3) == pytest.approx(-5.0)

    def test_a_state_no_option_covers_is_worth_nothing(self) -> None:
        agent = an_agent()
        assert agent.best_value(99) == 0.0

    def test_the_value_map_reads_the_same_options(self) -> None:
        agent = an_agent(CORRIDOR)
        assert len(agent.action_values(0) or []) == 5
        assert len(agent.action_values(3) or []) == 4

    def test_an_agent_with_no_options_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one option"):
            OptionsQ(Rng(1), FOUR, [])

    def test_acting_where_nothing_can_start_is_refused(self) -> None:
        agent = an_agent()
        with pytest.raises(ValueError, match="can start in state 99"):
            agent.act(99)


class TestGreedyKeepsNoRunningState:
    def test_asking_about_other_cells_does_not_disturb_a_run(self) -> None:
        # The renderer asks about every cell of a grid in any order, and an
        # evaluation run asks every step. Neither may move the option along.
        agent = an_agent(CORRIDOR)
        agent.values(0)[4] = 1.0
        agent.act(0)
        for state in (3, 2, 1, 0):
            agent.greedy(state)
        assert agent.running == 4
        assert agent.length == 0

    def test_it_gives_a_primitive_action(self) -> None:
        agent = an_agent(CORRIDOR)
        agent.values(0)[4] = 1.0
        assert agent.greedy(0) == CORRIDOR.act(0)


class TestItLearns:
    def test_it_reaches_the_best_four_rooms_policy(self) -> None:
        rng = Rng(1)
        env = four_rooms(rng.stream("env"))
        options = [
            *primitives(env.action_space, range(env.observation_space.n)),
            *hallway_options(env, env.gaps()),
        ]
        agent = OptionsQ(
            rng.stream("agent"),
            env.action_space,
            options,
            step_size=0.5,
            discount=0.95,
            epsilon=0.1,
        )
        train(env, agent, 400, discount=0.95)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=0.95)
        best = value_iteration(env, discount=0.95).start_value
        assert report.reaches_end
        assert report.start_value == pytest.approx(best)

    def test_it_keeps_no_row_for_a_cell_it_never_stands_in(self) -> None:
        rng = Rng(1)
        env = four_rooms(rng.stream("env"))
        options = [
            *primitives(env.action_space, range(env.observation_space.n)),
            *hallway_options(env, env.gaps()),
        ]
        agent = OptionsQ(rng.stream("agent"), env.action_space, options, step_size=0.5)

        stood_in: set[int] = set()
        watched = agent.observe

        def observe(transition: Transition[int]) -> None:
            stood_in.add(transition.observation)
            watched(transition)

        agent.observe = observe  # type: ignore[method-assign]
        train(env, agent, 30, discount=0.95)
        # The option credits the state it started in, which is one the agent
        # stood in, so nothing outside that set gets a row.
        assert set(agent.q) - stood_in == set()


class TestCountingWhatItChose:
    def test_choosing_a_long_option_is_counted(self) -> None:
        agent = an_agent(CORRIDOR)
        agent.values(0)[4] = 1.0
        agent.act(0)
        assert agent.long_chosen == 1

    def test_choosing_a_primitive_one_is_not(self) -> None:
        agent = an_agent()
        agent.act(0)
        assert agent.long_chosen == 0

    def test_the_mean_length_alone_cannot_say_it(self) -> None:
        # A run that never chose a long option and one that chose a long
        # option which stopped after one step give the same mean length.
        short = Option(name="to next", policy={0: 1}, stops=frozenset({1}))
        agent = an_agent(short)
        agent.values(0)[4] = 1.0
        agent.act(0)
        agent.observe(go(0, 1, 0.0, 1))
        assert agent.steps_in_options / agent.finished == 1.0
        assert agent.long_chosen == 1


class TestSayingWhatTheChoiceIs:
    def test_a_cell_where_a_long_option_is_best_says_so(self) -> None:
        agent = an_agent(CORRIDOR)
        agent.values(0)[4] = 1.0
        assert agent.choice_lasts(0)

    def test_a_cell_where_an_action_is_best_does_not(self) -> None:
        agent = an_agent(CORRIDOR)
        agent.values(0)[1] = 1.0
        assert not agent.choice_lasts(0)

    def test_a_cell_no_option_covers_does_not(self) -> None:
        assert not an_agent().choice_lasts(99)

    def test_an_agent_that_only_takes_actions_never_says_so(self) -> None:
        agent = an_agent()
        assert not any(agent.choice_lasts(state) for state in STATES)
