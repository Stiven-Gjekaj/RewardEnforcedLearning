"""Tests for learning from a model the agent builds itself."""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.dyna import DynaQ, DynaQPlus
from rel.agents.sweeping import PrioritisedSweeping
from rel.agents.td import QLearning
from rel.envs.classic import dyna_maze
from rel.envs.gridworld import GridWorld
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import digest_of, train

TWO = Discrete(2)


def go(state: int, action: int, reward: float, landed: int) -> Transition[int]:
    return Transition(state, action, reward, landed, False, False)


class TestTheModel:
    def test_a_step_is_remembered(self) -> None:
        agent = DynaQ(Rng(1), TWO, planning_steps=0)
        agent.observe(go(0, 1, 3.0, 4))
        assert agent.model[(0, 1)] == (3.0, 4, False)

    def test_the_newest_result_replaces_the_old_one(self) -> None:
        # One result for each state and action. This is right where the same
        # action always does the same thing and wrong where it does not, and
        # the class docstring says which environments those are.
        agent = DynaQ(Rng(1), TWO, planning_steps=0)
        agent.observe(go(0, 1, 3.0, 4))
        agent.observe(go(0, 1, 9.0, 7))
        assert agent.model[(0, 1)] == (9.0, 7, False)
        assert len(agent._seen) == 1

    def test_no_planning_is_plain_q_learning(self) -> None:
        steps = [go(0, 0, -1.0, 1), go(1, 1, -1.0, 2), go(2, 0, 5.0, 3)]

        planner = DynaQ(Rng(1), TWO, planning_steps=0, step_size=0.5, discount=0.9)
        plain = QLearning(Rng(1), TWO, step_size=0.5, discount=0.9)
        for agent in (planner, plain):
            for step in steps:
                agent.observe(step)
            agent.end_episode()

        assert dict(planner.q) == dict(plain.q)

    def test_planning_moves_entries_that_the_step_did_not_touch(self) -> None:
        # This is the whole point. One real step at state one is replayed at
        # state zero as well, so a value moves in a cell the agent was not
        # standing in.
        agent = DynaQ(Rng(1), TWO, planning_steps=0, step_size=1.0, discount=1.0)
        agent.observe(go(0, 0, 0.0, 1))
        agent.observe(go(1, 0, 10.0, 2))
        before = agent.q[0][0]

        agent.planning_steps = 200
        agent.observe(go(1, 0, 10.0, 2))
        assert agent.q[0][0] > before

    def test_a_negative_number_of_planning_steps_is_refused(self) -> None:
        with pytest.raises(ValueError, match="below zero"):
            DynaQ(Rng(1), TWO, planning_steps=-1)


class TestPlanningPays:
    def test_more_planning_reaches_a_short_path_in_fewer_episodes(self) -> None:
        # Sutton and Barto, figure 8.2. On a maze where nothing pays until the
        # goal, the first episode teaches one cell without planning and a whole
        # path with it.
        def lengths(planning: int) -> list[int]:
            rng = Rng(3)
            env = dyna_maze(rng.stream("env"))
            agent = DynaQ(
                rng.stream("agent"),
                env.action_space,
                planning_steps=planning,
                step_size=0.1,
                epsilon=0.1,
                discount=0.95,
            )
            return train(env, agent, 20).lengths

        without = lengths(0)
        with_plan = lengths(50)

        # By the fifth episode the planner is near the shortest path and the
        # one that does not plan is not.
        assert sum(with_plan[4:10]) * 3 < sum(without[4:10])


class TestDynaQPlus:
    def test_it_pays_a_bonus_for_a_step_it_has_not_taken_lately(self) -> None:
        agent = DynaQPlus(
            Rng(1), TWO, kappa=1.0, planning_steps=0, step_size=1.0, discount=0.0
        )
        agent.observe(go(0, 0, 0.0, 1))
        agent.last_taken[(0, 0)] = agent.steps - 100

        agent._replay(0, 0, 0.0, 1, False)
        # The reward replayed is zero plus one times the square root of a
        # hundred, and with no discount the target is that bonus alone.
        assert agent.q[0][0] == pytest.approx(10.0)

    def test_an_action_never_taken_at_a_visited_state_is_in_the_model(self) -> None:
        # Without this the bonus can only be paid for something already tried,
        # and the corner that was never entered stays invisible.
        agent = DynaQPlus(Rng(1), Discrete(4), planning_steps=0)
        agent.observe(go(0, 2, 1.0, 1))
        assert set(agent.model) >= {(0, 0), (0, 1), (0, 2), (0, 3)}

    def test_an_untried_action_is_modelled_as_doing_nothing(self) -> None:
        agent = DynaQPlus(Rng(1), Discrete(4), planning_steps=0)
        agent.observe(go(5, 2, 1.0, 6))
        assert agent.model[(5, 0)] == (0.0, 5, False)

    def test_it_finds_a_shortcut_that_opens_after_it_has_learned_a_route(
        self,
    ) -> None:
        # Sutton and Barto, figure 8.5. A wall with a gap on the left. The
        # agent learns the long way round, and then a second gap opens on the
        # right that is much shorter. Plain Dyna-Q has a route that works and
        # no reason to look for another, so it keeps the long one.
        #
        # Measured on seed 5: the long way runs about 18 steps and the agent
        # that goes looking settles at about 12.
        before = (
            "........G",
            ".########",
            ".........",
            ".........",
            ".........",
            "...S.....",
        )
        after = (
            "........G",
            ".#######.",
            ".........",
            ".........",
            ".........",
            "...S.....",
        )

        def length_after_the_shortcut_opens(
            builder: type[DynaQ], **extra: float
        ) -> float:
            rng = Rng(5)
            settings: dict[str, float] = {
                "planning_steps": 20,
                "step_size": 0.5,
                "epsilon": 0.1,
                "discount": 0.95,
                **extra,
            }
            env = GridWorld(
                rng.stream("env"),
                before,
                step_reward=0.0,
                goal_reward=1.0,
                max_episode_steps=3000,
            )
            agent = builder(rng.stream("agent"), env.action_space, **settings)  # type: ignore[arg-type]
            train(env, agent, 60)

            opened = GridWorld(
                Rng(5).stream("moved"),
                after,
                step_reward=0.0,
                goal_reward=1.0,
                max_episode_steps=3000,
            )
            record = train(opened, agent, 120)
            return sum(record.lengths[-40:]) / 40

        plain = length_after_the_shortcut_opens(DynaQ)
        curious = length_after_the_shortcut_opens(DynaQPlus, kappa=0.001)

        assert plain > 16.0
        assert curious < 14.0

    def test_a_bonus_larger_than_the_rewards_wrecks_it(self) -> None:
        # The bonus is added to a remembered reward, so it has to be small
        # against the rewards the environment really pays. On this maze the
        # goal pays one and nothing else pays anything. With kappa at 0.01 the
        # bonus passes that after a thousand steps, the planning stops being
        # about the environment, and the agent wanders.
        #
        # This is a setting with no safe default, and it is the reason kappa is
        # an argument with 0.001 behind it rather than a constant.
        maze = (
            "........G",
            ".########",
            ".........",
            ".........",
            ".........",
            "...S.....",
        )

        def length(kappa: float) -> float:
            rng = Rng(5)
            env = GridWorld(
                rng.stream("env"),
                maze,
                step_reward=0.0,
                goal_reward=1.0,
                max_episode_steps=600,
            )
            agent = DynaQPlus(
                rng.stream("agent"),
                env.action_space,
                kappa=kappa,
                planning_steps=20,
                step_size=0.5,
                epsilon=0.1,
                discount=0.95,
            )
            record = train(env, agent, 40)
            return sum(record.lengths[-10:]) / 10

        assert length(0.001) < 40.0
        assert length(0.05) > 300.0


class TestWhatItLearned:
    def test_the_model_is_part_of_it(self) -> None:
        # Two agents holding the same table and different models believe
        # different things about the environment, and the next replay will
        # show it. A digest over the table alone would call them one agent.
        one = DynaQ(Rng(1), TWO, planning_steps=0)
        other = DynaQ(Rng(1), TWO, planning_steps=0)
        one.observe(go(0, 0, -1.0, 1))
        other.observe(go(0, 0, -1.0, 2))

        assert one.q == other.q
        assert digest_of(one) != digest_of(other)

    def test_the_table_is_still_part_of_it(self) -> None:
        one = DynaQ(Rng(1), TWO, planning_steps=0, step_size=0.5)
        other = DynaQ(Rng(1), TWO, planning_steps=0, step_size=0.5)
        one.observe(go(0, 0, -1.0, 1))
        other.observe(go(0, 0, -5.0, 1))
        assert digest_of(one) != digest_of(other)


class TestItCountsWhatItSpends:
    """Every planner here works by asking a model what would happen.

    `prioritised-sweeping` counted its replays and the tree search counts its
    simulated steps. Dyna counted nothing, so the three could not be put on one
    axis, and a comparison against episodes lets an agent that asks a thousand
    times a step look free.
    """

    def test_a_step_makes_its_quota_of_replays(self) -> None:
        agent = DynaQ(Rng(1), Discrete(2), planning_steps=7, epsilon=0.0)
        agent.observe(Transition(0, 0, 1.0, 1, False, False))
        assert agent.replays == 7

    def test_the_first_step_has_nothing_to_replay_before_it(self) -> None:
        """The model is empty until a step has been remembered."""
        agent = DynaQ(Rng(1), Discrete(2), planning_steps=7, epsilon=0.0)
        assert agent.replays == 0

    def test_no_planning_spends_nothing(self) -> None:
        agent = DynaQ(Rng(1), Discrete(2), planning_steps=0, epsilon=0.0)
        for _ in range(5):
            agent.observe(Transition(0, 0, 1.0, 1, False, False))
        assert agent.replays == 0

    def test_the_sweeper_counts_the_same_thing(self) -> None:
        """It inherits the counter rather than declaring a second one."""
        agent = PrioritisedSweeping(Rng(1), Discrete(2), planning_steps=5)
        agent.observe(Transition(0, 0, 1.0, 1, True, False))
        assert agent.replays > 0
