"""Tests for control over states grouped together.

The load bearing test is `TestAtFullResolutionItIsTheTable`. This agent exists
to be a dial with a table at one end, and the claim that one end really is the
table is what makes every rung below it a comparison rather than two different
agents. A one hot row turns `w[i] += a * d * x[i]` into `q[s] += a * d`, so the
two should agree exactly and not nearly.
"""

from __future__ import annotations

import pytest

from rel.agents import AGENTS
from rel.agents.base import Transition
from rel.agents.linear import SemiGradientQ, SemiGradientSarsa
from rel.agents.lookup import aggregated
from rel.envs import ENVIRONMENTS
from rel.rng import Rng


def an_env(name: str = "boatrace") -> object:
    return ENVIRONMENTS.make(name, Rng(1).stream("env"))


def steps(count: int, states: int) -> list[Transition[int, int]]:
    return [
        Transition(
            step % states, step % 2, float(step % 3), (step + 1) % states, False, False
        )
        for step in range(count)
    ]


class TestAtFullResolutionItIsTheTable:
    def test_it_agrees_with_q_learning_exactly(self) -> None:
        env = an_env()
        table = AGENTS.make(
            "q-learning",
            Rng(4).stream("agent"),
            env,
            step_size=0.1,
            discount=1.0,
            epsilon=0.0,
        )
        grouped = AGENTS.make(
            "grouped-q",
            Rng(4).stream("agent"),
            env,
            step_size=0.1,
            discount=1.0,
            epsilon=0.0,
        )
        for step in steps(60, 16):
            table.observe(step)
            grouped.observe(step)

        for state in range(16):
            assert list(grouped.action_values(state)) == list(
                table.action_values(state)
            )

    def test_it_agrees_with_sarsa_exactly(self) -> None:
        env = an_env()
        table = AGENTS.make(
            "sarsa",
            Rng(4).stream("agent"),
            env,
            step_size=0.1,
            discount=1.0,
            epsilon=0.0,
        )
        grouped = AGENTS.make(
            "grouped-sarsa",
            Rng(4).stream("agent"),
            env,
            step_size=0.1,
            discount=1.0,
            epsilon=0.0,
        )
        for step in steps(60, 16):
            table.observe(step)
            grouped.observe(step)

        for state in range(16):
            assert list(grouped.action_values(state)) == list(
                table.action_values(state)
            )

    def test_no_groups_asked_for_means_one_for_each_state(self) -> None:
        agent = AGENTS.make("grouped-q", Rng(1).stream("agent"), an_env())
        assert agent.coder.features == 16

    def test_asking_for_all_of_them_is_the_same_thing(self) -> None:
        asked = AGENTS.make("grouped-q", Rng(1).stream("agent"), an_env(), groups=16)
        assert asked.coder.features == 16


class TestTheDial:
    @pytest.mark.parametrize("groups", [1, 2, 4, 8, 16])
    def test_it_holds_one_weight_for_each_group(self, groups: int) -> None:
        agent = AGENTS.make(
            "grouped-q", Rng(1).stream("agent"), an_env(), groups=groups
        )
        assert agent.coder.features == groups
        assert all(len(row) == groups for row in agent.weights)

    def test_one_group_makes_every_state_the_same(self) -> None:
        agent = AGENTS.make(
            "grouped-q",
            Rng(1).stream("agent"),
            an_env(),
            groups=1,
            step_size=0.1,
            discount=1.0,
            epsilon=0.0,
        )
        for step in steps(40, 16):
            agent.observe(step)

        first = list(agent.action_values(0))
        for state in range(1, 16):
            assert list(agent.action_values(state)) == first

    def test_learning_at_one_state_moves_its_neighbours(
        self,
    ) -> None:
        # Which is what an approximator is. A table cannot do this.
        agent = AGENTS.make(
            "grouped-q",
            Rng(1).stream("agent"),
            an_env(),
            groups=4,
            step_size=0.5,
            discount=1.0,
            epsilon=0.0,
        )
        before = list(agent.action_values(1))
        agent.observe(Transition(0, 0, 10.0, 1, True, False))
        assert list(agent.action_values(1)) != before

    def test_a_state_in_another_group_is_untouched(self) -> None:
        agent = AGENTS.make(
            "grouped-q",
            Rng(1).stream("agent"),
            an_env(),
            groups=4,
            step_size=0.5,
            discount=1.0,
            epsilon=0.0,
        )
        before = list(agent.action_values(15))
        agent.observe(Transition(0, 0, 10.0, 1, True, False))
        assert list(agent.action_values(15)) == before

    def test_more_groups_than_states_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no more of them than"):
            AGENTS.make("grouped-q", Rng(1).stream("agent"), an_env(), groups=17)

    def test_fewer_than_none_is_refused_rather_than_read_as_the_default(
        self,
    ) -> None:
        # Zero is the way of asking for one group per state. A negative count
        # is a mistake, and reading it as the default would hand back a table
        # to somebody who asked for something impossible.
        with pytest.raises(ValueError, match="at least one"):
            AGENTS.make("grouped-q", Rng(1).stream("agent"), an_env(), groups=-5)


class TestWhatItRefuses:
    def test_an_observation_that_is_not_a_number_is_refused(self) -> None:
        # States are grouped by their number, so there has to be one.
        with pytest.raises(TypeError, match="tagged 'tabular'"):
            AGENTS.make("grouped-q", Rng(1).stream("agent"), an_env("mountaincar"))

    def test_the_message_names_the_environment_and_what_it_gave(self) -> None:
        with pytest.raises(TypeError, match="mountaincar has a Box observation"):
            AGENTS.make("grouped-sarsa", Rng(1).stream("agent"), an_env("mountaincar"))


class TestTheClassIsGenericInWhatItReads:
    def test_it_takes_a_coder_over_whole_numbers(self) -> None:
        agent: SemiGradientQ[int] = SemiGradientQ(
            Rng(1),
            ENVIRONMENTS.make("boatrace", Rng(1)).action_space,
            aggregated(16, 4),
        )
        assert agent.coder.features == 4
        assert len(agent.action_values(3)) == 2

    def test_sarsa_takes_one_too(self) -> None:
        agent: SemiGradientSarsa[int] = SemiGradientSarsa(
            Rng(1),
            ENVIRONMENTS.make("boatrace", Rng(1)).action_space,
            aggregated(16, 8),
        )
        assert agent.coder.features == 8


class TestTheRegistry:
    def test_both_are_there_and_tagged_linear(self) -> None:
        for name in ("grouped-sarsa", "grouped-q"):
            assert "linear" in AGENTS[name].tags

    def test_they_differ_in_being_on_policy(self) -> None:
        assert "on-policy" in AGENTS["grouped-sarsa"].tags
        assert "off-policy" in AGENTS["grouped-q"].tags

    def test_groups_is_a_setting(self) -> None:
        assert "groups" in AGENTS["grouped-q"].options(AGENTS.fixed)
