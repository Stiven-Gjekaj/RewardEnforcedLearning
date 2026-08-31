"""One suite that every registered agent has to pass.

The same idea as `tests/test_environments.py`. A test written against one agent
covers one agent. These ask the same questions of all of them, so an agent
added next month is covered the moment its entry lands in the table.
"""

from __future__ import annotations

from typing import Any

import pytest

from rel.agents import AGENTS
from rel.agents.base import Transition
from rel.core import Env
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import evaluate, train

NAMES = AGENTS.names()
KNOWN_TAGS = frozenset(
    {
        "average-reward",
        "bandit",
        "baseline",
        "linear",
        "needs-model",
        "network",
        "off-policy",
        "on-policy",
        "prediction",
        "planning",
        "reference",
        "tabular",
    }
)

#: Which environment each kind of agent is tried on. An agent that keeps a
#: table cannot work on a box of real numbers, and one over a tile coder cannot
#: work on a single integer, so one environment for all of them would only test
#: whichever kind matched it.
SUITABLE = {
    "bandit": "bandit",
    "linear": "mountaincar",
}
DEFAULT_ENVIRONMENT = "cliff"

#: One agent at a time, where the tag is not the reason.
#:
#: `mcts` spends `simulations` times `depth` model steps for every action it
#: takes. On the cliff walk its episodes run to the step limit, because a
#: random rollout there reaches no ending and a rollout that stops at the
#: depth limit is worth minus the depth wherever it went. Five episodes then
#: cost millions of simulated steps, and this file runs several such episodes
#: for every agent. The Dyna maze exercises the same code, settles in about
#: sixteen steps, and costs a second.
#: `linear-td` and `gradient-td` are handed their features by the environment
#: rather than working them out from a box, so the tag they share with the
#: agents over a tile coder points at the wrong environment for them. Baird's
#: counterexample is the one that carries a table.
INSTEAD = {"mcts": "maze", "linear-td": "baird", "gradient-td": "baird"}


def environment_for(name: str, seed: int = 1) -> Env[Any]:
    if name in INSTEAD:
        return ENVIRONMENTS.make(INSTEAD[name], Rng(seed).stream("env"))
    entry = AGENTS[name]
    for tag, environment in SUITABLE.items():
        if tag in entry.tags:
            return ENVIRONMENTS.make(environment, Rng(seed).stream("env"))
    return ENVIRONMENTS.make(DEFAULT_ENVIRONMENT, Rng(seed).stream("env"))


class TestTheTable:
    def test_every_name_is_lower_case_with_hyphens(self) -> None:
        for name in NAMES:
            assert name == name.lower()
            assert " " not in name
            assert "_" not in name

    def test_every_summary_is_a_sentence(self) -> None:
        for entry in AGENTS:
            assert entry.summary.endswith("."), entry.name
            assert entry.summary[0].isupper(), entry.name

    def test_every_tag_is_one_this_project_uses(self) -> None:
        for entry in AGENTS:
            assert set(entry.tags) <= KNOWN_TAGS, entry.name

    def test_every_agent_carries_a_tag(self) -> None:
        for entry in AGENTS:
            assert entry.tags, entry.name

    def test_every_learning_agent_says_whether_it_is_on_policy(self) -> None:
        # The difference decides what a run means, and the cliff walk exists in
        # this project to show it. An agent whose entry does not say is an
        # entry somebody forgot to finish.
        for entry in AGENTS:
            if {"tabular", "linear", "network"} & set(entry.tags):
                assert {"on-policy", "off-policy"} & set(entry.tags), entry.name

    def test_the_settings_of_every_agent_can_be_read(self) -> None:
        for entry in AGENTS:
            settings = entry.options(AGENTS.fixed)
            assert "env" not in settings, entry.name
            assert "rng" not in settings, entry.name


@pytest.mark.parametrize("name", NAMES)
class TestEveryAgent:
    def test_it_builds(self, name: str) -> None:
        env = environment_for(name)
        agent = AGENTS.make(name, Rng(1).stream("agent"), env)
        assert agent.actions == env.action_space

    def test_it_chooses_a_legal_action(self, name: str) -> None:
        env = environment_for(name)
        agent = AGENTS.make(name, Rng(1).stream("agent"), env)
        observation = env.reset()
        agent.start_episode()
        for _ in range(200):
            action = agent.act(observation)
            assert env.action_space.contains(action), name
            outcome = env.step(action)
            observation = outcome.observation
            if outcome.done:
                observation = env.reset()
                agent.start_episode()

    def test_its_greedy_choice_is_legal_too(self, name: str) -> None:
        env = environment_for(name)
        agent = AGENTS.make(name, Rng(1).stream("agent"), env)
        observation = env.reset()
        agent.start_episode()
        assert env.action_space.contains(agent.greedy(observation))

    def test_it_survives_a_short_run(self, name: str) -> None:
        env = environment_for(name)
        agent = AGENTS.make(name, Rng(1).stream("agent"), env)
        record = train(env, agent, 5)
        assert len(record) == 5
        assert record.digest.steps > 0

    def test_it_can_be_watched_without_learning(self, name: str) -> None:
        env = environment_for(name)
        agent = AGENTS.make(name, Rng(1).stream("agent"), env)
        train(env, agent, 3)
        assert len(evaluate(environment_for(name, 2), agent, 2)) == 2

    def test_the_same_seed_gives_the_same_run(self, name: str) -> None:
        def digest(seed: int) -> str:
            env = environment_for(name, seed)
            agent = AGENTS.make(name, Rng(seed).stream("agent"), env)
            return train(env, agent, 5).digest.hexdigest()

        assert digest(4) == digest(4)

    def test_a_different_seed_gives_a_different_run(self, name: str) -> None:
        # Except for the reference, which plays a policy with no chance in it
        # on an environment with no chance in it. Two seeds there really do
        # give the same run, and that is the answer rather than a fault.
        def digest(seed: int) -> str:
            env = environment_for(name, seed)
            agent = AGENTS.make(name, Rng(seed).stream("agent"), env)
            return train(env, agent, 5).digest.hexdigest()

        if "reference" in AGENTS[name].tags:
            assert digest(4) == digest(5)
        else:
            assert digest(4) != digest(5)

    def test_the_counters_move(self, name: str) -> None:
        env = environment_for(name)
        agent = AGENTS.make(name, Rng(1).stream("agent"), env)
        train(env, agent, 4)
        assert agent.episodes == 4
        # The baseline and the reference learn nothing, so they never look at a
        # transition and their step count stays where it started.
        if "baseline" not in AGENTS[name].tags and "reference" not in AGENTS[name].tags:
            assert agent.steps > 0


class TestWhenAnAgentDoesNotFit:
    def test_the_reference_says_so_on_an_environment_with_no_model(self) -> None:
        env = ENVIRONMENTS.make("cartpole", Rng(1).stream("env"))
        with pytest.raises(TypeError, match="keeps no model"):
            AGENTS.make("optimal", Rng(1).stream("agent"), env)

    def test_a_tile_coder_says_so_on_a_table_of_states(self) -> None:
        env = ENVIRONMENTS.make("cliff", Rng(1).stream("env"))
        with pytest.raises(TypeError, match="tile coder divides a Box"):
            AGENTS.make("tile-sarsa", Rng(1).stream("agent"), env)

    def test_a_radial_basis_says_so_on_a_table_of_states(self) -> None:
        env = ENVIRONMENTS.make("cliff", Rng(1).stream("env"))
        with pytest.raises(TypeError, match="radial basis divides a Box"):
            AGENTS.make("rbf-sarsa", Rng(1).stream("agent"), env)

    def test_a_setting_that_does_not_exist_is_named(self) -> None:
        env = ENVIRONMENTS.make("cliff", Rng(1).stream("env"))
        with pytest.raises(TypeError, match="no setting named 'wobble'"):
            AGENTS.make("sarsa", Rng(1).stream("agent"), env, wobble=1.0)


class TestTheReference:
    def test_it_plays_the_best_possible_policy(self) -> None:
        # The reference has to really be the reference, or every number
        # measured against it is measured against nothing.
        from rel.agents.dp import value_iteration

        env = ENVIRONMENTS.make("cliff", Rng(1).stream("env"))
        agent = AGENTS.make("optimal", Rng(1).stream("agent"), env)
        best = value_iteration(env)

        record = evaluate(ENVIRONMENTS.make("cliff", Rng(2).stream("env")), agent, 3)
        assert record.final() == pytest.approx(best.start_value)
        assert record.final() == -13.0


class TestNoAgentLearnsAboutACellItNeverStoodIn:
    """A table row is a claim that the agent has been somewhere.

    Every one of these agents computes its target from the value of the state
    it is moving to. Read through `values` that lookup makes a row, so the goal
    cell of every grid ended up in the table of three of them, and `knows` then
    said the agent had stood in a cell it can only ever arrive at.

    This is the second time this project has had that fault. The first was the
    renderer asking for the greedy action of every cell in order to draw the
    policy, and it is why `peek` exists.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "sarsa",
            "expected-sarsa",
            "q-learning",
            "double-q",
            "n-step-sarsa",
            "sarsa-lambda",
            "q-lambda",
            "dyna-q",
            "dyna-q-plus",
            "monte-carlo",
        ],
    )
    def test_the_table_holds_no_state_it_never_acted_in(self, name: str) -> None:
        root = Rng(1)
        env = ENVIRONMENTS.make("cliff", root.stream("env"))
        agent = AGENTS.make(name, root.stream("agent"), env)

        stood_in: set[int] = set()
        watched = agent.observe

        def observe(transition: Transition[int]) -> None:
            stood_in.add(transition.observation)
            watched(transition)

        agent.observe = observe  # type: ignore[method-assign]
        train(env, agent, 60, discount=1.0)

        extra = set(agent.q) - stood_in  # type: ignore[attr-defined]
        assert extra == set(), f"{name} has rows for {sorted(extra)}"
