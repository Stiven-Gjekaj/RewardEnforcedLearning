"""The measurement behind the aliased corridor table on the algorithms page.

The load bearing test is `TestTheMirageColumn`. The whole point of that table
is the gap between what an agent scores while learning and what the policy it
learned is worth, and the two are told apart by one argument to one function.
If `watched` ran with the learning still on, both columns would say the same
thing, the table would still print, and the finding would quietly vanish.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.agents import AGENTS
from rel.envs import ENVIRONMENTS
from rel.envs.aliased import ANYWHERE, AliasedCorridor
from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_aliased.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_aliased", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def an_env() -> AliasedCorridor:
    env = ENVIRONMENTS.make("aliased", Rng(1).stream("env"))
    assert isinstance(env, AliasedCorridor)
    return env


class TestTheMirageColumn:
    def test_the_frozen_columns_do_not_teach_the_agent(
        self, script: ModuleType
    ) -> None:
        env = an_env()
        agent = AGENTS.make("q-learning", Rng(1).stream("agent"), env, epsilon=0.1)
        for _ in range(20):
            script.watched(env, agent, 5, greedy=False)
        before = list(agent.action_values(ANYWHERE))
        script.watched(env, agent, 20, greedy=False)
        script.watched(env, agent, 3, greedy=True)
        assert list(agent.action_values(ANYWHERE)) == before

    def test_greedy_and_its_own_policy_are_different_runs(
        self, script: ModuleType
    ) -> None:
        # An agent that ranks its actions never finishes greedily here and
        # does finish with its exploring on, so the two columns must not be
        # the same call.
        env = an_env()
        agent = AGENTS.make("q-learning", Rng(2).stream("agent"), env, epsilon=0.1)
        from rel.training import train

        train(env, agent, 300, discount=1.0)
        assert script.watched(env, agent, 3, greedy=True) > 500
        assert script.watched(env, agent, 100, greedy=False) < 200


class TestWhichAgentsGetAnEpsilon:
    def test_a_ranking_agent_gets_one(self, script: ModuleType) -> None:
        assert script.exploring("q-learning", 0.1) == {"epsilon": 0.1}

    def test_a_policy_gradient_agent_gets_none(self, script: ModuleType) -> None:
        # Not an oversight. It holds a probability, and exploring is what the
        # probability already is.
        assert script.exploring("reinforce", 0.1) == {}

    def test_it_asks_the_registry_rather_than_holding_a_list(
        self, script: ModuleType
    ) -> None:
        for name in script.AGENT_NAMES:
            wanted = "epsilon" in AGENTS[name].options(AGENTS.fixed)
            assert bool(script.exploring(name, 0.1)) == wanted


class TestHowItReadsAPolicy:
    def test_a_policy_gradient_agent_hands_over_its_probability(
        self, script: ModuleType
    ) -> None:
        env = an_env()
        agent = AGENTS.make("reinforce", Rng(1).stream("agent"), env)
        share = script.share_of_right(agent)
        assert 0.0 < share < 1.0
        assert share == pytest.approx(agent.probabilities(ANYWHERE)[1])

    def test_a_ranking_agent_has_no_probability_to_hand_over(
        self, script: ModuleType
    ) -> None:
        env = an_env()
        agent = AGENTS.make("q-learning", Rng(1).stream("agent"), env, epsilon=0.1)
        assert script.share_of_right(agent) in (0.0, 1.0)


class TestWhatItPrintsAboutTheArithmetic:
    @pytest.fixture
    def printed(self, script: ModuleType, capsys: pytest.CaptureFixture[str]) -> str:
        script.closed_form_section((0.2, 0.5858, 0.95), 0.1)
        return capsys.readouterr().out

    def test_it_names_both_reference_numbers(self, printed: str) -> None:
        assert "0.5858" in printed
        assert "11.66" in printed
        assert "44.21" in printed

    def test_it_says_where_the_best_share_comes_from(self, printed: str) -> None:
        assert "2 - sqrt 2" in printed

    def test_it_says_where_a_ranking_is_stuck(self, printed: str) -> None:
        assert "1 - epsilon / 2" in printed

    def test_the_shares_it_prints_bracket_the_best_one(
        self, script: ModuleType
    ) -> None:
        # A table that only showed shares on one side of the best would look
        # like the cost rises in one direction, and it rises in both.
        best = AliasedCorridor.best_share()
        assert min(script.SHARES) < best < max(script.SHARES)
