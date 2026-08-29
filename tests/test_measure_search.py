"""The two pieces of `scripts/measure_search.py` worth pinning.

`model_steps` is what puts three agents on one axis, and it reads a different
attribute for each of them, so a rename anywhere would quietly turn a column of
work into a column of zeros. `_steps_to_end` counts a route by walking it,
which is the correction that a length read off a discounted value is one short.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rel.agents import AGENTS
from rel.agents.base import Transition
from rel.envs import ENVIRONMENTS
from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_search.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_search", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestModelSteps:
    def test_a_planner_reports_its_replays(self, script: ModuleType) -> None:
        agent = AGENTS.make(
            "dyna-q",
            Rng(1).stream("agent"),
            ENVIRONMENTS.make("maze", Rng(1).stream("env")),
            planning_steps=4,
        )
        agent.observe(Transition(0, 0, 0.0, 1, False, False))
        assert script.model_steps(agent) == 4

    def test_a_search_reports_its_simulated_steps(self, script: ModuleType) -> None:
        env = ENVIRONMENTS.make("maze", Rng(1).stream("env"))
        agent = AGENTS.make("mcts", Rng(1).stream("agent"), env, simulations=5, depth=3)
        agent.act(env.reset())
        assert script.model_steps(agent) > 0

    def test_an_agent_that_asks_no_model_reports_nothing(
        self, script: ModuleType
    ) -> None:
        agent = AGENTS.make(
            "q-learning",
            Rng(1).stream("agent"),
            ENVIRONMENTS.make("maze", Rng(1).stream("env")),
        )
        assert script.model_steps(agent) == 0

    def test_it_reads_two_names_and_no_more(self, script: ModuleType) -> None:
        """An agent counting something else calls it something else.

        `options-q` keeps an `updates` counter of its own that means table
        updates rather than model steps, and a search over every integer
        attribute would report it here as though it belonged.
        """

        class Counting:
            updates = 99

        assert script.model_steps(Counting()) == 0


class TestStepsToEnd:
    @pytest.mark.parametrize(
        ("grid", "discount", "steps"),
        [("cliff", 1.0, 13), ("maze", 0.95, 14), ("rooms", 1.0, 20)],
    )
    def test_it_counts_the_route_by_walking_it(
        self, script: ModuleType, grid: str, discount: float, steps: int
    ) -> None:
        from rel.agents.dp import value_iteration

        env = ENVIRONMENTS.make(grid, Rng(1).stream("env"))
        solved = value_iteration(env, discount=discount)
        assert script._steps_to_end(env, list(solved.policy)) == steps

    def test_the_cliff_walk_agrees_with_its_own_return(
        self, script: ModuleType
    ) -> None:
        """Thirteen steps at -1 each is a return of -13, and that one can be
        read straight off the value because the discount there is one."""
        from rel.agents.dp import value_iteration

        env = ENVIRONMENTS.make("cliff", Rng(1).stream("env"))
        solved = value_iteration(env, discount=1.0)
        assert script._steps_to_end(env, list(solved.policy)) == -solved.start_value


def test_every_arm_names_a_registered_agent(script: ModuleType) -> None:
    for _, name, _ in script.ARMS:
        assert name in AGENTS.names()
