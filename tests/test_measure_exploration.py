"""The one piece of `scripts/measure_exploration.py` worth pinning.

Everything else in that script prints a table. `first_arrival` decides what
counts as having found the goal, and the first version of it decided wrongly:
it asked whether the return was above zero, which is right on a grid where only
the goal pays and reports "never" on every run of the cliff walk, where every
return is negative and the goal is reached on almost all of them.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rel.training import Record

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_exploration.py"


def loaded() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_exploration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(pairs: list[tuple[bool, int]], reward: float = 0.0) -> Record:
    """A record of episodes, each an ending or not and a length."""
    made = Record()
    for ended, length in pairs:
        made.terminated.append(ended)
        made.lengths.append(length)
        made.returns.append(reward)
    return made


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return loaded()


class TestFirstArrival:
    def test_it_names_the_episode_and_the_steps_before_it(
        self, script: ModuleType
    ) -> None:
        made = record([(False, 100), (False, 200), (True, 47)])
        assert script.first_arrival(made) == (3, 300)

    def test_arriving_first_costs_nothing(self, script: ModuleType) -> None:
        assert script.first_arrival(record([(True, 47)])) == (1, 0)

    def test_never_arriving_is_none_rather_than_the_last_episode(
        self, script: ModuleType
    ) -> None:
        """`None` and not 3, which would read as a run that arrived at the end."""
        made = record([(False, 100), (False, 100), (False, 100)])
        assert script.first_arrival(made) == (None, 300)

    def test_a_negative_return_still_counts_as_arriving(
        self, script: ModuleType
    ) -> None:
        """The fault this test exists for.

        Every return of the cliff walk is negative, and reaching the goal there
        is what the column is asking about.
        """
        made = record([(False, 500), (True, 17)], reward=-17.0)
        assert script.first_arrival(made) == (2, 500)


class TestTheArms:
    def test_there_are_four_of_them(self, script: ModuleType) -> None:
        assert len(script.ARMS) == 4

    def test_each_names_a_rule_the_agents_accept(self, script: ModuleType) -> None:
        from rel.agents.explore import as_rule

        for _, explore, _ in script.ARMS:
            as_rule(explore)

    def test_only_the_optimistic_one_starts_above_zero(
        self, script: ModuleType
    ) -> None:
        """Optimism is a starting value rather than a rule, so it is the
        column that varies where the other three vary the rule."""
        raised = [label for label, _, optimism in script.ARMS if optimism > 0.0]
        assert raised == ["optimistic"]
