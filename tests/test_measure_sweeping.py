"""The per seed half of `scripts/measure_sweeping.py`.

The page prints a row for every seed beside the median that summarises them,
and until the check went looking for the command behind that row there was no
way to print it. This is that half.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_sweeping.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_sweeping", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def measured(script: ModuleType) -> tuple[tuple[str, ...], list[str]]:
    # Twelve episodes and three seeds. `dyna-q` solves two of the three in
    # that many, so one run gives both a number and a dash to check.
    row: tuple[tuple[str, ...], list[str]] = script.measure("dyna-q", "maze", 12, 3, 5)
    return row


class TestEverySeedIsReported:
    def test_one_entry_for_each_seed(
        self, measured: tuple[tuple[str, ...], list[str]]
    ) -> None:
        assert len(measured[1]) == 3

    def test_a_seed_that_never_solved_is_a_dash(
        self, measured: tuple[tuple[str, ...], list[str]]
    ) -> None:
        """Kept in place rather than left out.

        The seeds are a row of a table with a column for each, so a seed that
        solved nothing has to hold its column. Dropping it would shift every
        seed after it one place left and the row would be wrong about which
        seed did what.
        """
        assert measured[1] == ["3534", "5412", "-"]

    def test_the_summary_counts_the_ones_that_did(
        self, measured: tuple[tuple[str, ...], list[str]]
    ) -> None:
        row, each = measured
        assert row[1] == f"{sum(1 for value in each if value != '-')} of 3"


class TestTheSummaryIsMadeOfThoseNumbers:
    def test_the_ends_come_from_the_seeds(
        self, measured: tuple[tuple[str, ...], list[str]]
    ) -> None:
        """The reason both are returned together.

        A median printed beside numbers it was not computed from is the
        failure this project keeps finding in its own documentation. Here the
        two cannot drift, because one run produces both.
        """
        row, each = measured
        solved = sorted(int(value) for value in each if value != "-")
        assert row[3] == str(min(solved))
        assert row[4] == str(max(solved))


class TestTheEpisodeCapDoesNotChangeTheAnswer:
    def test_more_episodes_give_the_same_updates(self, script: ModuleType) -> None:
        """Worth knowing and not obvious.

        What is reported is the update count at which the greedy policy first
        became optimal. Once a seed has reached that, running longer cannot
        move it, so the cap only decides which seeds get there at all.

        The documented run is 400 episodes and prints 1060, 880 and 965 for
        the first three seeds. Twelve episodes prints the same three.
        """
        short = script.measure("prioritised-sweeping", "maze", 12, 3, 5)[1]
        longer = script.measure("prioritised-sweeping", "maze", 30, 3, 5)[1]
        assert short == longer == ["1060", "880", "965"]
