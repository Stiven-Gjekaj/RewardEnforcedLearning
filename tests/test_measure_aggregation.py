"""The measurement behind the state aggregation table on the algorithms page.

The load bearing test is `TestTheFloorIsArithmetic`. The floor is what no
amount of learning can beat, so it has to be worked out from the true values
and not from a run, and a floor that a run could beat would be a floor nobody
should read.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.agents.lookup import aggregated

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_aggregation.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_aggregation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheValuesItMeasuresAgainst:
    def test_there_is_one_for_each_state(self, script: ModuleType) -> None:
        assert len(script.true_values()) == 1002

    def test_the_two_endings_are_left_out_of_the_scoring(
        self, script: ModuleType
    ) -> None:
        # An agent is never in one, so its estimate there never moves off what
        # it started at, and scoring it would measure the starting value.
        assert list(script.cells()) == list(range(1, 1001))

    def test_they_rise_from_one_end_to_the_other(self, script: ModuleType) -> None:
        values = script.true_values()
        walk = [values[state] for state in script.cells()]
        assert walk == sorted(walk)

    def test_they_are_read_once(self, script: ModuleType) -> None:
        # The sweep is over a thousand states with a hundred branches each and
        # every row of both tables wants the same answer.
        assert script.true_values() is script.true_values()


class TestTheGroupsMatchTheCoder:
    """The floor is only the floor of what the agent is doing if the split it
    is worked out over is the split the agent was given."""

    @pytest.mark.parametrize("groups", [1, 3, 10, 97])
    def test_the_split_is_the_one_the_coder_makes(
        self, script: ModuleType, groups: int
    ) -> None:
        states = len(script.true_values())
        coder = aggregated(states, groups)
        for state in range(states):
            assert coder.encode(state)[0][0] == script.group_of(state, groups, states)


class TestTheFloorIsArithmetic:
    def test_one_group_is_the_spread_of_everything(self, script: ModuleType) -> None:
        # One number for a thousand cells cannot be less wrong than the spread
        # of what it is averaging.
        values = [script.true_values()[state] for state in script.cells()]
        assert script.floor_for(1) == pytest.approx(statistics.pstdev(values), abs=1e-9)

    def test_a_group_for_each_state_is_no_error_at_all(
        self, script: ModuleType
    ) -> None:
        assert script.floor_for(len(script.true_values())) == pytest.approx(0.0)

    def test_more_groups_never_raises_the_floor(self, script: ModuleType) -> None:
        floors = [script.floor_for(groups) for groups in (1, 2, 5, 10, 20, 50, 100)]
        assert floors == sorted(floors, reverse=True)

    def test_doubling_the_groups_about_halves_the_floor(
        self, script: ModuleType
    ) -> None:
        """Which is what a staircase does to a line, and says the floor is one.

        The true values are nearly straight, and the best a step of half the
        width can do is half the error, so each doubling takes about half off.
        """
        for groups in (2, 5, 10, 25):
            assert script.floor_for(2 * groups) == pytest.approx(
                script.floor_for(groups) / 2.0, rel=0.1
            )

    def test_nothing_that_learns_beats_it(self, script: ModuleType) -> None:
        # The one thing a floor has to be. An agent that came in under it would
        # mean the floor was worked out over a different split or different
        # values than the agent was learning.
        for groups in (2, 10, 50):
            reached = script.one_run(groups, seed=1, episodes=300, step_size=0.2)
            assert reached > script.floor_for(groups), groups


class TestARun:
    def test_it_learns_something(self, script: ModuleType) -> None:
        started = script.one_run(10, seed=1, episodes=0, step_size=0.2)
        learned = script.one_run(10, seed=1, episodes=300, step_size=0.2)
        assert learned < started

    def test_a_seed_replays_it(self, script: ModuleType) -> None:
        assert script.one_run(5, 2, 40, 0.2) == script.one_run(5, 2, 40, 0.2)

    def test_two_seeds_do_not_give_the_same_run(self, script: ModuleType) -> None:
        assert script.one_run(5, 2, 40, 0.2) != script.one_run(5, 3, 40, 0.2)


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            ["measure_aggregation", "--runs", "1", "--episodes", "20", *extra],
        )
        return int(script.main())

    def test_both_tables_are_printed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch, "--groups", "1", "10") == 0
        printed = capsys.readouterr().out
        assert "best a staircase can do" in printed
        assert "closed form" in printed
        assert "Lowest floor" in printed
        assert "Lowest reached" in printed

    def test_the_closed_form_table_is_the_check_that_it_is_arithmetic(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The sweep and the formula, side by side, on the walk whose values
        # are known without any sweep at all.
        self._run(script, monkeypatch, "--groups", "1")
        printed = capsys.readouterr().out.split("closed form")[1]
        assert "0.166667" in printed
        assert "0.833333" in printed

    def test_the_groups_it_was_given_are_the_ones_it_walks(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--groups", "2", "4")
        ladder = capsys.readouterr().out.split("The true values")[0]
        rungs = [
            line.split()[0]
            for line in ladder.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        assert rungs == ["2", "4"]
