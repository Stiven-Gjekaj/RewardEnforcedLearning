"""The measurement behind the exploration bonus table on the algorithms page.

That table stood for nineteen tracks with no command that could print it. It
was right, and every one of its four numbers comes out of this script exactly,
which is the part worth knowing: `scripts/check_numbers.py` found a table
nothing produced rather than a number that had moved.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_shortcut.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_shortcut", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheTwoGrids:
    def test_they_differ_by_one_cell(self, script: ModuleType) -> None:
        """The second gap, and nothing else.

        A grid that differed anywhere else would make this a measurement of
        two mazes rather than of one maze that changed.
        """
        pairs = list(zip(script.BEFORE, script.AFTER, strict=True))
        moved = [
            (row, column)
            for row, (was, now) in enumerate(pairs)
            for column, (one, two) in enumerate(zip(was, now, strict=True))
            if one != two
        ]
        assert len(moved) == 1
        row, column = moved[0]
        assert script.BEFORE[row][column] == "#"
        assert script.AFTER[row][column] == "."

    def test_the_new_gap_is_on_the_right(self, script: ModuleType) -> None:
        # The agent learns the way round the left gap first, so a second gap
        # on the left would be the route it already has.
        row = next(
            index
            for index, line in enumerate(script.AFTER)
            if line != script.BEFORE[index]
        )
        assert script.AFTER[row].index(".") == 0
        assert script.AFTER[row].rindex(".") == len(script.AFTER[row]) - 1


class TestOneRun:
    def test_the_same_seed_gives_the_same_number(self, script: ModuleType) -> None:
        one = script.one_run(1, None, 4, 6)
        assert one == script.one_run(1, None, 4, 6)

    def test_a_different_seed_gives_a_different_number(
        self, script: ModuleType
    ) -> None:
        lengths = {script.one_run(seed, None, 4, 6) for seed in (1, 2, 3, 4)}
        assert len(lengths) > 1

    def test_a_run_shorter_than_the_tail_is_not_divided_by_the_tail(
        self, script: ModuleType
    ) -> None:
        """The fault this had while the episode counts were fixed numbers.

        The mean was the sum of the last forty episodes over forty, and a run
        of six episodes has six of them. It reported a sixth of the truth and
        it looked like a result.
        """
        assert script.TAIL > 6
        short = script.one_run(1, None, 4, 6)
        assert short > 10.0

    def test_the_bonus_finds_the_shortcut_and_no_bonus_does_not(
        self, script: ModuleType
    ) -> None:
        # The claim the whole table exists to make, on one seed and a tenth
        # of the episodes. The long way is about eighteen steps and the new
        # gap makes it about twelve.
        plain = script.one_run(1, None)
        curious = script.one_run(1, 0.001)
        assert plain > 16.0
        assert curious < 14.0

    def test_a_bonus_larger_than_the_rewards_wrecks_it(
        self, script: ModuleType
    ) -> None:
        # The goal pays one and nothing else pays anything, so a kappa of
        # 0.01 passes that after a thousand steps and the planning stops
        # being about the environment.
        assert script.one_run(1, 0.01) > 300.0


class TestMeasure:
    def test_one_number_for_each_seed(self, script: ModuleType) -> None:
        assert len(script.measure(None, 3, 4, 6)) == 3

    def test_the_seeds_are_one_upwards(self, script: ModuleType) -> None:
        # So that two settings meet the same seeds, which is what makes the
        # rows of the table comparable.
        assert script.measure(None, 2, 4, 6) == [
            script.one_run(1, None, 4, 6),
            script.one_run(2, None, 4, 6),
        ]


class TestTheReport:
    def test_it_has_a_row_for_each_setting_and_one_without(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_shortcut",
                "--runs",
                "1",
                "--before",
                "4",
                "--after",
                "6",
                "--kappas",
                "0.001",
                "0.01",
            ],
        )
        assert script.main() == 0

        printed = capsys.readouterr().out
        assert "dyna-q " in printed
        assert "dyna-q-plus, kappa 0.001" in printed
        assert "dyna-q-plus, kappa 0.01" in printed
        assert "dyna-q-plus, kappa 0.05" not in printed

    def test_the_heading_counts_the_episodes_it_ran(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # It said 60 and 120 whatever it was asked for, which is a number in
        # prose beside a number from a run, in a script about that.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_shortcut",
                "--runs",
                "1",
                "--before",
                "4",
                "--after",
                "6",
                "--kappas",
                "0.001",
            ],
        )
        script.main()

        printed = capsys.readouterr().out
        assert "4 episodes" in printed
        assert "6 after" in printed
        assert "over the last 6" in printed
