"""The measurement behind the Fourier basis table on the algorithms page.

The load bearing test is `TestBothSidesAreSwept`. Every scale a Fourier basis
asks for is at most one, so scaling makes each step smaller as well as uneven.
A comparison at one step size would find "smaller steps are better here" and
report it as "uneven steps are better here", and the only thing that stops
that is sweeping both sides over the same step sizes. So what is checked is
that both sides really are swept, rather than that the table has two columns.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_fourier.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_fourier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def table_of(printed: str) -> list[str]:
    """The lines of the table, without the notes that follow it.

    Split on blank lines rather than read from the top, because the notes
    under the table start with a count of seeds and a test that took every
    line beginning with a digit would read that count as a row.
    """
    for block in printed.split("\n\n"):
        if "features" in block:
            return block.splitlines()
    raise AssertionError("nothing printed looks like the table")


class TestHowManyFeaturesItReports:
    def test_it_is_the_order_plus_one_to_the_dimensions(
        self, script: ModuleType
    ) -> None:
        # The mountain car is two dimensions, so the count is a square.
        assert script.features_of(0) == 1
        assert script.features_of(1) == 4
        assert script.features_of(3) == 16
        assert script.features_of(7) == 64


class TestARun:
    def test_there_is_one_score_for_each_seed(self, script: ModuleType) -> None:
        # Every seed rather than their mean, because the two sides of the
        # comparison are paired by seed and a mean cannot be paired.
        assert len(script.one_setting(1, True, 0.5, runs=3, episodes=2)) == 3

    def test_a_seed_replays_it(self, script: ModuleType) -> None:
        first = script.one_setting(1, True, 0.5, runs=1, episodes=5)
        assert script.one_setting(1, True, 0.5, runs=1, episodes=5) == first

    def test_scaling_the_steps_gives_a_different_run(self, script: ModuleType) -> None:
        scaled = script.one_setting(3, True, 0.5, runs=1, episodes=5)
        flat = script.one_setting(3, False, 0.5, runs=1, episodes=5)
        assert scaled != flat

    def test_order_zero_is_a_constant_and_cannot_steer(
        self, script: ModuleType
    ) -> None:
        # One feature for the whole box means one number for each action and
        # nothing that depends on where the car is, so it never gets out.
        assert script.one_setting(0, True, 0.5, runs=1, episodes=5) == [-1000.0]


class TestBestOf:
    def test_it_returns_every_seed_of_the_best_and_the_step_that_got_it(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            script,
            "one_setting",
            lambda order, scaled, step, runs, episodes: {
                0.1: [-300.0, -300.0],
                0.5: [-100.0, -140.0],
                1.0: [-400.0, -400.0],
            }[step],
        )
        assert script.best_of(3, True, (0.1, 0.5, 1.0), 2, 5) == (
            [-100.0, -140.0],
            0.5,
        )

    def test_best_is_the_largest_return_rather_than_the_smallest(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Returns here are negative, so the best one is the largest. A sweep
        # that took the smallest would report the worst setting of each side
        # and the table would be upside down.
        monkeypatch.setattr(
            script,
            "one_setting",
            lambda order, scaled, step, runs, episodes: [-step * 1000.0],
        )
        assert script.best_of(3, True, (0.2, 0.9), 1, 5) == ([-200.0], 0.2)

    def test_it_compares_the_means_rather_than_any_one_seed(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The second setting has the best single seed and the worse mean. A
        # sweep that ranked on one seed would pick it.
        monkeypatch.setattr(
            script,
            "one_setting",
            lambda order, scaled, step, runs, episodes: {
                0.1: [-150.0, -150.0],
                0.5: [-10.0, -400.0],
            }[step],
        )
        assert script.best_of(3, True, (0.1, 0.5), 2, 5)[1] == 0.1


class TestBothSidesAreSwept:
    def test_each_side_sees_every_step_size(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[tuple[int, bool, float]] = []

        def record(
            order: int, scaled: bool, step: float, runs: int, episodes: int
        ) -> list[float]:
            asked.append((order, scaled, step))
            return [-100.0, -110.0]

        monkeypatch.setattr(script, "one_setting", record)
        script.scales_section((3,), (0.1, 0.5, 1.0), 1, 5, Rng(1).stream("compare"))

        assert sorted(step for order, scaled, step in asked if scaled) == [
            0.1,
            0.5,
            1.0,
        ]
        assert sorted(step for order, scaled, step in asked if not scaled) == [
            0.1,
            0.5,
            1.0,
        ]

    def test_the_two_sides_differ_only_in_the_scaling(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[int, bool, float, int, int]] = []
        monkeypatch.setattr(
            script,
            "one_setting",
            lambda *asked: (seen.append(asked), [-100.0, -110.0])[1],
        )
        script.scales_section((5,), (0.2,), 3, 7, Rng(1).stream("compare"))

        assert len(seen) == 2
        first, second = seen
        assert first[1] is not second[1]
        assert first[0] == second[0]
        assert first[2:] == second[2:]


class TestTheSweepSaysWhenItDidNotBracketASide:
    """Reading a side at its own best is fair only if the best is in range.

    A side whose best sits at an end of the swept step sizes might do better
    outside it, and the row then compares one side at its best against the
    other at the edge of what it was allowed. That is worth saying rather
    than leaving for a reader to notice.
    """

    def test_an_end_of_the_sweep_is_an_end(self, script: ModuleType) -> None:
        steps = (0.1, 0.5, 1.0)
        assert script.at_an_end(0.1, steps)
        assert script.at_an_end(1.0, steps)
        assert not script.at_an_end(0.5, steps)

    def test_one_step_swept_is_both_ends_at_once(self, script: ModuleType) -> None:
        assert script.at_an_end(0.5, (0.5,))

    def test_the_report_names_the_rows_it_did_not_bracket(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            script,
            "one_setting",
            lambda order, scaled, step, runs, episodes: [-1000.0 * step] * 3,
        )
        script.scales_section((3,), (0.1, 0.5, 1.0), 1, 5, Rng(1).stream("compare"))
        printed = capsys.readouterr().out
        assert "did not" in printed
        assert "order 3, scaled, at 0.1" in printed
        assert "order 3, flat, at 0.1" in printed

    def test_it_says_nothing_when_both_sides_are_bracketed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            script,
            "one_setting",
            lambda order, scaled, step, runs, episodes: [-abs(step - 0.5) * 1000.0] * 3,
        )
        script.scales_section((3,), (0.1, 0.5, 1.0), 1, 5, Rng(1).stream("compare"))
        assert "bracket" not in capsys.readouterr().out


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_fourier",
                "--runs",
                "1",
                "--episodes",
                "3",
                "--orders",
                "1",
                "--step-sizes",
                "0.5",
                *extra,
            ],
        )
        return int(script.main())

    def test_the_table_names_both_sides_and_the_gap(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch) == 0
        printed = capsys.readouterr().out
        assert "scaled" in printed
        assert "flat" in printed
        assert "scaled minus flat" in printed
        assert "95 percent interval" in printed

    def test_it_says_that_both_sides_were_swept(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A reader who does not know that cannot tell the finding from the
        # smaller steps that scaling also brings.
        self._run(script, monkeypatch)
        assert "own best" in capsys.readouterr().out

    def test_the_orders_it_was_given_are_the_ones_it_runs(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--orders", "1", "2")
        rows = [
            line.split()[0]
            for line in table_of(capsys.readouterr().out)
            if line.strip() and line.strip()[0].isdigit()
        ]
        assert rows == ["1", "2"]
