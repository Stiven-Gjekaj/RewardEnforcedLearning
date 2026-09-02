"""The measurement behind the sigma table on the algorithms page.

The load bearing test is `TestEverySigmaIsSweptAndComparedAgainstTreeBackup`.
Sigma changes how large the target is as well as what it is made of, so a
comparison at one step size would find which sigma suits that step size and
report it as which sigma is better. What is checked is that every sigma really
is swept over the same step sizes and read at its own best.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_sigma.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_sigma", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestWhatItMeasuresAgainst:
    def test_the_best_possible_return_is_the_grids_own(
        self, script: ModuleType
    ) -> None:
        reachable, discount = script.best_possible("cliff")
        assert reachable == pytest.approx(-13.0)
        assert discount == 1.0

    def test_it_is_read_once(self, script: ModuleType) -> None:
        assert script.best_possible("cliff") is script.best_possible("cliff")

    def test_another_grid_gives_another_answer(self, script: ModuleType) -> None:
        assert script.best_possible("maze") != script.best_possible("cliff")


class TestARun:
    def test_it_scores_the_policy_exactly_rather_than_the_return(
        self, script: ModuleType
    ) -> None:
        # A run that collected -60 while learning can still have learned a
        # policy worth -15, and it is the policy the table is about.
        got = script.one_run("cliff", 0.0, 0.2, 200, 1)
        assert -20.0 <= got <= -13.0

    def test_a_seed_replays_it(self, script: ModuleType) -> None:
        assert script.one_run("cliff", 0.5, 0.2, 40, 2) == script.one_run(
            "cliff", 0.5, 0.2, 40, 2
        )

    def test_the_sigma_it_was_given_reaches_the_agent(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checked at the agent rather than through the score.

        The exact value of a cliff walk policy takes very few numbers: -13
        for the edge, -15 for one row up, -17 for two. Two sigmas that learn
        different policies often land on the same one of those, so a test
        that asked whether two runs scored differently would fail on a script
        that ignored sigma altogether.
        """
        seen: list[object] = []
        real = script.AGENTS

        class Watched:
            def make(
                self, name: str, rng: object, env: object, **options: object
            ) -> object:
                seen.append(options["sigma"])
                return real.make(name, rng, env, **options)

        monkeypatch.setattr(script, "AGENTS", Watched())
        script.one_run("cliff", 0.75, 0.2, 2, 1)
        assert seen == [0.75]

    def test_a_policy_that_never_finishes_scores_the_cap(
        self, script: ModuleType
    ) -> None:
        """Rather than minus infinity, which would swallow a whole row.

        A mean over ten seeds with one infinity in it is an infinity, and the
        row would then say nothing about the other nine.
        """
        from rel.envs import ENVIRONMENTS

        env = ENVIRONMENTS.make("cliff", Rng(1).stream("env"))
        got = script.one_run("cliff", 0.0, 0.4, 0, 1)
        assert got == -float(env.spec.max_episode_steps)


class TestEverySigmaIsSweptAndComparedAgainstTreeBackup:
    """Both halves of the comparison, checked rather than described."""

    def test_every_sigma_sees_every_step_size(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[tuple[object, float]] = []

        def record(
            grid: str,
            sigma: object,
            step: float,
            episodes: int,
            seed: int,
            window: int = 3,
        ) -> float:
            asked.append((sigma, step))
            return -15.0

        monkeypatch.setattr(script, "one_run", record)
        script.sigma_section(
            "cliff", (0.0, 1.0), (0.1, 0.2), 10, 1, Rng(1).stream("compare")
        )

        for sigma in (0.0, 1.0):
            steps = sorted(step for asked_sigma, step in asked if asked_sigma == sigma)
            assert steps == [0.1, 0.2], sigma

    def test_the_best_step_is_the_one_with_the_best_mean(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: -100.0 * step,
        )
        got, at_step = script.best_of("cliff", 0.5, (0.1, 0.5), 10, 2)
        assert at_step == 0.1
        assert got == [-10.0, -10.0]

    def test_tree_backup_is_run_whether_or_not_it_was_asked_for(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # It is the row every other row is read against.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_sigma",
                "--sigmas",
                "1",
                "--step-sizes",
                "0.2",
                "--episodes",
                "20",
                "--runs",
                "2",
            ],
        )
        assert script.main() == 0
        rows = [
            line.split()[0]
            for line in capsys.readouterr().out.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        assert "0" in rows

    def test_tree_backup_is_compared_against_nothing(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A row against itself is a difference of zero with an interval of
        # zero, which reads as a finding. A dash does not.
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: -15.0 - seed,
        )
        script.sigma_section(
            "cliff", (0.0, 1.0), (0.2,), 10, 3, Rng(1).stream("compare")
        )
        row = next(
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("0 ")
        )
        assert row.count("-") >= 3

    def test_it_is_run_once_rather_than_twice(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Tree backup is both a row and the thing every row is measured
        # against, and running it twice would double the slowest part of the
        # sweep for nothing.
        seen: list[object] = []
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: (
                seen.append(sigma),
                -15.0,
            )[1],
        )
        script.sigma_section("cliff", (0.0,), (0.2,), 10, 1, Rng(1).stream("compare"))
        assert seen.count(0.0) == 1


class TestTheSchedule:
    def test_it_starts_at_one_and_ends_at_nothing(self, script: ModuleType) -> None:
        # The book's suggestion: sample early and average late.
        falling = script.falling(100)
        assert falling(0) == pytest.approx(1.0)
        assert falling(100 * script.STEPS_AN_EPISODE) == pytest.approx(0.0)

    def test_it_arrives_before_the_run_ends(self, script: ModuleType) -> None:
        """On purpose, and the docstring says so.

        The schedule is written in episodes and read on steps, and a cliff
        walk episode is longer than the estimate. So sigma reaches nothing
        about half way and the rest of the run is tree backup, which is the
        suggestion rather than a slower sigma of a half.
        """
        falling = script.falling(400)
        assert falling(14948) == 0.0

    def test_it_falls_in_a_straight_line(self, script: ModuleType) -> None:
        falling = script.falling(100)
        assert falling(1000) == pytest.approx(0.5)

    def test_it_gets_a_row_of_its_own(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: -15.0,
        )
        script.sigma_section("cliff", (0.0,), (0.2,), 10, 2, Rng(1).stream("compare"))
        assert "1 falling to 0" in capsys.readouterr().out

    def test_the_row_really_runs_a_schedule(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[object] = []
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: (
                seen.append(sigma),
                -15.0,
            )[1],
        )
        script.sigma_section("cliff", (0.0,), (0.2,), 10, 1, Rng(1).stream("compare"))
        assert any(callable(sigma) for sigma in seen)


class TestTheReport:
    def test_it_says_what_the_best_possible_return_is(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_sigma",
                "--sigmas",
                "0",
                "--step-sizes",
                "0.2",
                "--episodes",
                "20",
                "--runs",
                "2",
            ],
        )
        script.main()
        printed = capsys.readouterr().out
        assert "-13.000" in printed
        assert "own best" in printed

    def test_it_says_when_the_seeds_cannot_reach_a_small_p(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: -15.0,
        )
        script.sigma_section(
            "cliff", (0.0, 1.0), (0.2,), 10, 3, Rng(1).stream("compare")
        )
        assert "cannot give a p below" in capsys.readouterr().out


class TestItRunsMoreThanOneWindow:
    def test_the_default_is_the_one_the_page_documents(
        self, script: ModuleType
    ) -> None:
        assert script.STEPS == 3

    def test_the_window_it_was_given_reaches_the_agent(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # At the agent rather than through the score, for the reason given
        # above about how few numbers a cliff walk policy is worth.
        seen: list[object] = []
        real = script.AGENTS

        class Watched:
            def make(
                self, name: str, rng: object, env: object, **options: object
            ) -> object:
                seen.append(options["n"])
                return real.make(name, rng, env, **options)

        monkeypatch.setattr(script, "AGENTS", Watched())
        script.one_run("cliff", 0.0, 0.1, 3, 1, 7)
        assert seen == [7]

    def test_the_sweep_carries_the_window_through(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[object] = []
        real = script.AGENTS

        class Watched:
            def make(
                self, name: str, rng: object, env: object, **options: object
            ) -> object:
                seen.append(options["n"])
                return real.make(name, rng, env, **options)

        monkeypatch.setattr(script, "AGENTS", Watched())
        script.best_of("cliff", 0.0, (0.1,), 3, 2, 10)
        assert seen == [10, 10]

    def test_the_heading_names_the_window_it_ran(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.sigma_section("cliff", (0.0,), (0.1,), 3, 1, Rng(1), 9)
        assert "n of 9" in capsys.readouterr().out

    def test_the_default_window_is_still_the_documented_one(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.sigma_section("cliff", (0.0,), (0.1,), 3, 1, Rng(1))
        assert "n of 3" in capsys.readouterr().out


class TestTheSweepSaysWhenItDidNotBracketARow:
    """The warning `measure_fourier.py` has had from the start.

    A row read at an end of the swept step sizes might do better outside the
    range, so its number is a floor rather than a best. This script did not
    say so, and five of the six rows of the table on the algorithms page chose
    the smallest step size offered, which is exactly the case it is for.
    """

    def test_an_end_of_the_sweep_is_an_end(self, script: ModuleType) -> None:
        assert script.at_an_end(0.05, (0.05, 0.1, 0.4))
        assert script.at_an_end(0.4, (0.05, 0.1, 0.4))
        assert not script.at_an_end(0.1, (0.05, 0.1, 0.4))

    def test_one_step_swept_is_both_ends_at_once(self, script: ModuleType) -> None:
        assert script.at_an_end(0.2, (0.2,))

    def test_it_names_the_rows_it_did_not_bracket(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Every row wants the largest step here, which is an end.
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: -1000.0 / step,
        )
        script.sigma_section("cliff", (0.0, 1.0), (0.1, 0.5), 10, 2, Rng(1))
        printed = capsys.readouterr().out
        assert "did not" in printed
        assert "sigma 0, at 0.5" in printed
        assert "sigma 1, at 0.5" in printed

    def test_it_says_nothing_when_every_row_is_bracketed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            script,
            "one_run",
            lambda grid, sigma, step, episodes, seed, window=3: (
                -abs(step - 0.5) * 100.0
            ),
        )
        script.sigma_section("cliff", (0.0, 1.0), (0.1, 0.5, 0.9), 10, 2, Rng(1))
        assert "did not" not in capsys.readouterr().out


class TestTheSweepBracketsItsOwnAnswer:
    """The step sizes offered have to reach past what any row wants.

    A row whose best sits at an end of the sweep is read at the edge of what
    it was allowed rather than at a best, and the number is then a statement
    about the sweep. The first version offered 0.05 to 0.4 and five of six
    rows at a window of three chose 0.05, which moved every table when the
    range was opened.

    `at_an_end` is what reports that, and these hold the default against it
    rather than against a remembered list.
    """

    def test_the_sweep_reaches_far_below_the_step_size_a_row_once_wanted(
        self, script: ModuleType
    ) -> None:
        assert min(script.STEP_SIZES) < 0.05

    def test_the_sweep_still_reaches_the_largest_step_it_did(
        self, script: ModuleType
    ) -> None:
        assert max(script.STEP_SIZES) == 0.4

    def test_every_step_size_is_offered_once(self, script: ModuleType) -> None:
        assert len(set(script.STEP_SIZES)) == len(script.STEP_SIZES)

    def test_the_step_sizes_rise(self, script: ModuleType) -> None:
        assert list(script.STEP_SIZES) == sorted(script.STEP_SIZES)

    def test_each_step_is_twice_the_one_below_it(self, script: ModuleType) -> None:
        # A ladder rather than a list, so a row that wants something between
        # two of them is at most a factor of two from what it got.
        for smaller, larger in itertools.pairwise(script.STEP_SIZES):
            assert larger == pytest.approx(smaller * 2.0)
