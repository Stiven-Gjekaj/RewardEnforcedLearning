"""The measurement behind the levels tables on the algorithms page.

The load bearing test is `TestTheLadderMeasuresTheCutAndNotTheAgent`. The
ladder is only about the number of levels if the agent walking it can learn
the problem at every count, and an agent that cannot would make the whole
table a picture of that agent instead.

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
from rel.envs.levels import Levels
from rel.envs.pendulum import Pendulum
from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_levels.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_levels", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestWhatItRuns:
    def test_the_ladder_agent_is_one_that_learns_this_problem(
        self, script: ModuleType
    ) -> None:
        assert script.LADDER_AGENT in AGENTS.names()
        assert "linear" in AGENTS[script.LADDER_AGENT].tags

    def test_every_agent_it_names_is_in_the_registry(self, script: ModuleType) -> None:
        for name in (*script.ON_A_LIST, script.ON_A_BOX):
            assert name in AGENTS.names(), name

    def test_the_one_on_the_box_is_the_one_that_takes_a_box(
        self, script: ModuleType
    ) -> None:
        assert "continuous-actions" in AGENTS[script.ON_A_BOX].tags

    def test_every_other_one_takes_a_list(self, script: ModuleType) -> None:
        for name in script.ON_A_LIST:
            assert "continuous-actions" not in AGENTS[name].tags, name

    def test_the_ladder_starts_at_a_switch(self, script: ModuleType) -> None:
        # Two levels is the whole point of the first row: it is the coarsest
        # cut there is and it is still enough to swing this problem up.
        assert min(script.LEVELS) == 2

    def test_the_middle_of_the_table_is_a_rung_of_the_ladder(
        self, script: ModuleType
    ) -> None:
        assert script.MIDDLE in script.LEVELS


class TestOneRun:
    def test_it_builds_the_cut_environment_it_was_asked_for(
        self, script: ModuleType
    ) -> None:
        env = script.levelled_pendulum(Rng(1).stream("env"), levels=4)
        assert isinstance(env, Levels)
        assert env.action_space.n == 4

    def test_a_run_on_a_cut_gives_a_return_below_zero(self, script: ModuleType) -> None:
        # Nothing in this environment pays anything above zero, so a positive
        # number would mean the run was not of this problem.
        assert script.on_levels("random", 3, seed=1, episodes=2) < 0.0

    def test_a_run_on_the_box_gives_a_return_below_zero(
        self, script: ModuleType
    ) -> None:
        assert script.on_the_box(seed=1, episodes=2) < 0.0

    def test_the_run_on_the_box_is_of_the_box(self, script: ModuleType) -> None:
        # It would be an easy and invisible mistake to run the continuous
        # agent on the cut version, and the two tables would then say the same
        # thing twice.
        source = SCRIPT.read_text()
        body = source[
            source.index("def on_the_box(") : source.index("def ladder_section(")
        ]
        assert "Pendulum(" in body
        assert "levelled" not in body

    def test_a_seed_replays_a_run(self, script: ModuleType) -> None:
        assert script.on_levels("tile-q", 3, seed=2, episodes=3) == script.on_levels(
            "tile-q", 3, seed=2, episodes=3
        )


class TestTheLadderMeasuresTheCutAndNotTheAgent:
    def test_the_agent_learns_the_coarsest_and_the_finest_cut(
        self, script: ModuleType
    ) -> None:
        """Both ends of the ladder, against doing nothing at all.

        A ladder walked by an agent that cannot learn one of its rungs is a
        picture of the agent. Sixty episodes is far short of the table's
        budget and is enough to see that both ends move.
        """
        for levels in (min(script.LEVELS), max(script.LEVELS)):
            learned = script.on_levels(script.LADDER_AGENT, levels, 1, 60)
            random = script.on_levels("random", levels, 1, 60)
            assert learned > random, levels

    def test_a_switch_is_enough_to_beat_doing_nothing(self, script: ModuleType) -> None:
        # Which is what makes the first row of the ladder worth reading. If
        # two levels could not learn the problem the ladder would only say
        # that a switch is too coarse.
        env = Pendulum(Rng(1).stream("env"))
        env.reset()
        nothing = sum(env.step((0.0,)).reward for _ in range(200))
        assert script.on_levels(script.LADDER_AGENT, 2, 1, 120) > nothing


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            ["measure_levels", "--runs", "1", "--episodes", "5", *extra],
        )
        return int(script.main())

    def test_both_tables_are_printed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch, "--levels", "2", "3") == 0
        printed = capsys.readouterr().out
        assert "levels" in printed
        assert "What learns this problem at all" in printed
        assert "gaussian-actor-critic" in printed
        assert "the box" in printed

    def test_the_levels_it_was_given_are_the_ones_it_walks(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--levels", "2", "4", "--middle", "4")
        ladder = capsys.readouterr().out.split("What learns")[0]
        rungs = [
            line.split()[0]
            for line in ladder.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        assert rungs == ["2", "4"]

    def test_the_best_row_of_each_column_is_worked_out(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The line under the ladder is read off the table rather than typed.

        The first reading of that table was from one budget, and the sentence
        under it said a finer cut is never better. Adding the second budget
        disagreed, and a sentence somebody typed does not notice.
        """
        self._run(script, monkeypatch, "--levels", "2", "3", "--middle", "3")
        ladder = capsys.readouterr().out.split("What learns")[0]
        said = [line for line in ladder.splitlines() if "Best at" in line]
        assert len(said) == 1
        assert said[0].split()[4] in {"2", "3"}

    def test_a_second_budget_is_a_second_column(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_levels",
                "--runs",
                "1",
                "--levels",
                "2",
                "--middle",
                "2",
                "--episodes",
                "4",
                "8",
            ],
        )
        script.main()
        printed = capsys.readouterr().out
        assert "4 episodes" in printed
        assert "8 episodes" in printed
