"""The measurement behind the two control problem tables on the algorithms page.

The load bearing tests are `TestTheGreedyReturnIsMeasuredElsewhere` and
`TestTheMeanIsTheMeanOfWhatItPrints`. The first is a design decision worth
holding: an agent is watched on an environment seeded differently from the one
it trained on, so the number reported is what it learned rather than what it
memorised of one stream. The second is the kind of thing nobody notices going
wrong, because a mean and a row of seeds beside it both look right on their
own.

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

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_control.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_control", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seeds_of(row: tuple[str, ...]) -> list[float]:
    """The per seed column of a row, back as numbers."""
    return [float(value) for value in row[2].split()]


class TestARow:
    def test_it_names_the_agent_it_ran(self, script: ModuleType) -> None:
        row = script.measure("random", "mountaincar", 1, 2, 1)
        assert row[0] == "random"

    def test_there_is_one_number_for_each_seed(self, script: ModuleType) -> None:
        # Every seed rather than a mean, because these agents vary far more
        # than the tabular ones and a mean on its own would hide it.
        assert len(seeds_of(script.measure("random", "mountaincar", 1, 3, 1))) == 3

    def test_a_seed_replays_it(self, script: ModuleType) -> None:
        first = script.measure("tile-sarsa", "mountaincar", 2, 1, 4)
        assert script.measure("tile-sarsa", "mountaincar", 2, 1, 4) == first

    def test_the_seeds_it_was_given_are_the_ones_it_runs(
        self, script: ModuleType
    ) -> None:
        # Two runs from seed 1 and one run from each of seeds 1 and 2 have to
        # agree, or the seed a row was made with is not the seed it names.
        both = seeds_of(script.measure("tile-sarsa", "mountaincar", 2, 2, 1))
        first = seeds_of(script.measure("tile-sarsa", "mountaincar", 2, 1, 1))
        second = seeds_of(script.measure("tile-sarsa", "mountaincar", 2, 1, 2))
        assert both == first + second

    def test_it_says_how_long_it_took(self, script: ModuleType) -> None:
        assert script.measure("random", "mountaincar", 1, 1, 1)[3].endswith("s")


class TestTheMeanIsTheMeanOfWhatItPrints:
    """A mean beside a row of seeds that it is not the mean of.

    Both columns look right on their own, so nothing about reading the table
    would find it. This is the test that would.
    """

    def test_they_agree(self, script: ModuleType) -> None:
        row = script.measure("tile-sarsa", "mountaincar", 2, 3, 1)
        # The seed column is rounded to whole numbers, so the mean of it is
        # within half a point of the mean the row reports.
        assert float(row[1]) == pytest.approx(statistics.mean(seeds_of(row)), abs=0.5)

    def test_one_seed_is_its_own_mean(self, script: ModuleType) -> None:
        row = script.measure("tile-sarsa", "mountaincar", 2, 1, 1)
        assert float(row[1]) == pytest.approx(seeds_of(row)[0], abs=0.5)


class TestTheGreedyReturnIsMeasuredElsewhere:
    """The agent is watched on an environment it did not train on.

    Both environments come from the same seed plus a fixed offset, so the run
    replays, but the stream the agent is watched over is not the stream it
    learned from. An agent measured on its own training stream would be
    reported for what it memorised of that stream.
    """

    def test_a_run_is_not_scored_on_the_stream_it_learned_from(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeds: list[int] = []
        real = script.Rng

        def watched(seed: int) -> object:
            seeds.append(seed)
            return real(seed)

        monkeypatch.setattr(script, "Rng", watched)
        script.measure("random", "mountaincar", 1, 1, 7)

        # The training seed and the watching seed, in the order the run asks
        # for them. They differ, which is the whole claim.
        assert seeds == [7, 507]

    def test_the_offset_holds_for_every_seed(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeds: list[int] = []
        real = script.Rng
        monkeypatch.setattr(
            script, "Rng", lambda seed: (seeds.append(seed), real(seed))[1]
        )
        script.measure("random", "mountaincar", 1, 3, 1)
        assert seeds == [1, 501, 2, 502, 3, 503]


class TestTheRandomPolicyNeverLeavesTheValley:
    """The clearest separation in the project between learning and not.

    The mountain car engine cannot climb the hill, so a policy that does not
    rock the car never reaches the flag at all and scores the step cap on
    every seed. A row of this that was not a thousand would mean the cap, the
    reward or the environment had changed under the table.
    """

    def test_every_seed_is_the_step_cap(self, script: ModuleType) -> None:
        row = script.measure("random", "mountaincar", 1, 3, 1)
        assert seeds_of(row) == [-1000.0, -1000.0, -1000.0]


class TestTheSettings:
    def test_they_reach_every_agent_measured(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `--set` is one setting for all of them, so an agent that did not
        # take it would be compared at a different setting than the heading
        # over the table says.
        seen: list[dict[str, object]] = []
        real = script.AGENTS

        class Watched:
            def make(
                self, name: str, rng: object, env: object, **options: object
            ) -> object:
                seen.append(dict(options))
                return real.make(name, rng, env, **options)

        monkeypatch.setattr(script, "AGENTS", Watched())
        script.measure("tile-sarsa", "mountaincar", 1, 2, 1, {"step_size": 0.25})
        assert seen == [{"step_size": 0.25}, {"step_size": 0.25}]

    def test_a_setting_changes_the_run(self, script: ModuleType) -> None:
        # The cart pole rather than the mountain car. A few episodes of the
        # mountain car never reach the flag at any step size, so both runs
        # would score the step cap and the test would pass on a script that
        # dropped the settings on the floor.
        plain = script.measure("tile-sarsa", "cartpole", 5, 1, 1)
        slowed = script.measure("tile-sarsa", "cartpole", 5, 1, 1, {"step_size": 0.01})
        assert seeds_of(plain) != seeds_of(slowed)


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            ["measure_control", "--episodes", "1", "--runs", "2", *extra],
        )
        return int(script.main())

    def test_it_runs_only_the_agents_it_was_given(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch, "--agents", "random") == 0
        printed = capsys.readouterr().out
        assert "random" in printed
        assert "tile-sarsa" not in printed

    def test_it_names_the_settings_every_agent_was_built_with(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A table made at a setting that the page does not say is a table a
        # reader cannot reproduce.
        self._run(
            script, monkeypatch, "--agents", "tile-sarsa", "--set", "step_size=0.25"
        )
        assert "step_size=0.25" in capsys.readouterr().out

    def test_it_names_the_seeds_it_ran(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--agents", "random", "--seed", "4")
        assert "seeds 4 to 5" in capsys.readouterr().out

    def test_every_column_has_a_heading(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--agents", "random")
        printed = capsys.readouterr().out
        for heading in ("agent", "greedy, mean", "each seed", "time"):
            assert heading in printed
