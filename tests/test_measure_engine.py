"""The measurement behind the engine table on the algorithms page.

The load bearing test is `TestTheDigestsAreThePoint`. The timings in that
table say what moved; the digests say that nothing else did. An optimisation
that reassociates a sum is faster and gives different numbers, and on a
learning agent different numbers look exactly like the same agent on another
seed. So a digest that did not depend on the arithmetic would let a change be
reported as a speedup when it was a change of behaviour.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_engine.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_engine", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestARun:
    def test_it_reports_seconds_and_two_digests(self, script: ModuleType) -> None:
        seconds, path, learned = script.one_run("cartpole", "reinforce", 2, 1)
        assert seconds > 0.0
        assert len(path) == 16
        assert learned != ""

    def test_an_agent_that_keeps_no_weights_says_so(self, script: ModuleType) -> None:
        # A dash rather than an empty cell, because an empty cell in a table
        # of digests reads as a digest that failed to print.
        _, _, learned = script.one_run("cliff", "random", 2, 1)
        assert learned == "-"


class TestTheDigestsAreThePoint:
    """They have to move with the arithmetic and not with the clock."""

    def test_the_same_run_gives_the_same_digests(self, script: ModuleType) -> None:
        first = script.one_run("cartpole", "reinforce", 3, 1)[1:]
        assert script.one_run("cartpole", "reinforce", 3, 1)[1:] == first

    def test_another_seed_gives_other_digests(self, script: ModuleType) -> None:
        first = script.one_run("cartpole", "reinforce", 3, 1)[1:]
        assert script.one_run("cartpole", "reinforce", 3, 2)[1:] != first

    def test_more_episodes_give_other_digests(self, script: ModuleType) -> None:
        # Which is what says the digest is of the run rather than of the
        # agent's settings. A digest that ignored the episodes would match
        # before and after a change that only shortened them.
        first = script.one_run("cartpole", "reinforce", 3, 1)[1:]
        assert script.one_run("cartpole", "reinforce", 6, 1)[1:] != first

    def test_the_two_digests_are_not_the_same_number(self, script: ModuleType) -> None:
        # One hashes the path taken and one what the weights ended at. Two
        # columns that always agreed would be one column.
        _, path, learned = script.one_run("cartpole", "reinforce", 3, 1)
        assert path != learned


class TestOneLayer:
    def test_it_times_the_two_passes_apart(self, script: ModuleType) -> None:
        forward, backward = script.one_layer(4, 16, 200)
        assert forward > 0.0
        assert backward > 0.0

    def test_a_wider_layer_takes_longer(self, script: ModuleType) -> None:
        # The shape of the answer rather than the numbers on the page. A
        # timing that did not grow with the layer would not be timing it.
        narrow, _ = script.one_layer(4, 16, 4000)
        wide, _ = script.one_layer(48, 16, 4000)
        assert wide > narrow

    def test_it_takes_the_best_of_several(self, script: ModuleType) -> None:
        """Rather than one timing, because one timing does not settle.

        The first version of this measurement timed each shape once and its
        answer for unchanged code moved by fifteen percent depending on what
        had run before it in the same process. The best of several throws
        that away rather than averaging it in, so more repeats can only
        report a time that is the same or smaller.
        """
        once, _ = script.one_layer(4, 16, 2000, repeats=1)
        often, _ = script.one_layer(4, 16, 2000, repeats=6)
        assert often <= once * 1.5


class TestTheOptimiser:
    def test_it_counts_the_numbers_it_stepped(self, script: ModuleType) -> None:
        # Four by 48 weights and 4 biases, then 16 by 4 and 16, worked out
        # here so a change to the shape fails rather than quietly rescales
        # the line in the table.
        _, numbers = script.one_optimiser(48, 16, 4, 10)
        assert numbers == 48 * 16 + 16 + 16 * 4 + 4

    def test_more_steps_take_longer(self, script: ModuleType) -> None:
        few, _ = script.one_optimiser(48, 16, 4, 50)
        many, _ = script.one_optimiser(48, 16, 4, 500)
        assert many > few

    def test_it_is_timed_apart_from_the_run(self, script: ModuleType) -> None:
        """Its own line because a whole run cannot see it.

        The optimiser is about a tenth of a `deep-q` run, so a change that
        makes it a fifth faster moves the run by two percent, and two percent
        is smaller than the run to run spread of the timing above it.
        """
        seconds, _ = script.one_optimiser(48, 16, 4, 100)
        assert seconds > 0.0


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_engine",
                "--episodes",
                "2",
                "--repeats",
                "2",
                "--passes",
                "200",
                *extra,
            ],
        )
        return int(script.main())

    def test_it_prints_a_line_for_every_shape_it_timed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch) == 0
        printed = capsys.readouterr().out
        for line in (
            "a whole run, best",
            "a whole run, median",
            "4 to 16 forward",
            "4 to 16 backward",
            "48 to 16 forward",
            "48 to 16 backward",
            "Adam over",
        ):
            assert line in printed

    def test_it_prints_both_digests(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch)
        printed = capsys.readouterr().out
        assert "the path" in printed
        assert "what it learned" in printed

    def test_it_says_what_a_matching_digest_means(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The instruction is the point of the script, so it is printed rather
        # than left in the docstring where a reader of the output would not
        # see it.
        self._run(script, monkeypatch)
        assert "Matching digests" in capsys.readouterr().out

    def test_the_agent_and_the_environment_can_be_changed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--agent", "deep-q", "--env", "cliff")
        assert "deep-q on cliff" in capsys.readouterr().out

    def test_the_best_is_never_above_the_median(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Both are printed, so a reader can see how far apart they are. The
        # one that could go wrong silently is the order of them.
        self._run(script, monkeypatch)
        printed = capsys.readouterr().out
        best = float(printed.split("a whole run, best")[1].split("s")[0])
        median = float(printed.split("a whole run, median")[1].split("s")[0])
        assert best <= median
