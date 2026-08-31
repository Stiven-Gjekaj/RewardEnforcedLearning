"""The measurement behind the intra-option table on the algorithms page.

The load bearing test is `TestWithoutOptionsBothAgentsAreOneAgent`. The last
two rows of the table run with `hallways=off`, where every option is a single
primitive action that stops after one step, so both agents reduce to
Q-learning and must give the same numbers. Those rows are the control: if they
ever disagree, the difference the other rows report is not the crediting rule
alone.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_intra_option.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_intra_option", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheCurve:
    def test_there_is_one_point_for_each_block_of_episodes(
        self, script: ModuleType
    ) -> None:
        curve, _, _, _ = script.measure("options-q", "rooms", 20, 1, 5, True)
        assert len(curve) == 4

    def test_a_block_that_does_not_divide_the_episodes_drops_the_remainder(
        self, script: ModuleType
    ) -> None:
        # Twenty two episodes in blocks of five is four blocks and two
        # episodes nobody sees. Written down rather than left to be found,
        # because a heading that says 21-25 over a block of two would read as
        # a block of five.
        curve, _, _, _ = script.measure("options-q", "rooms", 22, 1, 5, True)
        assert len(curve) == 4

    def test_the_points_are_episode_lengths_rather_than_returns(
        self, script: ModuleType
    ) -> None:
        # Lengths are positive and returns on this grid are negative, so the
        # sign alone says which column the table is showing.
        curve, _, _, _ = script.measure("options-q", "rooms", 20, 1, 10, True)
        assert all(point > 0.0 for point in curve)


class TestTheUpdatesPerStep:
    def test_crediting_every_option_costs_more_updates(
        self, script: ModuleType
    ) -> None:
        # The cost side of the table. One update per step is what waiting for
        # an option to stop gives, and crediting every option that agreed
        # with a step gives more.
        _, waiting, _, _ = script.measure("options-q", "rooms", 20, 2, 10, True)
        _, crediting, _, _ = script.measure("intra-option-q", "rooms", 20, 2, 10, True)
        assert crediting > waiting

    def test_waiting_for_an_option_gives_fewer_updates_than_steps(
        self, script: ModuleType
    ) -> None:
        # Below one, which is the thing worth knowing about this column and
        # the opposite of what its name suggests. An option that runs for
        # three steps produces one update, so the agent that waits for one to
        # stop learns less often than it moves.
        _, per_step, _, _ = script.measure("options-q", "rooms", 20, 1, 10, True)
        assert per_step < 1.0

    def test_crediting_every_option_gives_at_least_one(
        self, script: ModuleType
    ) -> None:
        _, per_step, _, _ = script.measure("intra-option-q", "rooms", 20, 1, 10, True)
        assert per_step >= 1.0

    def test_without_options_neither_can_be_below_one(self, script: ModuleType) -> None:
        # A primitive option stops after every step, so every step is one
        # update whichever agent is running.
        for name in script.AGENT_NAMES:
            _, per_step, _, _ = script.measure(name, "rooms", 20, 1, 10, False)
            assert per_step == 1.0, name


class TestWithoutOptionsBothAgentsAreOneAgent:
    """The control rows, and the reason to trust the rest of the table.

    A primitive option stops after every step, so there is never a state
    passed through inside one and the two crediting rules have nothing to
    disagree about. Both agents are Q-learning there.
    """

    def test_the_curves_are_the_same(self, script: ModuleType) -> None:
        waiting = script.measure("options-q", "rooms", 30, 2, 10, False)
        crediting = script.measure("intra-option-q", "rooms", 30, 2, 10, False)
        assert waiting[0] == crediting[0]

    def test_the_updates_per_step_are_the_same(self, script: ModuleType) -> None:
        waiting = script.measure("options-q", "rooms", 30, 2, 10, False)
        crediting = script.measure("intra-option-q", "rooms", 30, 2, 10, False)
        assert waiting[1] == crediting[1] == 1.0

    def test_they_really_do_disagree_once_the_options_are_back(
        self, script: ModuleType
    ) -> None:
        # Without this the class above would pass on a script that ran the
        # same agent twice.
        waiting = script.measure("options-q", "rooms", 30, 2, 10, True)
        crediting = script.measure("intra-option-q", "rooms", 30, 2, 10, True)
        assert waiting[0] != crediting[0]


class TestTheStuckCount:
    def test_it_counts_seeds_whose_greedy_policy_never_reaches_the_end(
        self, script: ModuleType
    ) -> None:
        # A count rather than a share, and blank rather than zero in the
        # table, because a column of zeros is a column a reader stops seeing.
        _, _, _, stuck = script.measure("options-q", "rooms", 20, 3, 10, True)
        assert 0 <= stuck <= 3

    def test_a_seed_replays_it(self, script: ModuleType) -> None:
        first = script.measure("options-q", "rooms", 20, 2, 10, True)
        assert script.measure("options-q", "rooms", 20, 2, 10, True) == first


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_intra_option",
                "--episodes",
                "20",
                "--runs",
                "1",
                "--block",
                "10",
                *extra,
            ],
        )
        return int(script.main())

    def test_all_four_rows_are_printed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch) == 0
        printed = capsys.readouterr().out
        for name in script.AGENT_NAMES:
            assert f"{name}, no options" in printed
        # Counted by line, because "intra-option-q" does not carry
        # "options-q" inside it and a count of either name alone would say
        # two where the table has four rows.
        rows = [
            line
            for line in printed.splitlines()
            if any(line.strip().startswith(name) for name in script.AGENT_NAMES)
        ]
        assert len(rows) == 4

    def test_the_headings_name_the_blocks_they_cover(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch)
        printed = capsys.readouterr().out
        assert "1-10" in printed
        assert "11-20" in printed

    def test_it_says_what_the_best_possible_return_is(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Solved rather than remembered, so a curve can be read against
        # something rather than only against the other curve.
        self._run(script, monkeypatch)
        assert "best possible return" in capsys.readouterr().out
