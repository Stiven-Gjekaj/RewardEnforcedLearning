"""The measurement behind the gambler tables on the algorithms page.

The load bearing test is `TestWhatCountsAsDecided`. The whole finding is that
a capital whose best two stakes are worth the same has no best stake, so the
count of capitals where one stake wins by more than the sweep's own tolerance
is the number the rest of the table is read against. A count that compared
actions rather than stakes would say a small capital had more to choose
between than it has, because several actions clip to the same stake there.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.agents.dp import value_iteration

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_gambler.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_gambler", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestWhatItRunsOver:
    def test_the_capitals_leave_out_both_endings(self, script: ModuleType) -> None:
        env = script.built(20, 0.4)
        assert list(script.playable(env)) == list(range(1, 20))

    def test_a_policy_is_read_as_stakes_rather_than_actions(
        self, script: ModuleType
    ) -> None:
        # Action seven stakes eight, and at a capital of three it stakes
        # three. A table of action numbers would be a table of an internal
        # encoding rather than of the problem.
        env = script.built(20, 0.4)
        policy = tuple([7] * 21)
        assert script.stakes(env, policy)[:4] == [1, 2, 3, 4]


class TestWhatEachStakeIsWorth:
    def test_it_is_keyed_by_the_stake_rather_than_the_action(
        self, script: ModuleType
    ) -> None:
        # At a capital of three only three stakes are possible, however many
        # actions clip onto them.
        env = script.built(100, 0.4)
        solved = value_iteration(env, discount=1.0)
        assert sorted(script.worths_at(env, solved.values, 3)) == [1, 2, 3]

    def test_winning_the_goal_is_worth_the_reward_rather_than_a_value(
        self, script: ModuleType
    ) -> None:
        # The goal is an ending, so its value is nothing. A worth that read it
        # off the value array would say that reaching the goal is worth zero.
        env = script.built(100, 0.4)
        solved = value_iteration(env, discount=1.0)
        worths = script.worths_at(env, solved.values, 50)
        assert worths[50] == pytest.approx(0.4)


class TestWhatCountsAsDecided:
    """A capital has a best stake only if one stake beats the rest.

    Beats them by more than the sweep's own tolerance, because a gap smaller
    than that is the sweep's arithmetic rather than the problem's answer.
    """

    def test_a_fair_coin_decides_nothing(self, script: ModuleType) -> None:
        # Every stake is exactly as good, so no capital has a best one.
        env = script.built(100, 0.5)
        solved = value_iteration(env, discount=1.0, tolerance=script.TIGHT)
        assert script.decided(env, solved.values, script.TIGHT) == 0

    def test_a_favourable_coin_decides_most_of_them(self, script: ModuleType) -> None:
        env = script.built(100, 0.55)
        solved = value_iteration(env, discount=1.0, tolerance=script.TIGHT)
        assert script.decided(env, solved.values, script.TIGHT) > 80

    def test_a_wide_margin_decides_nothing_anywhere(self, script: ModuleType) -> None:
        # The margin is what makes the count mean something. At a margin of
        # one no gap in a problem whose values run from zero to one can beat
        # it, and a count that ignored the margin would not notice.
        env = script.built(100, 0.55)
        solved = value_iteration(env, discount=1.0, tolerance=script.TIGHT)
        assert script.decided(env, solved.values, 1.0) == 0


class TestTheWidestGap:
    def test_a_fair_coin_leaves_almost_nothing_between_stakes(
        self, script: ModuleType
    ) -> None:
        # In arithmetic it is nothing at all. What a sweep reports is the size
        # of its own rounding, which is what makes the number worth printing.
        env = script.built(100, 0.5)
        solved = value_iteration(env, discount=1.0, tolerance=script.TIGHT)
        assert script.widest(env, solved.values) < 1e-9

    def test_a_biased_coin_leaves_a_real_gap(self, script: ModuleType) -> None:
        env = script.built(100, 0.4)
        solved = value_iteration(env, discount=1.0, tolerance=script.TIGHT)
        assert script.widest(env, solved.values) > 0.01


class TestTwoPoliciesApart:
    def test_the_same_policy_is_nowhere_apart(self, script: ModuleType) -> None:
        assert script.apart([1, 2, 3], [1, 2, 3]) == 0

    def test_it_counts_the_places_rather_than_the_size_of_the_difference(
        self, script: ModuleType
    ) -> None:
        assert script.apart([1, 2, 3], [1, 40, 3]) == 1


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(sys, "argv", ["measure_gambler", "--goal", "20", *extra])
        return int(script.main())

    def test_the_closed_form_is_printed_beside_the_sweep(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A reference nothing checks is a reference nobody should trust.
        assert self._run(script, monkeypatch, "--coins", "0.4") == 0
        printed = capsys.readouterr().out
        assert "closed form" in printed
        assert "swept" in printed
        assert "Worst gap" in printed

    def test_the_fair_coin_is_run_whether_or_not_it_was_asked_for(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # It is the control. Without a row that is known to be undetermined,
        # every other row is read against nothing.
        self._run(script, monkeypatch, "--coins", "0.4")
        rows = [
            line.split()[0]
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("0.")
        ]
        assert "0.5" in rows

    def test_every_coin_it_was_given_gets_a_row(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--coins", "0.3", "0.4")
        printed = capsys.readouterr().out
        for coin in ("0.3", "0.4", "0.5"):
            assert f"  {coin} " in printed or f" {coin} " in printed

    def test_the_columns_say_what_moved_and_what_did_not(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--coins", "0.4")
        printed = capsys.readouterr().out
        for heading in (
            "widest gap between stakes",
            "capitals with one best stake",
            "moved by the tolerance",
            "moved by the solver",
        ):
            assert heading in printed
