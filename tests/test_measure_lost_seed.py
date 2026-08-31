"""The measurement behind the lost seed tables on the algorithms page.

The load bearing test is `TestTheHoldoutIsTheSeedThatHeldOutLongest`. The
second table exists to say what one seed needs, and it picks that seed from
the first table rather than naming one. A seed that never reached the goal at
all has no episode number to sort on, so the pick has to put it above every
seed that did, and a sort that read a missing number as zero would pick the
fastest seed and report the ladder for the wrong one.

The script is slow. A policy gradient is hundreds of times slower than
Q-learning on the same grid, so every test here runs a handful of episodes and
checks the shape of the answer rather than the numbers on the page.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_lost_seed.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_lost_seed", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestARun:
    def test_the_goal_is_numbered_from_one(self, script: ModuleType) -> None:
        # The table reads as "not there yet after 500 episodes", so an episode
        # counted from zero would be off by one against every budget in it.
        got = script.run(seed=1, episodes=60, entropy=None)
        assert got.first_goal is None or got.first_goal >= 1

    def test_a_seed_that_never_arrives_says_so_rather_than_zero(
        self, script: ModuleType
    ) -> None:
        # Two episodes is not enough for any seed, and the difference between
        # "never" and "episode zero" is the whole of the first table.
        assert script.run(seed=1, episodes=2, entropy=None).first_goal is None

    def test_a_seed_replays_it(self, script: ModuleType) -> None:
        assert script.run(1, 20, None) == script.run(1, 20, None)

    def test_two_seeds_do_not_give_the_same_run(self, script: ModuleType) -> None:
        assert script.run(1, 20, None) != script.run(2, 20, None)

    def test_the_entropy_reaches_the_agent(self, script: ModuleType) -> None:
        assert script.run(1, 20, 0.4) != script.run(1, 20, None)

    def test_a_policy_that_never_finishes_has_no_exact_value(
        self, script: ModuleType
    ) -> None:
        # An early policy on the cliff walk usually walks into a wall for
        # ever. A number there would be a value read off a policy that has no
        # value, which is worse than a dash.
        got = script.run(seed=1, episodes=2, entropy=None)
        assert got.exact is None or got.exact <= 0.0


class TestTheHoldoutIsTheSeedThatHeldOutLongest:
    """Which seed the second table is about, worked out rather than named.

    A seed that never arrived has no episode number, so the sort has to rank
    it above every seed that did. Reading a missing number as zero would rank
    it below all of them and the ladder would report the fastest seed.
    """

    @staticmethod
    def _pick(first_goal: dict[int, int | None]) -> int:
        # The expression the script sorts on, kept here so a change to it
        # fails this test rather than quietly picking another seed.
        return max(
            first_goal,
            key=lambda seed: (first_goal[seed] is None, first_goal[seed] or 0),
        )

    def test_the_latest_arrival_wins(self) -> None:
        assert self._pick({1: 10, 2: 400, 3: 90}) == 2

    def test_a_seed_that_never_arrived_beats_every_seed_that_did(self) -> None:
        assert self._pick({1: 10, 2: 9999, 3: None}) == 3

    def test_it_is_the_expression_the_script_uses(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Driven through the script rather than copied out of it. Seed two
        # never arrives, so it is the one the second table has to be about.
        made = {1: script.Run(first_goal=5, final=-20.0, exact=-13.0)}
        made[2] = script.Run(first_goal=None, final=-99.0, exact=None)
        made[3] = script.Run(first_goal=40, final=-30.0, exact=-17.0)
        monkeypatch.setattr(script, "run", lambda seed, episodes, entropy: made[seed])
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_lost_seed",
                "--runs",
                "3",
                "--budgets",
                "50",
                "--entropies",
                "0.1",
                "--ladder-episodes",
                "10",
            ],
        )
        assert script.main() == 0
        assert "Seed 2 holds out longest" in capsys.readouterr().out


class TestTheFirstTable:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_lost_seed",
                "--runs",
                "2",
                "--budgets",
                "5",
                "10",
                "--entropies",
                "0.1",
                "--ladder-episodes",
                "5",
                *extra,
            ],
        )
        assert script.main() == 0

    def test_every_budget_gets_a_row(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch)
        printed = capsys.readouterr().out
        assert "not there yet" in printed
        assert "which seeds" in printed

    def test_each_seed_is_run_once_at_the_longest_budget(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once rather than once per budget, which is what makes it affordable.

        A run of a thousand episodes contains the run of five hundred, so the
        first table reads one run at several budgets. A script that re-ran
        each budget would take three times as long and could disagree with
        itself between rows.
        """
        asked: list[tuple[int, int]] = []
        real = script.Run(first_goal=3, final=-20.0, exact=-13.0)
        monkeypatch.setattr(
            script,
            "run",
            lambda seed, episodes, entropy: (asked.append((seed, episodes)), real)[1],
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_lost_seed",
                "--runs",
                "2",
                "--budgets",
                "5",
                "10",
                "20",
                "--entropies",
                "0.1",
                "--ladder-episodes",
                "7",
            ],
        )
        script.main()

        # Two seeds at the longest budget, then the ladder at its own length.
        assert asked[:2] == [(1, 20), (2, 20)]


class TestTheLadder:
    def test_it_runs_one_seed_by_default(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[int] = []
        real = script.Run(first_goal=3, final=-20.0, exact=-13.0)
        monkeypatch.setattr(
            script,
            "run",
            lambda seed, episodes, entropy: (asked.append(seed), real)[1],
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_lost_seed",
                "--runs",
                "3",
                "--budgets",
                "5",
                "--entropies",
                "0.1",
                "0.2",
                "--ladder-episodes",
                "7",
            ],
        )
        script.main()
        assert len(asked) == 3 + 2

    def test_ladder_all_runs_every_seed_at_every_entropy(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Which is what says whether the default should move, rather than
        # whether one seed can be rescued.
        asked: list[int] = []
        real = script.Run(first_goal=3, final=-20.0, exact=-13.0)
        monkeypatch.setattr(
            script,
            "run",
            lambda seed, episodes, entropy: (asked.append(seed), real)[1],
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_lost_seed",
                "--runs",
                "3",
                "--budgets",
                "5",
                "--entropies",
                "0.1",
                "0.2",
                "--ladder-episodes",
                "7",
                "--ladder-all",
            ],
        )
        script.main()
        assert len(asked) == 3 + 3 * 2

    def test_a_column_that_counts_nothing_is_left_blank(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A column of zeros is a column a reader stops seeing, so the counts
        # of seeds that never arrived and seeds that are stuck are blank when
        # they are nothing.
        real = script.Run(first_goal=3, final=-20.0, exact=-13.0)
        monkeypatch.setattr(script, "run", lambda seed, episodes, entropy: real)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_lost_seed",
                "--runs",
                "2",
                "--budgets",
                "5",
                "--entropies",
                "0.1",
                "--ladder-episodes",
                "7",
            ],
        )
        script.main()
        ladder = capsys.readouterr().out.split("never got there")[1]
        assert " 0 " not in ladder

    def test_a_ladder_with_no_arrivals_says_never(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Rather than a median of an empty list, which would be an error, and
        # rather than a zero, which would read as arriving immediately.
        lost = script.Run(first_goal=None, final=-100.0, exact=None)
        monkeypatch.setattr(script, "run", lambda seed, episodes, entropy: lost)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_lost_seed",
                "--runs",
                "2",
                "--budgets",
                "5",
                "--entropies",
                "0.1",
                "--ladder-episodes",
                "7",
            ],
        )
        script.main()
        printed = capsys.readouterr().out
        assert "never" in printed.split("never got there")[1]
