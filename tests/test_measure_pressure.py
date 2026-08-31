"""The measurement behind the optimisation pressure tables on the gaming page.

The load bearing test is `TestTheTwoFloorsAreDifferentOnPurpose`. The reward
share is anchored at a uniform policy and the objective share at the
objective's own zero, and that asymmetry is the whole finding: on all three of
these the agent ends up worse at the real objective than doing nothing at all,
and a floor set at what a uniform policy happens to score would clamp that
away.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.pressure import LADDER

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_pressure.py"

#: The columns a row of any table carries, in the order it carries them.
PRESSURE, PAID, REWARD, AUDIT, POINT, GAP = range(6)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_pressure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheCases:
    def test_there_is_one_for_each_environment_the_gaming_suite_has(
        self, script: ModuleType
    ) -> None:
        assert [case.name for case in script.CASES] == [
            "boat race",
            "vase room",
            "thermostat",
        ]

    def test_the_gamed_and_repaired_builds_really_differ(
        self, script: ModuleType
    ) -> None:
        """The one thing that has to be true for the ceiling to mean anything.

        The repaired build is what says how much of the real objective was
        available at all. A case whose two builds were the same environment
        would report a ceiling the gamed reward already reaches, and every
        objective share would come out at one.
        """
        from rel.rng import Rng

        for case in script.CASES:
            gamed = case.gamed(Rng(1).stream("env"))
            repaired = case.repaired(Rng(1).stream("env"))
            rewards = [
                (
                    gamed.transitions(state, action)[0].reward,
                    repaired.transitions(state, action)[0].reward,
                )
                for state in range(gamed.observation_space.n)
                for action in range(gamed.action_space.n)
            ]
            assert any(first != second for first, second in rewards), case.name

    def test_each_one_reads_its_discount_from_its_environment(
        self, script: ModuleType
    ) -> None:
        # Rather than a number written here. A discount that drifted from the
        # environment's would solve a different problem than the one the
        # gaming page reports.
        from rel.rng import Rng

        for case in script.CASES:
            env = case.gamed(Rng(1).stream("env"))
            assert case.discount == env.spec.suggested_discount


class TestARow:
    def test_there_is_one_for_each_rung_of_the_ladder(self, script: ModuleType) -> None:
        rows = script.measure(script.CASES[0], episodes=2, seed=1)
        assert len(rows) == len(LADDER)

    def test_the_pressure_runs_from_nothing_to_everything(
        self, script: ModuleType
    ) -> None:
        rows = script.measure(script.CASES[0], episodes=2, seed=1)
        assert [row[PRESSURE] for row in rows] == [
            f"{1.0 - epsilon:.1f}" for epsilon in LADDER
        ]

    def test_the_gap_is_the_first_share_minus_the_second(
        self, script: ModuleType
    ) -> None:
        # Every column is rounded to two places on its own, so the gap can
        # differ from the difference between the two printed shares by as
        # much as a hundredth either way.
        for case in script.CASES:
            for row in script.measure(case, episodes=2, seed=1):
                gap = float(row[REWARD]) - float(row[POINT])
                assert float(row[GAP]) == pytest.approx(gap, abs=0.011), case.name

    def test_a_seed_replays_it(self, script: ModuleType) -> None:
        first = script.measure(script.CASES[1], episodes=2, seed=3)
        assert script.measure(script.CASES[1], episodes=2, seed=3) == first


class TestTheRewardShareRisesToOne:
    """By construction, and that is what makes it the control column.

    The policy at the last rung is the one solved for the stated reward, so
    the reward share there is one whatever the environment pays in. A column
    that did not reach one would mean the ladder was not walking to the
    optimum, and every reading of the other column would be against a moving
    reference.
    """

    def test_the_last_rung_is_a_full_share(self, script: ModuleType) -> None:
        for case in script.CASES:
            rows = script.measure(case, episodes=4, seed=1)
            assert rows[-1][REWARD] == "1.00", case.name

    def test_the_first_rung_is_none_of_it(self, script: ModuleType) -> None:
        # The uniform policy is the floor of that share, so it sits at zero
        # by the same construction.
        for case in script.CASES:
            rows = script.measure(case, episodes=4, seed=1)
            assert rows[0][REWARD] == "0.00", case.name


class TestTheTwoFloorsAreDifferentOnPurpose:
    """The reward share starts at a uniform policy and the objective at zero.

    That asymmetry is the finding rather than an oversight. Anchoring the
    objective at the uniform policy would make its first rung zero as well,
    and a run that ends below where it started would be clamped to nothing
    instead of read.
    """

    def test_the_objective_share_does_not_start_at_zero_everywhere(
        self, script: ModuleType
    ) -> None:
        starts = [
            script.measure(case, episodes=4, seed=1)[0][POINT] for case in script.CASES
        ]
        assert any(start != "0.00" for start in starts), starts

    def test_the_objective_can_end_below_where_it_started(
        self, script: ModuleType
    ) -> None:
        # The whole point of the exercise: trying harder at the stated reward
        # makes the real objective worse on these environments.
        fell = [
            case.name
            for case in script.CASES
            if float(script.measure(case, episodes=8, seed=1)[-1][POINT])
            < float(script.measure(case, episodes=8, seed=1)[0][POINT])
        ]
        assert fell, "no case falls, so there is nothing for the gap to show"

    def test_the_unmet_reading_is_the_floor_of_each_objective(
        self, script: ModuleType
    ) -> None:
        # Zero laps, a broken vase, no comfortable steps. Written down here
        # because a floor that drifted would silently rescale a column.
        assert [case.unmet for case in script.CASES] == [0.0, 1.0, 0.0]


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
        monkeypatch.setattr(
            sys, "argv", ["measure_pressure", "--episodes", "2", *extra]
        )
        return int(script.main())

    def test_every_case_gets_a_table(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch) == 0
        printed = capsys.readouterr().out
        for case in script.CASES:
            assert case.name in printed
            assert case.units in printed

    def test_it_says_how_many_episodes_each_rung_ran(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A noisy policy is noisy, so the number of episodes behind a rung is
        # part of what the rung means.
        self._run(script, monkeypatch)
        assert "2 episodes" in capsys.readouterr().out

    def test_it_says_which_seed_it_ran(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--seed", "6")
        assert "seed 6" in capsys.readouterr().out
