"""The measurement behind the car rental table on the algorithms page.

The load bearing test is `TestTheCountOfStatesThatMove`. The table's last two
columns say how much of the board the van is used on, and several actions clip
onto moving nothing at the edges of that board. A count of actions that are
not the middle one would count states where nothing happens, and the column
would say the van was used where it was not.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_rental.py"

#: Small enough to sweep in a test. The book's board is 441 states and eleven
#: actions, which is half a minute a row.
SMALL = {"capacity": 4, "van": 2}


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_rental", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSolving:
    def test_it_gives_back_the_environment_it_solved(self, script: ModuleType) -> None:
        # The policy is read through the environment's own `moved`, so the
        # two have to be the same one.
        env, solved = script.solve(4, 2, 2.0)
        assert env.capacity == 4
        assert env.van == 2
        assert len(solved.policy) == env.observation_space.n

    def test_the_price_reaches_the_environment(self, script: ModuleType) -> None:
        env, _ = script.solve(4, 2, 7.0)
        assert env.move_cost == 7.0

    def test_a_van_that_holds_nothing_has_one_action(self, script: ModuleType) -> None:
        env, _ = script.solve(4, 0, 0.0)
        assert env.action_space.n == 1


class TestTheCountOfStatesThatMove:
    """Counted as moves rather than as action numbers.

    At the edges of the board several actions clip onto moving nothing, so a
    count of actions that are not the middle one would say the van was used
    where nothing happens.
    """

    def test_a_van_that_holds_nothing_moves_nowhere(self, script: ModuleType) -> None:
        env, solved = script.solve(4, 0, 0.0)
        assert script.moving_states(env, solved) == 0
        assert script.largest_move(env, solved) == 0

    def test_a_free_van_moves_somewhere(self, script: ModuleType) -> None:
        env, solved = script.solve(4, 2, 0.0)
        assert script.moving_states(env, solved) > 0

    def test_a_van_nobody_would_pay_for_moves_nowhere(self, script: ModuleType) -> None:
        # The takings for a car are 10, so a van charging 200 a car is one no
        # policy uses, and the count has to see that.
        env, solved = script.solve(4, 2, 200.0)
        assert script.moving_states(env, solved) == 0

    def test_a_clipped_action_is_not_counted_as_a_move(
        self, script: ModuleType
    ) -> None:
        # Driven rather than argued. A policy of nothing but the outermost
        # action moves cars in the middle of the board and nothing at the
        # corner where there is neither a car nor a space.
        env, _ = script.solve(4, 2, 2.0)

        class Everywhere:
            policy = tuple([env.action_space.n - 1] * env.observation_space.n)

        moving = script.moving_states(env, Everywhere())
        assert moving < env.observation_space.n
        assert env.moved(env.fold(4, 0), env.action_space.n - 1) == 0

    def test_the_largest_move_is_never_more_than_the_van_holds(
        self, script: ModuleType
    ) -> None:
        env, solved = script.solve(4, 2, 0.0)
        assert script.largest_move(env, solved) <= env.van


class TestAGapTheSweepCannotSee:
    """A van priced past what it is worth is never used.

    So it reaches the value that no van reaches, and the two sweeps land
    about 1e-10 apart. Printing that as a small negative number would say the
    van cost something, which is a claim the sweep cannot support.
    """

    def test_a_gap_below_the_tolerance_is_nothing(self, script: ModuleType) -> None:
        assert script.settled(-1.2e-10) == "0.000"
        assert script.settled(0.0) == "0.000"

    def test_a_gap_above_it_keeps_its_sign(self, script: ModuleType) -> None:
        assert script.settled(2.5) == "+2.500"
        assert script.settled(-2.5) == "-2.500"

    def test_the_threshold_is_the_solver_s_own_tolerance(
        self, script: ModuleType
    ) -> None:
        # Read from `rel.agents.dp` rather than written here, so a change to
        # the solver's tolerance moves this with it.
        from rel.agents.dp import TOLERANCE

        assert script.settled(TOLERANCE * 2) != "0.000"
        assert script.settled(TOLERANCE / 2) == "0.000"


class TestThePriceTable:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_rental",
                "--capacity",
                str(SMALL["capacity"]),
                "--van",
                str(SMALL["van"]),
                *extra,
            ],
        )
        assert script.main() == 0

    def test_the_control_row_is_a_van_that_holds_nothing(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Without a row that cannot move a car, every other row is read
        # against nothing.
        self._run(script, monkeypatch, "--costs", "2")
        assert "no van" in capsys.readouterr().out

    def test_every_price_it_was_given_gets_a_row(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--costs", "0", "3")
        printed = capsys.readouterr().out
        rows = [
            line.split()[0]
            for line in printed.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        assert rows == ["0", "3"]

    def test_a_free_van_is_worth_more_than_a_priced_one(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--costs", "0", "2")
        values = [
            float(line.split()[1])
            for line in capsys.readouterr().out.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        assert values[0] > values[1]

    def test_a_price_nobody_pays_gains_nothing(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A van that is never used leaves the value where a van that cannot
        # move a car leaves it, and the two sweeps land about 1e-10 apart.
        self._run(script, monkeypatch, "--costs", "200")
        row = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("200")
        ]
        assert len(row) == 1
        assert " 0.000 " in row[0]
        assert "-0.000" not in row[0]

    def test_the_day_column_is_the_gap_undiscounted(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A gain of g every day is worth g over one minus the discount, so
        # the per day number is the gap times one tenth. That is what makes
        # it comparable with the price the van charges.
        self._run(script, monkeypatch, "--costs", "0")
        row = next(
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("0 ")
        )
        gap, a_day = float(row.split()[2]), float(row.split()[3])
        assert a_day == pytest.approx(gap * (1.0 - script.DISCOUNT), abs=0.001)

    def test_it_says_how_many_states_the_count_is_out_of(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--costs", "2")
        assert "out\n  of 25" in capsys.readouterr().out
