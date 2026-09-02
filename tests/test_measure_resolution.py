"""The measurement behind the resolution ladder on the gaming pages.

The load bearing test is `TestTheAnchorRows`. That table's whole finding rests
on one comparison: `grouped-q` at one group pays what a uniform policy pays and
behaves nothing like one. If the uniform anchor were computed from the agent
rather than from a uniform policy, both rows would agree, the table would still
print, and the finding would be a tautology.

`TestTheSharesMatchThePressureLadder` is the other one. The two scripts print
columns with the same names, and they are only readable side by side if the
same value maps to the same share in both.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.pressure import ladder, share

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_resolution.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_resolution", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def boat(script: ModuleType) -> object:
    return next(case for case in script.CASES if case.name == "boat race")


class TestTheAnchorRows:
    def test_the_uniform_row_is_a_uniform_policy(self, script: ModuleType) -> None:
        case = boat(script)
        rows = script.measure(case, 1, 60)
        wanted = ladder(case.gamed, case.discount, epsilons=(1.0,), episodes=40)[0]
        assert rows[0][0] == "uniform"
        assert rows[0][2] == f"{wanted.paid:.1f}"

    def test_the_last_row_is_the_solved_policy(self, script: ModuleType) -> None:
        case = boat(script)
        rows = script.measure(case, 1, 60)
        wanted = ladder(case.gamed, case.discount, epsilons=(0.0,), episodes=40)[0]
        assert rows[-1][0] == "solved"
        assert rows[-1][2] == f"{wanted.paid:.1f}"

    def test_the_uniform_row_reads_one_at_the_reward_it_is_the_floor_of(
        self, script: ModuleType
    ) -> None:
        # Zero by construction, which is what makes it a floor rather than a
        # measurement, and is why the control is its audited column instead.
        rows = script.measure(boat(script), 1, 60)
        assert rows[0][3] == "0.00"

    def test_the_solved_row_reads_one_at_the_reward(self, script: ModuleType) -> None:
        rows = script.measure(boat(script), 1, 60)
        assert rows[-1][3] == "1.00"

    def test_the_anchors_are_not_computed_from_the_agent(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An agent replaced wholesale must move every learned row and neither
        # anchor. Anchors taken from the agent would agree with it by
        # construction and the table would compare the agent with itself.
        case = boat(script)
        before = script.measure(case, 1, 60)

        monkeypatch.setattr(script, "learned", lambda *args: ([0.0], [99.0]))
        after = script.measure(case, 1, 60)

        assert before[0] == after[0]
        assert before[-1] == after[-1]
        for row in after[1:-1]:
            assert row[2] == "0.0"
            assert row[4] == "99.00"


class TestTheSharesMatchThePressureLadder:
    def test_the_reward_share_runs_from_the_uniform_to_the_solved(
        self, script: ModuleType
    ) -> None:
        case = boat(script)
        ends = ladder(case.gamed, case.discount, epsilons=(1.0, 0.0), episodes=40)
        middle = (ends[0].paid + ends[1].paid) / 2.0
        assert share(middle, ends[0].paid, ends[1].paid) == pytest.approx(0.5)

    def test_the_point_share_runs_from_the_unmet_to_the_repaired(
        self, script: ModuleType
    ) -> None:
        case = boat(script)
        reachable = ladder(case.repaired, case.discount, epsilons=(0.0,), episodes=40)[
            0
        ]
        assert share(case.unmet, case.unmet, reachable.audit[case.key]) == 0.0
        assert (
            share(reachable.audit[case.key], case.unmet, reachable.audit[case.key])
            == 1.0
        )

    def test_a_broken_vase_counts_downwards(self, script: ModuleType) -> None:
        # Its unmet value is one rather than zero, because a broken vase is
        # the objective not being met. A share that ran the other way would
        # report the worst outcome as the best.
        case = next(one for one in script.CASES if one.name == "vase room")
        assert case.unmet == 1.0
        assert share(1.0, case.unmet, 0.0) == 0.0
        assert share(0.0, case.unmet, 0.0) == 1.0


class TestTheLadders:
    def test_every_rung_fits_inside_its_environment(self, script: ModuleType) -> None:
        for case in script.CASES:
            assert max(case.rungs) <= case.states
            assert min(case.rungs) == 1

    def test_the_bottom_rung_is_the_table(self, script: ModuleType) -> None:
        # Which is what makes the ladder a dial rather than a set of agents.
        for case in script.CASES:
            assert case.states in case.rungs

    def test_the_rungs_rise(self, script: ModuleType) -> None:
        for case in script.CASES:
            assert list(case.rungs) == sorted(case.rungs)

    def test_it_runs_the_same_three_the_pressure_ladder_does(
        self, script: ModuleType
    ) -> None:
        assert [case.name for case in script.CASES] == [
            "boat race",
            "vase room",
            "thermostat",
        ]


class TestEachSeed:
    def test_it_fills_in_the_seeds_when_asked(self, script: ModuleType) -> None:
        case = boat(script)
        seeds: dict[int, list[float]] = {}
        script.measure(case, 3, 60, seeds)
        assert sorted(seeds) == sorted(case.rungs)
        assert all(len(one) == 3 for one in seeds.values())

    def test_it_fills_in_nothing_when_not_asked(self, script: ModuleType) -> None:
        seeds: dict[int, list[float]] = {}
        script.measure(boat(script), 3, 60)
        assert seeds == {}

    def test_the_table_prints_the_mean_of_what_it_filled_in(
        self, script: ModuleType
    ) -> None:
        import statistics

        case = boat(script)
        seeds: dict[int, list[float]] = {}
        rows = script.measure(case, 3, 60, seeds)
        for row in rows[1:-1]:
            assert row[4] == f"{statistics.mean(seeds[int(row[0])]):.2f}"

    def test_the_mean_can_be_a_run_no_seed_made(self, script: ModuleType) -> None:
        # Which is why the seeds are printable. At one group on the boat race
        # every seed is one of two policies and neither of them is the mean.
        seeds: dict[int, list[float]] = {}
        script.measure(boat(script), 6, 200, seeds)
        assert len(set(seeds[1])) == 2


class TestWhatItPrints:
    def test_it_names_the_control_under_the_tables(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch
    ) -> None:
        monkeypatch.setattr(script, "CASES", (boat(script),))
        monkeypatch.setattr(
            sys, "argv", ["measure_resolution.py", "--runs", "1", "--episodes", "40"]
        )
        script.main()
        printed = capsys.readouterr().out
        assert "uniform" in printed
        assert "solved" in printed
        assert "audited column" in printed

    def test_a_case_says_how_many_states_it_has(self, script: ModuleType) -> None:
        assert boat(script).states == 16
        assert boat(script).discount == pytest.approx(0.99)


def test_the_seed_gives_the_same_table(script: ModuleType) -> None:
    assert script.measure(boat(script), 1, 60) == script.measure(boat(script), 1, 60)


class TestTheSettingsItCanSweep:
    """Four things the ladder held fixed, and now takes.

    The page named all four as unswept: one agent, one step size, one epsilon
    and one grouping rule. Each is now an option, and these check that an
    option actually reaches the agent rather than being accepted and dropped.
    """

    def test_an_agent_name_it_cannot_build_is_refused(self, script: ModuleType) -> None:
        with pytest.raises(KeyError):
            script.learned(script.CASES[0], 2, 1, 2, "grouped-nothing", {})

    def test_it_runs_the_on_policy_agent_when_asked(self, script: ModuleType) -> None:
        paid, audited = script.learned(
            script.CASES[0], 4, 2, 30, "grouped-sarsa", {"grouping": "blocks"}
        )
        assert len(paid) == 2
        assert len(audited) == 2

    def test_a_setting_reaches_the_agent(self, script: ModuleType) -> None:
        # A step size of zero learns nothing, so a run at zero and a run at
        # the default cannot agree unless the setting was dropped.
        still = script.learned(
            script.CASES[0], 4, 2, 30, "grouped-q", {"step_size": 0.0}
        )
        moved = script.learned(script.CASES[0], 4, 2, 30, "grouped-q", {})
        assert still != moved

    def test_the_grouping_reaches_the_coder(self, script: ModuleType) -> None:
        # A grouping the coder does not have is refused there, so a run that
        # dropped the setting would quietly succeed under the default.
        with pytest.raises(ValueError, match="is not a grouping"):
            script.learned(
                script.CASES[0], 4, 1, 2, "grouped-q", {"grouping": "diagonal"}
            )

    def test_one_group_is_the_same_run_under_either_grouping(
        self, script: ModuleType
    ) -> None:
        # At one group there is nothing to arrange, so the two rules have to
        # give the same run. A difference here would be a fault in the coder.
        blocks = script.learned(
            script.CASES[0], 1, 2, 60, "grouped-q", {"grouping": "blocks"}
        )
        stripes = script.learned(
            script.CASES[0], 1, 2, 60, "grouped-q", {"grouping": "stripes"}
        )
        assert blocks == stripes

    def test_the_default_grouping_is_the_one_the_page_reports(
        self, script: ModuleType
    ) -> None:
        assert script.GROUPING == "blocks"
