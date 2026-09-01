"""The measurement behind the double estimation tables on the algorithms page.

The load bearing test is `TestTheFloorItComparesAgainst`. Every number in
those tables is read against the share of episodes an agent that has learned
the answer still goes the wrong way, and that share is `epsilon / actions`. If
it were computed wrongly the tables would still be printed, the ordering of
the rows would be unchanged, and every "times the floor" figure would be wrong
by the same factor with nothing to catch it.

`TestItRefusesOverlappingWindows` is the other one. The two columns are the
start of a run and the end of it, and on a short run they are the same
episodes twice, which would read as the bias having faded.

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

from rel.envs.bias import MaximisationBias
from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_double.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_double", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheFloorItComparesAgainst:
    def test_it_is_the_exploring_share_split_over_the_actions(
        self, script: ModuleType
    ) -> None:
        env = MaximisationBias(Rng(1), actions=10)
        assert script.floor_share(env, 0.1) == pytest.approx(0.01)

    def test_it_follows_the_action_count_rather_than_being_quoted(
        self, script: ModuleType
    ) -> None:
        # The book's two action version gives the 5 percent the figure shows.
        assert script.floor_share(MaximisationBias(Rng(1), actions=2), 0.1) == (
            pytest.approx(0.05)
        )
        assert script.floor_share(MaximisationBias(Rng(1), actions=20), 0.1) == (
            pytest.approx(0.005)
        )

    def test_no_exploring_means_no_wrong_turns_at_all(self, script: ModuleType) -> None:
        assert script.floor_share(MaximisationBias(Rng(1), actions=10), 0.0) == 0.0


class TestItRefusesOverlappingWindows:
    def test_a_run_too_short_for_both_windows_is_refused(
        self, script: ModuleType
    ) -> None:
        with pytest.raises(SystemExit, match="without overlapping"):
            script.bias_section(
                "heading",
                script.TABLES,
                "bias",
                2,
                30,
                0.1,
                Rng(1),
                (20, 20),
            )

    def test_a_run_exactly_long_enough_is_allowed(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.bias_section(
            "heading",
            (("q-learning", {"step_size": 0.1}),),
            "bias",
            2,
            40,
            0.1,
            Rng(1),
            (20, 20),
        )
        assert "q-learning" in capsys.readouterr().out


class TestWhatItCompares:
    def test_the_tabular_pair_is_the_one_table_against_the_two(
        self, script: ModuleType
    ) -> None:
        assert [name for name, _ in script.TABLES] == ["q-learning", "double-q"]

    def test_the_network_pair_is_one_agent_with_one_setting_changed(
        self, script: ModuleType
    ) -> None:
        # Both sides being the same code is what makes the difference the
        # setting rather than two implementations.
        settings = [dict(one) for _, one in script.NETWORKS]
        assert settings == [{"double": False}, {"double": True}]

    def test_the_first_row_is_the_one_the_rest_are_read_against(
        self, script: ModuleType
    ) -> None:
        assert script.TABLES[0][1] == {}
        assert script.NETWORKS[0][1] == {"double": False}


class TestTheSettlingLadder:
    def test_the_ladder_doubles(self, script: ModuleType) -> None:
        # So that a bias fading like one over the episodes and a bias
        # flattening out look different rather than similar.
        rungs = script.LADDER
        assert len(rungs) > 2
        for lower, upper in itertools.pairwise(rungs):
            assert upper == lower * 2

    def test_it_prints_a_median_beside_every_mean(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # At the long budgets a seed that never recovers moves the mean by
        # more than the whole difference between the agents, so a mean alone
        # would be a failure count wearing the clothes of a result.
        script.settling_section(
            (("q-learning", {"step_size": 0.1}), ("double-q", {"step_size": 0.1})),
            "bias",
            3,
            (60, 120),
            0.1,
            40,
        )
        printed = capsys.readouterr().out
        assert printed.count("median") == 2
        assert printed.count("mean") >= 2

    def test_a_rung_shorter_than_the_window_is_left_out(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Averaging the last forty episodes of a run of twenty would average
        # the whole run and call it the end of it.
        script.settling_section(
            (("q-learning", {"step_size": 0.1}),), "bias", 2, (20, 80), 0.1, 40
        )
        printed = capsys.readouterr().out
        assert "\n" in printed
        rows = [line for line in printed.splitlines() if line.strip().startswith("8")]
        assert len(rows) == 1
        assert not [
            line for line in printed.splitlines() if line.strip().startswith("20 ")
        ]


class TestWhatItPrints:
    @pytest.fixture
    def printed(self, script: ModuleType, capsys: pytest.CaptureFixture[str]) -> str:
        script.bias_section(
            "Two tables against one.",
            (("q-learning", {"step_size": 0.1}), ("double-q", {"step_size": 0.1})),
            "bias",
            4,
            120,
            0.1,
            Rng(1),
            (20, 40),
        )
        return capsys.readouterr().out

    def test_it_says_what_the_floor_is(self, printed: str) -> None:
        assert "still goes left 0.010" in printed

    def test_it_names_both_windows_by_their_lengths(self, printed: str) -> None:
        assert "first 20" in printed
        assert "last 40" in printed

    def test_it_has_a_row_for_each_agent(self, printed: str) -> None:
        assert "q-learning" in printed
        assert "double-q" in printed

    def test_it_warns_when_the_seeds_cannot_reach_a_small_p(self, printed: str) -> None:
        # Four seeds have sixteen sign patterns, so the smallest p is 0.125.
        assert "0.1250" in printed
