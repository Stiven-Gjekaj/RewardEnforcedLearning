"""The measurement behind the clipped policy tables on the algorithms page.

The load bearing test is `TestTheBudgetsAreMatched`. The whole first table is
a claim about equal gradient steps on unequal environment budgets, and the
column that says so is `passes * episodes`. If a row's settings and its printed
step count came apart, the table would still print, the ordering of the rows
would be unchanged, and the comparison it exists to make would be gone.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_clipped.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_clipped", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def steps(row: tuple[str, str, int, dict[str, object]]) -> int:
    _, _, episodes, settings = row
    return int(settings.get("passes", 1)) * episodes  # type: ignore[arg-type]


class TestTheBudgetsAreMatched:
    def test_the_row_it_compares_against_is_the_full_budget(
        self, script: ModuleType
    ) -> None:
        assert script.BUDGETS[0][0].startswith("reinforce")
        assert script.BUDGETS[0][2] == 400

    def test_the_reusing_row_has_the_same_gradient_steps(
        self, script: ModuleType
    ) -> None:
        rows = {row[0]: row for row in script.BUDGETS}
        assert steps(rows["clipped 4 x 100"]) == steps(rows["reinforce, 400"])

    def test_the_reusing_row_asks_the_environment_for_a_quarter(
        self, script: ModuleType
    ) -> None:
        rows = {row[0]: row for row in script.BUDGETS}
        assert rows["clipped 4 x 100"][2] * 4 == rows["reinforce, 400"][2]

    def test_the_control_at_the_small_budget_is_there(self, script: ModuleType) -> None:
        # A method that reuses an episode has to beat collecting nothing as
        # well as beat collecting more, and that row is what says which.
        rows = {row[0]: row for row in script.BUDGETS}
        assert rows["reinforce, 100"][2] == rows["clipped 4 x 100"][2]

    def test_one_pass_is_a_control_rather_than_a_second_agent(
        self, script: ModuleType
    ) -> None:
        rows = {row[0]: row for row in script.BUDGETS}
        assert rows["clipped 1 x 400"][3]["passes"] == 1
        assert rows["clipped 1 x 400"][2] == rows["reinforce, 400"][2]

    def test_every_label_is_its_own(self, script: ModuleType) -> None:
        labels = [row[0] for row in script.BUDGETS]
        assert len(set(labels)) == len(labels)


class TestWhatOneSettingReturns:
    def test_it_gives_one_number_for_each_seed(self, script: ModuleType) -> None:
        got, _ = script.one_setting("reinforce", 3, 2, "cartpole", {})
        assert len(got) == 2

    def test_an_agent_with_no_clip_reports_no_share(self, script: ModuleType) -> None:
        _, share = script.one_setting("reinforce", 3, 1, "cartpole", {})
        assert share == 0.0

    def test_an_agent_with_a_clip_reports_a_share_between_none_and_all(
        self, script: ModuleType
    ) -> None:
        _, share = script.one_setting(
            "clipped-policy", 12, 1, "cartpole", {"passes": 4}
        )
        assert 0.0 <= share <= 1.0

    def test_a_narrow_clip_binds_more_than_a_wide_one(self, script: ModuleType) -> None:
        _, narrow = script.one_setting(
            "clipped-policy", 12, 2, "cartpole", {"passes": 4, "clip_range": 0.01}
        )
        _, wide = script.one_setting(
            "clipped-policy", 12, 2, "cartpole", {"passes": 4, "clip_range": 2.0}
        )
        assert narrow > wide

    def test_the_same_seeds_give_the_same_numbers(self, script: ModuleType) -> None:
        first, _ = script.one_setting("reinforce", 4, 2, "cartpole", {})
        again, _ = script.one_setting("reinforce", 4, 2, "cartpole", {})
        assert first == again


class TestItRuns:
    """The sections print, on budgets small enough to be a test.

    Not `BUDGETS`, which carries the 400 episodes the page reports. Running
    those here cost this file three minutes of a suite that runs in eight, to
    check that a table has a heading.
    """

    #: Two rows shaped like the real ones and small enough to run twice.
    TINY = (
        ("reinforce, 6", "reinforce", 6, {}),
        ("clipped 2 x 3", "clipped-policy", 3, {"passes": 2}),
    )

    def test_the_short_run_prints_all_three_sections(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.budget_section(self.TINY, 2, "cartpole", script.Rng(1))
        script.sweep_section((0.2,), (1,), 1, 6, "cartpole")
        script.pendulum_section(("random",), 1, 5, "pendulum-levels")
        printed = capsys.readouterr().out
        assert "Reusing an episode" in printed
        assert "Where the clip binds" in printed
        assert "no policy gradient agent here learns" in printed

    def test_it_says_when_the_seeds_cannot_reach_a_p(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.budget_section(self.TINY, 2, "cartpole", script.Rng(1))
        assert "cannot give a p below" in capsys.readouterr().out

    def test_the_two_rows_it_tests_with_are_shaped_like_the_real_ones(
        self, script: ModuleType
    ) -> None:
        # Same four fields in the same order, so a change to the real rows
        # that broke the section would break these too.
        assert [len(row) for row in self.TINY] == [
            len(row) for row in script.BUDGETS[:2]
        ]
        assert all(row[1] in {"reinforce", "clipped-policy"} for row in self.TINY)
