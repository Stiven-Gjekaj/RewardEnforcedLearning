"""The measurement behind the two options tables on the algorithms page.

The page states three rows for the collapse, and nothing printed the first of
them. That row is the one that makes the section a claim rather than a
tautology: `q-learning` is a different class reaching the same number as
`options-q` with its options taken away.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_options.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_options", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestMeasure:
    def test_it_runs_the_agent_it_is_asked_for(self, script: ModuleType) -> None:
        """The change that let the collapse be printed at all.

        It ran `options-q` and nothing else, so the row that says Q-learning
        reaches the same number could not be produced by it.
        """
        one = script.measure("rooms", False, 0.1, 20, 2, "options-q")
        two = script.measure("rooms", False, 0.1, 20, 2, "q-learning")
        assert one[0] != two[0] or one[1] == two[1]

    def test_an_agent_with_no_options_counts_none(self, script: ModuleType) -> None:
        # Q-learning has no options to have chosen and none that ran for more
        # than a step, and asking it would be an attribute error.
        _, _, _, long, length = script.measure("rooms", False, 0.1, 20, 2, "q-learning")
        assert long == 0.0
        assert length == 1.0

    def test_options_are_chosen_when_they_are_there(self, script: ModuleType) -> None:
        _, _, _, long, length = script.measure("rooms", True, 0.1, 40, 2)
        assert long > 0.0
        assert length > 1.0

    def test_the_default_is_the_options_agent(self, script: ModuleType) -> None:
        with_name = script.measure("rooms", True, 0.1, 20, 2, "options-q")
        without = script.measure("rooms", True, 0.1, 20, 2)
        # Compared a field at a time, because the exact value of the greedy
        # policy is a nan when no seed reached the end, and a nan is not
        # equal to itself.
        assert all(
            (math.isnan(one) and math.isnan(two)) or one == two
            for one, two in zip(with_name, without, strict=True)
        )


class TestTheCollapse:
    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        *rest: str,
    ) -> str:
        monkeypatch.setattr(
            sys,
            "argv",
            ["measure_options", "--episodes", "20", "--runs", "2", *rest],
        )
        assert script.main() == 0
        return capsys.readouterr().out

    def test_it_prints_the_three_rows(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        printed = self.go(script, monkeypatch, capsys, "--epsilons", "0.1")
        collapse = printed.split("## The cost")[0]
        assert "q-learning" in collapse
        assert "options-q, hallways=off" in collapse
        assert collapse.count("options-q") == 2

    def test_the_ladder_still_follows_it(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        printed = self.go(script, monkeypatch, capsys, "--epsilons", "0.1", "0.05")
        assert printed.index("## The collapse") < printed.index("## The cost")
        assert "cost/epsilon" in printed

    def test_it_is_measured_at_the_registry_default(self, script: ModuleType) -> None:
        # The ladder can be asked for any rates. The collapse is the claim
        # about the default, so it does not move with `--epsilons`.
        from rel.agents import AGENTS

        assert AGENTS["options-q"].options(2)["epsilon"] == script.DEFAULT_EPSILON

    def test_a_stuck_count_of_zero_is_blank(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The same as the ladder table below it, and the same as the page.
        printed = self.go(script, monkeypatch, capsys, "--epsilons", "0.1")
        collapse = printed.split("## The cost")[0]
        assert " 0\n" not in collapse
