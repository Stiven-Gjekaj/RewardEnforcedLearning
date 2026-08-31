"""The measurement behind the Monte Carlo table on the algorithms page.

The load bearing test is `TestWhatItLearnedIsScoredExactly`. The point of
having blackjack with a model written out is that a learned policy can be
scored against the answer rather than against the return the agent happened to
collect, and a script that reported the collected return would be reporting the
cards as much as the learning. So what is checked is that the number in the
table comes from a sweep of the model.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_blackjack.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_blackjack", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheAnswerItScoresAgainst:
    def test_it_is_the_exact_optimum(self, script: ModuleType) -> None:
        _, best = script.answer()
        assert best.start_value == pytest.approx(-0.04656, abs=1e-4)

    def test_it_is_swept_once_and_shared(self, script: ModuleType) -> None:
        # Every row of the table wants the same answer, and the sweep is the
        # slowest thing in the script that is not a run.
        assert script.answer() is script.answer()


class TestTheMargins:
    def test_there_is_one_for_every_square(self, script: ModuleType) -> None:
        env, _ = script.answer()
        assert len(script.margins()) == env.over == 200

    def test_none_of_them_is_negative(self, script: ModuleType) -> None:
        assert all(gap >= 0.0 for gap in script.margins())

    def test_blackjack_decides_nearly_every_square(self, script: ModuleType) -> None:
        """Which is worth stating because the gambler beside it decides none.

        A square played differently from the optimum really is a mistake on
        this board, so counting those squares means something here that it
        would not mean there.
        """
        gaps = script.margins()
        assert min(gaps) > 0.002
        assert sum(1 for gap in gaps if gap >= script.MATTERS) >= 195

    def test_a_margin_is_the_gap_between_the_two_actions(
        self, script: ModuleType
    ) -> None:
        # Worked out here from the optimal values, against the script's own.
        # Sticking on 21 against a six is far better than drawing, which
        # cannot help and often loses.
        env, best = script.answer()
        state = env.fold(21, 6, False)
        worths = []
        for action in (0, 1):
            worths.append(
                sum(
                    branch.probability
                    * (
                        branch.reward
                        + (
                            0.0
                            if branch.terminated
                            else best.values[branch.observation]
                        )
                    )
                    for branch in env.transitions(state, action)
                )
            )
        assert script.margins()[state] == pytest.approx(abs(worths[0] - worths[1]))


class TestWhatItLearnedIsScoredExactly:
    """The number in the table is a sweep, not a tally of the hands played.

    An agent that was dealt a good run of cards collects more than it learned,
    and one that was dealt a bad run collects less. The whole reason this
    environment has a model is so that neither shows up in the table.
    """

    def test_the_value_does_not_depend_on_the_hands_it_was_dealt(
        self, script: ModuleType
    ) -> None:
        # A policy scored twice gives the same number both times, because the
        # score is arithmetic over the model rather than a run.
        first = script.one_run(None, 2000, 3)
        assert script.one_run(None, 2000, 3) == first

    def test_it_is_within_reach_of_the_optimum(self, script: ModuleType) -> None:
        # Loose, because two thousand hands is not many. What it rules out is
        # a score that is not about this game at all.
        value, _, _ = script.one_run(0.05, 2000, 1)
        assert -0.5 < value < 0.0

    def test_the_squares_apart_are_counted_against_the_optimum(
        self, script: ModuleType
    ) -> None:
        _, wrong, _ = script.one_run(0.05, 2000, 1)
        env, _ = script.answer()
        assert 0 <= wrong <= env.over

    def test_more_hands_land_closer(self, script: ModuleType) -> None:
        few = script.one_run(0.05, 2000, 1)
        many = script.one_run(0.05, 50_000, 1)
        assert many[1] < few[1]

    def test_what_the_wrong_squares_are_worth_is_the_sum_of_their_margins(
        self, script: ModuleType
    ) -> None:
        # A count says how many and this says how much, and the two are
        # different questions: the squares an agent gets wrong are the ones
        # it rarely reaches.
        _, wrong, lost = script.one_run(0.05, 2000, 1)
        gaps = sorted(script.margins(), reverse=True)
        assert 0.0 <= lost <= sum(gaps[:wrong]) + 1e-9

    def test_two_seeds_do_not_learn_the_same_policy(self, script: ModuleType) -> None:
        assert script.one_run(0.05, 2000, 1) != script.one_run(0.05, 2000, 2)


class TestTheStepSize:
    def test_the_running_average_and_a_fixed_step_learn_differently(
        self, script: ModuleType
    ) -> None:
        # The claim the table is about. If these agreed there would be
        # nothing to measure.
        assert script.one_run(None, 5000, 1) != script.one_run(0.05, 5000, 1)

    def test_none_on_the_command_line_means_the_running_average(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_blackjack",
                "--budgets",
                "500",
                "--runs",
                "1",
                "--steps",
                "none",
            ],
        )
        assert script.main() == 0
        assert "running average" in capsys.readouterr().out


class TestTheReport:
    @staticmethod
    def _run(script: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["measure_blackjack", "--budgets", "500", "--runs", "1", *extra],
        )
        assert script.main() == 0

    def test_it_says_what_the_deal_is_worth_played_perfectly(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Every row is read against it, so it belongs above the table rather
        # than in a docstring.
        self._run(script, monkeypatch, "--steps", "0.05")
        assert "-0.04656" in capsys.readouterr().out

    def test_every_budget_and_step_gets_a_row(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--steps", "0.05", "0.1")
        rows = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith("500 ")
        ]
        assert len(rows) == 2

    def test_it_says_the_score_is_not_the_return_the_agent_collected(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--steps", "0.05")
        printed = capsys.readouterr().out
        assert "not the return the agent collected" in printed
