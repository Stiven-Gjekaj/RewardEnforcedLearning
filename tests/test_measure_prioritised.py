"""The measurement behind the priority sampling table on the algorithms page.

The load bearing test is `TestTheAnswerItCompilesAgainst`. The whole first
section of that script rests on one claim of arithmetic: the constant an
uncorrected priority draw is pulled towards is the root of a particular sum,
and it is not the mean. If the bisection that finds that root were wrong, the
section would still print a table and the table would still show a gap, and
nothing else would notice.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_prioritised.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_prioritised", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheAnswerItCompilesAgainst:
    def test_the_root_makes_the_sum_zero(self, script: ModuleType) -> None:
        rewards = (0.0,) * 15 + (1.0,) * 5
        root = script.uncorrected_answer(rewards)
        pull = sum(abs(root - y) * (root - y) for y in rewards)
        assert pull == pytest.approx(0.0, abs=1e-9)

    def test_it_matches_the_closed_form_of_this_split(self, script: ModuleType) -> None:
        # Fifteen at zero and five at one gives 15c^2 = 5(1 - c)^2, so
        # c = 1 / (1 + sqrt 3).
        wanted = 1.0 / (1.0 + 3.0**0.5)
        assert script.uncorrected_answer((0.0,) * 15 + (1.0,) * 5) == pytest.approx(
            wanted
        )

    def test_a_symmetric_set_puts_the_root_at_the_mean(
        self, script: ModuleType
    ) -> None:
        # And so shows nothing, which is why the rewards in the script are
        # skewed on purpose.
        rewards = (0.0, 1.0, 2.0, 3.0, 4.0)
        assert script.uncorrected_answer(rewards) == pytest.approx(
            statistics.mean(rewards)
        )

    def test_the_root_is_above_the_mean_when_the_tail_is_above_it(
        self, script: ModuleType
    ) -> None:
        rewards = (0.0,) * 15 + (1.0,) * 5
        assert script.uncorrected_answer(rewards) > statistics.mean(rewards)

    def test_the_root_is_below_the_mean_when_the_tail_is_below_it(
        self, script: ModuleType
    ) -> None:
        rewards = (1.0,) * 15 + (0.0,) * 5
        assert script.uncorrected_answer(rewards) < statistics.mean(rewards)

    def test_the_rewards_it_uses_are_skewed(self, script: ModuleType) -> None:
        mean = statistics.mean(script.REWARDS)
        assert script.uncorrected_answer(script.REWARDS) != pytest.approx(mean)


class TestWhatItMeasures:
    def test_the_middle_setting_is_the_mistake(self, script: ModuleType) -> None:
        # It draws by priority and corrects nothing. A script whose three rows
        # were all corrected would show that priority helps and would never
        # show why the correction is there.
        for settings in (script.SETTINGS, script.EXACT_SETTINGS):
            names = [name for name, _, _ in settings]
            assert names == ["even", "priority", "corrected"]
            assert settings[0][1] == 0.0
            assert settings[1][1] > 0.0 and settings[1][2] == 0.0
            assert settings[2][1] > 0.0 and settings[2][2] > 0.0

    def test_the_settling_rows_use_the_full_powers(self, script: ModuleType) -> None:
        # The first section is about where the fit lands and not how fast it
        # gets there, so the correction it applies is the whole one. Anything
        # less would land between the two answers and prove neither.
        assert script.EXACT_SETTINGS[2][1] == 1.0
        assert script.EXACT_SETTINGS[2][2] == 1.0

    def test_an_uncorrected_draw_settles_off_the_mean(self, script: ModuleType) -> None:
        rewards = (0.0,) * 15 + (1.0,) * 5
        mean = statistics.mean(rewards)
        even = script.where_it_settles(1, 0.0, 0.0, rewards, 200)
        wrong = script.where_it_settles(1, 1.0, 0.0, rewards, 200)
        assert abs(even - mean) < 0.05
        assert wrong - mean > 0.05

    def test_the_correction_brings_it_back(self, script: ModuleType) -> None:
        rewards = (0.0,) * 15 + (1.0,) * 5
        mean = statistics.mean(rewards)
        right = script.where_it_settles(1, 1.0, 1.0, rewards, 200)
        assert abs(right - mean) < 0.05


class TestWhatItPrints:
    @pytest.fixture
    def printed(self, script: ModuleType, capsys: pytest.CaptureFixture[str]) -> str:
        script.settling_section((0.0,) * 6 + (1.0,) * 2, 40, 2)
        return capsys.readouterr().out

    def test_it_names_both_answers(self, printed: str) -> None:
        assert "the mean of the rewards" in printed
        assert "pulled instead" in printed

    def test_it_has_a_row_for_each_setting(self, printed: str) -> None:
        for name in ("even", "priority", "corrected"):
            assert name in printed

    def test_it_says_how_far_off_the_mean_each_landed(self, printed: str) -> None:
        assert "off the mean" in printed
        assert "+" in printed or "-" in printed
