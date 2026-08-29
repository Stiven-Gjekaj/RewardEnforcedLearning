"""Tests for saying whether a difference between two agents is real.

The class that matters is the last one. A paired permutation test over `n`
seeds has a smallest possible p of `2 / 2**n`, whatever the difference is, and
at five seeds that is 0.0625. **No five seed comparison in this project can
reach 0.05.** Several of its own measurements run five seeds, and that is worth
a test rather than a footnote.

The rest hold the arithmetic to hand computed answers. A permutation test that
was wrong by a factor of two would still print a plausible number.
"""

from __future__ import annotations

import pytest

from rel.compare import (
    EXACT_LIMIT,
    bootstrap_interval,
    compare,
    differences,
    permutation_p,
)
from rel.rng import Rng


class TestPairing:
    def test_it_subtracts_within_each_seed(self) -> None:
        assert differences([3.0, 5.0], [1.0, 1.0]) == [2.0, 4.0]

    def test_different_numbers_of_seeds_are_refused(self) -> None:
        """Not paired, so these functions are the wrong ones and nothing in
        them could tell from the numbers alone."""
        with pytest.raises(ValueError, match="same seeds on both sides"):
            differences([1.0, 2.0], [1.0])

    def test_no_seeds_at_all_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one seed"):
            differences([], [])


class TestThePermutationTest:
    def test_no_difference_at_all_is_certain_of_nothing(self) -> None:
        assert permutation_p([0.0, 0.0, 0.0], Rng(1)) == pytest.approx(1.0)

    def test_it_is_worked_out_by_hand_on_two_seeds(self) -> None:
        """Gaps of 1 and 3. The four sign patterns sum to 4, -2, 2 and -4, and
        two of those are at least 4 away from zero, so the answer is 2/4."""
        assert permutation_p([1.0, 3.0], Rng(1)) == pytest.approx(0.5)

    def test_a_gap_that_changes_sign_is_less_certain(self) -> None:
        together = permutation_p([1.0, 1.0, 1.0, 1.0], Rng(1))
        apart = permutation_p([1.0, -1.0, 1.0, -1.0], Rng(1))
        assert together < apart

    def test_which_agent_came_first_does_not_matter(self) -> None:
        gaps = [1.0, 0.5, 2.0, -0.25, 1.5]
        flipped = [-gap for gap in gaps]
        assert permutation_p(gaps, Rng(1)) == permutation_p(flipped, Rng(1))

    def test_it_never_answers_zero(self) -> None:
        """A test that could would be claiming a certainty no finite number of
        seeds carries. The pattern really seen is always counted."""
        assert permutation_p([100.0] * 10, Rng(1)) > 0.0

    def test_above_the_limit_it_samples_and_says_so(self) -> None:
        gaps = [1.0] * (EXACT_LIMIT + 2)
        answer = compare(gaps, [0.0] * len(gaps), Rng(1))
        assert not answer.exact
        assert answer.p_value > 0.0

    def test_below_the_limit_it_is_exact(self) -> None:
        answer = compare([1.0] * 6, [0.0] * 6, Rng(1))
        assert answer.exact
        assert answer.p_value == pytest.approx(2.0 / 64.0)


class TestTheBootstrapInterval:
    def test_it_brackets_the_difference_it_saw(self) -> None:
        gaps = [1.0, 1.2, 0.8, 1.1, 0.9]
        low, high = bootstrap_interval(gaps, Rng(1))
        assert low <= sum(gaps) / len(gaps) <= high

    def test_a_difference_that_never_varies_has_no_width(self) -> None:
        """Every resample of the same number is the same number."""
        low, high = bootstrap_interval([2.0] * 8, Rng(1))
        assert low == pytest.approx(2.0)
        assert high == pytest.approx(2.0)

    def test_a_wider_confidence_gives_a_wider_interval(self) -> None:
        gaps = [1.0, -0.5, 2.0, 0.25, 1.5, -1.0]
        narrow = bootstrap_interval(gaps, Rng(2), confidence=0.5)
        wide = bootstrap_interval(gaps, Rng(2), confidence=0.99)
        assert wide[0] <= narrow[0] and narrow[1] <= wide[1]

    def test_a_confidence_outside_zero_and_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            bootstrap_interval([1.0], Rng(1), confidence=1.0)

    def test_one_seed_gives_an_interval_of_that_one_number(self) -> None:
        assert bootstrap_interval([3.0], Rng(1)) == pytest.approx((3.0, 3.0))


class TestTheWholeAnswer:
    def test_a_clear_difference_reads_as_one(self) -> None:
        answer = compare([1.0] * 8, [0.0] * 8, Rng(1))
        assert answer.difference == pytest.approx(1.0)
        assert answer.certain
        assert answer.p_value < 0.05

    def test_no_difference_reads_as_none(self) -> None:
        answer = compare([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8], Rng(1))
        assert not answer.certain
        assert answer.p_value > 0.05

    def test_an_interval_across_zero_is_not_certain(self) -> None:
        answer = compare([1.0, -1.0, 1.0, -1.0, 0.5, -0.5], [0.0] * 6, Rng(1))
        assert answer.low < 0.0 < answer.high
        assert not answer.certain

    def test_it_says_what_it_is(self) -> None:
        text = repr(compare([1.0] * 5, [0.0] * 5, Rng(1)))
        assert "difference=1.000" in text
        assert "seeds=5" in text


class TestFiveSeedsCannotReachFivePercent:
    """The finding, and it is about this project rather than about an agent.

    A paired permutation test over `n` seeds has `2 ** n` sign patterns, and
    the two most extreme of them are always at least as far from zero as
    whatever was seen. So the smallest p it can ever report is `2 / 2**n`, and
    that floor does not move however large the difference is.

    Several measurements in this project run five seeds. At five the floor is
    0.0625, so none of them could have reached 0.05 whatever they found.
    """

    @pytest.mark.parametrize(
        ("seeds", "floor"),
        [(3, 0.25), (4, 0.125), (5, 0.0625), (6, 0.03125), (10, 2 / 1024)],
    )
    def test_the_floor_is_two_over_two_to_the_seeds(
        self, seeds: int, floor: float
    ) -> None:
        """An enormous difference, to show the floor is not about the size."""
        assert permutation_p([1e6] * seeds, Rng(1)) == pytest.approx(floor)

    def test_five_seeds_cannot_reach_five_percent(self) -> None:
        assert permutation_p([1e6] * 5, Rng(1)) > 0.05

    def test_six_seeds_is_the_fewest_that_can(self) -> None:
        assert permutation_p([1e6] * 6, Rng(1)) < 0.05

    def test_the_floor_does_not_depend_on_how_large_the_gap_is(self) -> None:
        tiny = permutation_p([1e-9] * 6, Rng(1))
        huge = permutation_p([1e9] * 6, Rng(1))
        assert tiny == pytest.approx(huge)
