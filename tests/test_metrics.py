"""Tests for the numbers a run reports about itself."""

from __future__ import annotations

import math

import pytest

from rel.metrics import Running, last_mean, moving_average, summarise


class TestRunning:
    def test_the_mean_is_the_mean(self) -> None:
        running = Running()
        for value in (1.0, 2.0, 3.0, 4.0):
            running.add(value)
        assert running.mean == pytest.approx(2.5)
        assert running.count == 4

    def test_the_variance_is_the_sample_variance(self) -> None:
        running = Running()
        for value in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
            running.add(value)
        # The sample variance of that list, which divides by seven.
        assert running.variance == pytest.approx(32.0 / 7.0)
        assert running.deviation == pytest.approx(math.sqrt(32.0 / 7.0))

    def test_it_survives_large_numbers_that_are_close_together(self) -> None:
        # The plain method keeps the sum and the sum of squares and subtracts
        # one from the other, which gives a negative variance for numbers this
        # size. Returns near -13 with a spread of 0.4 are the same shape, at a
        # smaller scale.
        running = Running()
        for value in (1e9 + 4.0, 1e9 + 7.0, 1e9 + 13.0, 1e9 + 16.0):
            running.add(value)
        assert running.variance == pytest.approx(30.0)
        assert running.variance > 0.0

    def test_the_error_shrinks_as_the_run_grows_and_the_deviation_does_not(
        self,
    ) -> None:
        # Two different questions. The deviation says how spread out the
        # episodes are. The error says whether two runs differ.
        short = Running()
        long = Running()
        for index in range(20):
            short.add(float(index % 10))
        for index in range(2000):
            long.add(float(index % 10))

        assert long.deviation == pytest.approx(short.deviation, abs=0.2)
        assert long.error < short.error / 5.0

    def test_one_value_has_no_spread(self) -> None:
        running = Running()
        running.add(5.0)
        assert running.variance == 0.0
        assert running.error == 0.0

    def test_nothing_at_all_reads_as_zero(self) -> None:
        running = Running()
        assert running.mean == 0.0
        assert running.count == 0

    def test_it_keeps_the_smallest_and_the_largest(self) -> None:
        running = Running()
        for value in (3.0, -1.0, 9.0, 2.0):
            running.add(value)
        assert running.smallest == -1.0
        assert running.largest == 9.0

    def test_it_reads_back_something_useful(self) -> None:
        running = Running()
        running.add(2.0)
        assert "count=1" in repr(running)


class TestSummarise:
    def test_it_reports_what_the_list_looked_like(self) -> None:
        summary = summarise([1.0, 2.0, 3.0])
        assert summary.count == 3
        assert summary.mean == pytest.approx(2.0)
        assert summary.smallest == 1.0
        assert summary.largest == 3.0

    def test_an_empty_list_does_not_raise(self) -> None:
        summary = summarise([])
        assert summary.count == 0
        assert summary.mean == 0.0
        assert summary.smallest == 0.0

    def test_it_prints_the_mean_and_the_error(self) -> None:
        assert "over 3" in str(summarise([1.0, 2.0, 3.0]))


class TestMovingAverage:
    def test_the_result_is_as_long_as_the_input(self) -> None:
        # A curve that starts at the window length has a gap at the left, and a
        # reader takes that gap for a run that started late.
        assert len(moving_average([1.0] * 30, 10)) == 30

    def test_it_averages_the_window(self) -> None:
        assert moving_average([0.0, 10.0, 20.0], 2) == [0.0, 5.0, 15.0]

    def test_the_first_positions_average_what_there_is(self) -> None:
        assert moving_average([4.0, 8.0], 10) == [4.0, 6.0]

    def test_a_window_of_one_changes_nothing(self) -> None:
        assert moving_average([1.0, 5.0, 3.0], 1) == [1.0, 5.0, 3.0]

    def test_a_window_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            moving_average([1.0], 0)

    def test_nothing_gives_nothing(self) -> None:
        assert moving_average([], 5) == []


class TestLastMean:
    def test_it_takes_the_tail(self) -> None:
        # The mean over a whole run includes every episode from before the
        # agent knew anything, which answers a different question.
        assert last_mean([0.0] * 90 + [10.0] * 10, 10) == 10.0

    def test_a_count_above_the_length_takes_everything(self) -> None:
        assert last_mean([2.0, 4.0], 100) == 3.0

    def test_nothing_reads_as_zero(self) -> None:
        assert last_mean([], 10) == 0.0
