"""Numbers about a run, worked out as the run goes.

Nothing here keeps a list in order to average it. A run of ten thousand
episodes is not large, and keeping the list would be fine, but the reason to
avoid it is not memory. It is that a mean taken by adding a list and dividing
loses accuracy as the list grows, and the loss is invisible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class Running:
    """Mean, variance, smallest and largest, in one pass.

    This is Welford's method. The plain approach keeps the sum and the sum of
    squares and subtracts one from the other, which is one line shorter and
    gives a negative variance when the numbers are large and close together.
    Returns near -13 with a spread of 0.4 are exactly that shape.
    """

    __slots__ = ("_mean", "_square_sum", "count", "largest", "smallest")

    def __init__(self) -> None:
        self.count = 0
        self._mean = 0.0
        self._square_sum = 0.0
        self.smallest = math.inf
        self.largest = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        step = value - self._mean
        self._mean += step / self.count
        self._square_sum += step * (value - self._mean)
        self.smallest = min(self.smallest, value)
        self.largest = max(self.largest, value)

    @property
    def mean(self) -> float:
        return self._mean if self.count else 0.0

    @property
    def variance(self) -> float:
        """The sample variance, which needs two values to exist."""
        if self.count < 2:
            return 0.0
        return self._square_sum / (self.count - 1)

    @property
    def deviation(self) -> float:
        return math.sqrt(self.variance)

    @property
    def error(self) -> float:
        """The standard error of the mean.

        This is the number that says whether two runs differ. The deviation
        says how spread out the episodes are, which is a different question and
        does not shrink as a run gets longer.
        """
        if self.count < 2:
            return 0.0
        return self.deviation / math.sqrt(self.count)

    def __repr__(self) -> str:
        return f"Running(count={self.count}, mean={self.mean:.4g})"


@dataclass(frozen=True, slots=True)
class Summary:
    """What a list of numbers looked like."""

    count: int
    mean: float
    deviation: float
    error: float
    smallest: float
    largest: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} +/- {self.error:.3f} over {self.count}"


def summarise(values: Sequence[float]) -> Summary:
    running = Running()
    for value in values:
        running.add(value)
    return Summary(
        count=running.count,
        mean=running.mean,
        deviation=running.deviation,
        error=running.error,
        smallest=running.smallest if running.count else 0.0,
        largest=running.largest if running.count else 0.0,
    )


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """The mean of the last `window` values, at each position.

    The first few positions average what there is rather than nothing, so the
    result is the same length as the input. A curve that starts at the window
    length has a gap at the left, and a reader takes that gap for a run that
    started late.
    """
    if window < 1:
        raise ValueError("A window needs at least one value.")

    smoothed: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        smoothed.append(total / min(index + 1, window))
    return smoothed


def last_mean(values: Sequence[float], count: int) -> float:
    """The mean of the final `count` values, or of all of them if there are fewer.

    This is the number a report gives for "what the agent ended up at". The
    mean over the whole run answers a different question, because it includes
    every episode from before the agent knew anything.
    """
    if not values:
        return 0.0
    tail = values[-count:] if count < len(values) else values
    return sum(tail) / len(tail)


__all__ = ["Running", "Summary", "last_mean", "moving_average", "summarise"]
