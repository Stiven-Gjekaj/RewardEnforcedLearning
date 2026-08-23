"""How a number changes as a run goes on.

Exploration and step size both want to start large and finish small. A schedule
puts that decision in one place, so an agent takes a number and never holds a
rule about when to change it.

Every schedule is a function from a count to a value. The count is episodes for
exploration and steps for a step size, and the agent says which it passes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

#: Anything that turns a count into a value.
Schedule = Callable[[int], float]


def constant(value: float) -> Schedule:
    """The same number for the whole run."""

    def schedule(count: int) -> float:
        return value

    return schedule


def linear(start: float, end: float, over: int) -> Schedule:
    """Straight from `start` to `end` across `over` counts, then `end`.

    This is the one to reach for first. It reaches the final value at a point
    that is written down, which an exponential decay does not, so a run can be
    described as "explored for the first two hundred episodes" and that
    statement can be checked.
    """
    if over <= 0:
        raise ValueError("A linear schedule needs a length above zero.")

    def schedule(count: int) -> float:
        if count >= over:
            return end
        return start + (end - start) * (count / over)

    return schedule


def exponential(start: float, half_life: int, floor: float = 0.0) -> Schedule:
    """Halves every `half_life` counts, and never goes below `floor`."""
    if half_life <= 0:
        raise ValueError("A half life must be above zero.")

    def schedule(count: int) -> float:
        return max(floor, start * math.pow(0.5, count / half_life))

    return schedule


def as_schedule(value: float | Schedule) -> Schedule:
    """Take a number or a schedule and give back a schedule.

    Every agent runs its arguments through this, so a caller writes
    `epsilon=0.1` for the common case and hands a schedule in for the rest.
    """
    if callable(value):
        return value
    return constant(float(value))


__all__ = ["Schedule", "as_schedule", "constant", "exponential", "linear"]
