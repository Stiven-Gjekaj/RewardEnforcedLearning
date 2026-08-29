"""Saying whether a difference between two agents is real, and how large.

`rel compare` prints a mean and a standard error for each agent. That is a
description of two sets of numbers and not a comparison of them, and reading it
as one is how a difference that is nothing gets written down as a result.

Two questions it does not answer:

    is it real     would two agents that are the same have looked this
                   different, on this many seeds
    how large      what is the difference, with an interval around it

Both answers are here, both need nothing but arithmetic, and both are exact
enough on the number of seeds this project runs to be worth trusting.

## The comparison is paired

Every agent in a `rel compare` meets the same seeds. Seed 4 hands both of them
the same environment, so a difference on seed 4 is a difference between the
agents rather than between the problems they were given.

That is worth a great deal here. The seeds of this project vary far more than
the agents do: on the cliff walk one agent's ten seeds run from -20 to -545,
and a comparison that threw the pairing away would be looking for a difference
of two inside a spread of five hundred. Subtracting within each seed removes
the part both agents shared.

**Pairing is an assumption about how the runs were made, not a setting.** If
two agents were run on different seeds these functions are the wrong ones, and
nothing in them can tell.

## The permutation test

Under the claim that the two agents are the same, the label on each of a pair
is arbitrary: seed 4 could as easily have given the difference the other way
round. So flip the sign of each difference every possible way, take the mean
each time, and count how often that mean is at least as far from zero as the
one really seen. That share is the p value.

With `n` seeds there are `2 ** n` sign patterns, which is 1024 at ten seeds and
a million at twenty. Below `EXACT_LIMIT` every one of them is enumerated and
the answer is exact. Above it they are sampled, and the result carries the
sampling error of that rather than being exact, which `Comparison` says.

## The bootstrap interval

Resample the seeds with replacement, take the mean difference of each resample,
and read the interval off the percentiles. This says how large the difference
is, which the p value does not: a difference can be certain and tiny.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rel.rng import Rng

#: Above this many seeds the permutation test is sampled rather than
#: enumerated. Ten seeds is 1024 patterns and twenty is over a million, and the
#: point where enumerating stops being instant is somewhere between.
EXACT_LIMIT = 16

#: How many resamples a bootstrap interval is taken over.
RESAMPLES = 2000


@dataclass(frozen=True, slots=True)
class Comparison:
    """What one agent is worth against another, on the seeds they shared."""

    #: The mean of `first - second` over the paired seeds.
    difference: float
    #: The interval around that difference, from the bootstrap.
    low: float
    high: float
    #: How often sign flipping produced a mean at least this far from zero.
    p_value: float
    #: How many seeds the two agents shared.
    seeds: int
    #: Whether the p value enumerated every sign pattern or sampled them.
    exact: bool

    @property
    def certain(self) -> bool:
        """Whether the interval stays on one side of zero.

        Not a verdict and not a threshold. It is the one thing a reader takes
        from an interval at a glance, and having it as a property keeps every
        caller from writing the same comparison out again.
        """
        return self.low > 0.0 or self.high < 0.0

    def __repr__(self) -> str:
        return (
            f"Comparison(difference={self.difference:.3f}, "
            f"[{self.low:.3f}, {self.high:.3f}], p={self.p_value:.3f}, "
            f"seeds={self.seeds})"
        )


def differences(first: Sequence[float], second: Sequence[float]) -> list[float]:
    """One number per seed: what the first agent got, less the second.

    Both sequences have to be the seeds in the same order, because that is the
    whole of what makes this paired.
    """
    if len(first) != len(second):
        raise ValueError(
            f"A paired comparison needs the same seeds on both sides. "
            f"One has {len(first)} and the other has {len(second)}."
        )
    if not first:
        raise ValueError("A comparison needs at least one seed.")
    return [one - other for one, other in zip(first, second, strict=True)]


def permutation_p(gaps: Sequence[float], rng: Rng, samples: int = 10_000) -> float:
    """How often the labels being arbitrary would have looked like this.

    Under the claim that the two agents are the same, the sign of each paired
    difference is arbitrary. This counts the sign patterns whose mean is at
    least as far from zero as the one seen, including the one seen, which is
    what keeps the answer from ever being zero: a test that can report a p of
    zero is claiming a certainty no finite number of seeds carries.
    """
    if not gaps:
        raise ValueError("A comparison needs at least one seed.")

    # The sum rather than the mean, because the count is the same for every
    # sign pattern and dividing by it every time would change nothing.
    total = abs(sum(gaps))
    count = len(gaps)

    if count <= EXACT_LIMIT:
        at_least = 0
        for pattern in range(2**count):
            flipped = sum(
                -gap if pattern >> index & 1 else gap for index, gap in enumerate(gaps)
            )
            if abs(flipped) >= total - 1e-12:
                at_least += 1
        return at_least / float(2**count)

    at_least = 1
    for _ in range(samples):
        flipped = sum(-gap if rng.chance(0.5) else gap for gap in gaps)
        if abs(flipped) >= total - 1e-12:
            at_least += 1
    return at_least / (samples + 1)


def bootstrap_interval(
    gaps: Sequence[float],
    rng: Rng,
    confidence: float = 0.95,
    resamples: int = RESAMPLES,
) -> tuple[float, float]:
    """The interval the mean difference sits in, by resampling the seeds.

    Seeds are drawn with replacement, which is what makes this a statement
    about the seeds that might have been run rather than about the ones that
    were.
    """
    if not gaps:
        raise ValueError("A comparison needs at least one seed.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("A confidence is between 0 and 1, and is neither.")

    count = len(gaps)
    means = sorted(
        sum(gaps[rng.below(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )

    tail = (1.0 - confidence) / 2.0
    return _percentile(means, tail), _percentile(means, 1.0 - tail)


def _percentile(sorted_values: Sequence[float], share: float) -> float:
    """The value at this share of the way through, interpolating between two.

    Interpolating rather than taking the nearest, because with two thousand
    resamples the nearest is fine and the interpolation costs one line. What it
    buys is that the answer moves smoothly as the resample count changes, which
    makes a table of intervals at different counts readable.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]

    place = share * (len(sorted_values) - 1)
    below = math.floor(place)
    above = min(below + 1, len(sorted_values) - 1)
    weight = place - below
    return sorted_values[below] * (1.0 - weight) + sorted_values[above] * weight


def compare(
    first: Sequence[float],
    second: Sequence[float],
    rng: Rng,
    confidence: float = 0.95,
) -> Comparison:
    """The whole answer: how large the difference is and whether it is real."""
    gaps = differences(first, second)
    low, high = bootstrap_interval(gaps, rng, confidence=confidence)
    return Comparison(
        difference=sum(gaps) / len(gaps),
        low=low,
        high=high,
        p_value=permutation_p(gaps, rng),
        seeds=len(gaps),
        exact=len(gaps) <= EXACT_LIMIT,
    )


__all__ = [
    "EXACT_LIMIT",
    "RESAMPLES",
    "Comparison",
    "bootstrap_interval",
    "compare",
    "differences",
    "permutation_p",
]
