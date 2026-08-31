"""Features that are waves over the whole box rather than patches of it.

A tile coder cuts the box into cells and a radial basis puts bumps at places.
Both are local: a point lights the features near it and nothing else, and a
weight learned in one corner says nothing about the far one.

A Fourier basis is the opposite. Every feature covers the whole box, and what
tells them apart is how fast each one waves:

    feature c   cos(pi * c . x)

where `x` is the point with each dimension moved onto zero to one, and `c` is a
vector of whole numbers between zero and the order. The one with `c` all zeros
is the constant, the ones with a single one are the slowest waves, and the rest
are faster in one dimension or another.

## What it costs and what it does not need

`(order + 1)` to the power of the dimensions, so a two dimensional box at order
three is sixteen features and a four dimensional one is 256. That is the same
growth a radial basis has and worse than a tile coder's.

What it does not need is anything else. No bins, no widths, no centres, no
offsets between grids. An order is the whole design, and that is the reason it
is worth having beside two encoders whose settings are the thing a reader has
to get right.

## Why the step size is divided for each feature

The feature with `c` all zeros is a constant and the feature with `c` of eight
waves eight times across the box. A step that suits the first is far too large
for the second: the same change to its weight moves the value up and down eight
times as often, so the value it contributes swings rather than settles.

The usual answer, and the one this uses, is to divide the step for each feature
by the length of its `c`. `step_scales` is what a coder says about that and
what `rel.agents.linear` asks it. Every other coder here returns nothing and
gets one step for every feature, which is what it always had.

**It is measured rather than copied.** `scripts/measure_fourier.py` runs the
same problem with the division and without it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

from rel.agents.linear import Encoded
from rel.spaces import Box


class FourierBasis:
    """Cosine waves over a box, one for each vector of whole numbers."""

    def __init__(self, box: Box, order: int = 3) -> None:
        if order < 0:
            raise ValueError("An order of waves is zero or more.")

        self.box = box
        self.order = order

        #: Every vector of whole numbers up to the order, lowest dimension
        #: moving fastest, so the constant is first and the order is the last.
        self.coefficients: tuple[tuple[int, ...], ...] = tuple(
            tuple(reversed(one))
            for one in product(range(order + 1), repeat=box.dimensions)
        )
        self._indices = tuple(range(len(self.coefficients)))
        self._scales = tuple(
            1.0 if not any(c) else 1.0 / math.sqrt(sum(value * value for value in c))
            for c in self.coefficients
        )

    @property
    def features(self) -> int:
        """How many weights an agent needs for each action."""
        return len(self.coefficients)

    def encode(self, observation: Sequence[float]) -> Encoded:
        """Every wave at this point, which is every feature it has.

        Dense, unlike a tile coder. There is no such thing as a wave that is
        off at a point, so the indices are all of them and the work of a step
        is the whole basis.
        """
        point = self.box.scaled(self.box.clip(observation))
        return self._indices, tuple(
            math.cos(
                math.pi
                * sum(weight * value for weight, value in zip(c, point, strict=True))
            )
            for c in self.coefficients
        )

    def squared_length(self, values: Sequence[float]) -> float:
        """The features of a point dotted with themselves, for the step size.

        Never below one, because the constant feature is one everywhere. An
        agent divides its step size by this, so a coder whose features could
        all be near zero at some point would hand that point a step of near
        infinity. This one cannot.
        """
        return sum(value * value for value in values)

    def step_scales(self) -> Sequence[float] | None:
        """A multiplier on the step size of each feature.

        One over the length of the feature's own vector, and one for the
        constant. A wave that crosses the box eight times moves the value
        eight times as often for the same change to its weight, so it takes a
        step an eighth as large to move it as far.
        """
        return self._scales

    def starting_weight(self, optimism: float) -> float:
        """The weight that makes a point nothing is known about worth this.

        There is none, unless it is nothing. The constant feature could carry
        an optimistic value on its own and the waves could not, and a coder
        answers with one number for all of its features. So this answers zero
        and refuses anything else, rather than returning a number that is
        right at the corner of the box and wrong everywhere else.
        """
        if optimism == 0.0:
            return 0.0
        raise ValueError(
            "A Fourier basis has no single weight that makes every point "
            "worth the same. The constant feature could carry an optimistic "
            "value and the waves could not. Start at nothing."
        )

    def __repr__(self) -> str:
        # The class rather than the name, so a `FlatSteps` says it is one. An
        # agent's own repr is its name and its coder's, and a report that
        # called the two of these by the same name would make the run that
        # was measured unreadable from the number it produced.
        return (
            f"{type(self).__name__}(order={self.order}, "
            f"dimensions={self.box.dimensions}, features={self.features})"
        )


class FlatSteps(FourierBasis):
    """The same waves with one step size for all of them.

    Here to be run against `FourierBasis` rather than to be used. The division
    by the speed of each wave is the one thing the literature says about this
    basis that is not the basis itself, and a project that copied it without
    running the other way round would not know what it bought.

    `scripts/measure_fourier.py` runs both.
    """

    def step_scales(self) -> Sequence[float] | None:
        return None


__all__ = ["FlatSteps", "FourierBasis"]
