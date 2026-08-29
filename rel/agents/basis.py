"""A second way of turning a point in a box into features.

A tile coder answers "which cells is this point in" and every answer is a
switch: on or off, and exactly one per grid. Radial basis features answer "how
close is this point to each of these places", and every answer is a number
between zero and one.

The difference is worth building a second encoder for, because it is the
difference between a step and a slope. Two points on opposite sides of a tile
boundary share nothing along that dimension, however close together they are. A
radial basis has no boundaries: a point a hair away has a value a hair
different, everywhere.

## Where the centres go

On a grid, `bins` of them along each dimension, evenly spaced from one bound to
the other. That is `bins` raised to the number of dimensions, which is 64 in
two and 4096 in four, and is the reason this is a demonstration rather than the
default: the count grows exactly as badly as the naive single grid the tile
coder exists to avoid.

## How wide each one is

The width is a multiple of the spacing between centres, and the multiple is the
one setting that decides everything about the encoder. Too narrow and a point
between two centres lights neither, so the agent learns nothing about it. Too
wide and every centre answers about the same, so the features carry no
information about where the point is.

`docs/algorithms.md` measures it rather than asserting it.

## The nearest few can be kept, and by default are not

A radial basis is dense: every centre answers every point, so a value is a sum
over every feature and an update touches every weight. At 4096 centres that is
a hundred times the work of a tile coder's eight switches, for the same
problem. So `kept` drops all but the largest few values.

It is off by default. The reason is that measuring it contradicted the two
sentences that used to stand here.

The first said the dropped values were near zero. They are not. At the default
width, eight of thirty six centres carry 74 percent of the total, and dividing
those eight by their own smaller total moves the largest of them by 0.05, which
is a quarter of itself.

The second said this encoder has no boundaries. With `kept` on it has them.
Between two centres there is a point where the smallest kept value and the
largest dropped one cross, and on the two sides of it the agent reads a
different weight for the same feature. The jump there is exactly the size of
the smallest kept value: at the default width and eight kept it is 0.067, where
the largest feature of that same point is 0.183.

That is smaller than a tile coder's boundary, which swaps a whole switch out of
eight, and it is the same kind of thing. A boundary is what this encoder was
built not to have, so it is not there unless a caller asks for it.
`docs/algorithms.md` measures what asking costs.

## They are normalised

The kept values are divided by their sum, so they add to one. Without it the
total activation rises and falls as a point moves between centres, and the
value of a state would rise and fall with it for no reason connected to the
task. Dividing also makes the step size mean what it means everywhere else in
this project, which is the share of the error the value moves by.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

from rel.spaces import Box


class RadialBasis:
    """Turns a point in a box into how close it is to each of some centres."""

    __slots__ = ("_centres", "_divisor", "bins", "box", "kept", "width")

    def __init__(
        self, box: Box, bins: int = 6, width: float = 1.0, kept: int | None = None
    ) -> None:
        if bins < 2:
            raise ValueError("A radial basis needs at least two centres a side.")
        if width <= 0.0:
            raise ValueError("A width is above zero.")
        if kept is not None and kept < 1:
            raise ValueError("A basis that keeps no centre says nothing at all.")

        self.box = box
        self.bins = bins
        self.width = width
        self.kept = kept

        # The centres live in the scaled box, which is the unit cube, so one
        # spacing is the same number in every dimension however wide the real
        # bounds are. A basis over the cart pole would otherwise be round in
        # the units of the angle and flat in the units of the position.
        spacing = 1.0 / (bins - 1)
        self._centres: tuple[tuple[float, ...], ...] = tuple(
            product([index * spacing for index in range(bins)], repeat=box.dimensions)
        )

        # Twice the squared spread, worked out once because it divides the
        # squared distance to every centre on every step.
        self._divisor = 2.0 * self.spread * self.spread
        if self._divisor <= 0.0:
            # A width small enough that its square is zero would divide by
            # zero below. The answer would be right and unreachable: the value
            # of every centre is zero at any distance at all.
            raise ValueError("A width this small leaves nothing to divide by.")

    @property
    def spread(self) -> float:
        """How far a centre reaches, in the units of the scaled box.

        The spacing between neighbouring centres, times the width. At exactly
        this distance from a centre its value is `exp(-0.5)`, which is about
        0.61, in the same way as for any other bell.
        """
        return self.width / (self.bins - 1)

    @property
    def features(self) -> int:
        """How many centres there are in total."""
        return len(self._centres)

    def squared_distances(self, observation: Sequence[float]) -> list[float]:
        """How far the point is from every centre, squared, in the scaled box.

        Its own method because `encode` needs it twice over: once to work out
        the values, and once more when every one of them has underflowed to
        zero and the distances are all that is left to choose by.
        """
        scaled = self.box.scaled(self.box.clip(observation))

        distances: list[float] = []
        for centre in self._centres:
            squared = 0.0
            for value, middle in zip(scaled, centre, strict=True):
                gap = value - middle
                squared += gap * gap
            distances.append(squared)
        return distances

    def all_values(self, observation: Sequence[float]) -> list[float]:
        """How close the point is to every centre, before anything is dropped.

        The exponential of minus the squared distance over twice the squared
        spread, which is the usual bell. Nothing here divides by anything else:
        this is what the centres say, and `encode` decides what to do with it.
        """
        divisor = self._divisor
        return [
            math.exp(-squared / divisor)
            for squared in self.squared_distances(observation)
        ]

    def encode(
        self, observation: Sequence[float]
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Which features are on for this point, and how strongly.

        Two parallel tuples rather than pairs, because the caller walks them
        together in an inner loop and pairs would allocate a tuple for each.
        """
        distances = self.squared_distances(observation)
        divisor = self._divisor
        values = [math.exp(-squared / divisor) for squared in distances]

        if self.kept is not None and self.kept < len(values):
            order = sorted(range(len(values)), key=values.__getitem__, reverse=True)
            chosen = sorted(order[: self.kept])
        else:
            chosen = [index for index, value in enumerate(values) if value > 0.0]

        total = sum(values[index] for index in chosen)
        if total <= 0.0:
            # Every centre underflowed, which means the width is far too narrow
            # for the spacing. One feature at full strength is a worse answer
            # than a good width and a better one than a row of zeros, which
            # would leave the agent unable to tell this state from any other.
            #
            # The nearest centre has to be found by distance and not by value.
            # The values are all exactly zero here, so the largest of them is
            # the first of them, which is the corner of the box and is the
            # nearest centre to one point in it. The first version of this
            # returned that corner for every point.
            nearest = min(range(len(distances)), key=distances.__getitem__)
            return (nearest,), (1.0,)

        return tuple(chosen), tuple(values[index] / total for index in chosen)

    def squared_length(self, values: Sequence[float]) -> float:
        """The features of a point, dotted with themselves.

        A step size is divided by this, so that it means the share of the
        error the value moves by, which is what it means everywhere else in
        the project. `TileCoder.squared_length` has the arithmetic.

        Here the answer changes from point to point, because the values do. A
        point sitting on a centre puts almost everything into one feature and
        the answer is near one; a point equally far from many centres spreads
        it out and the answer is near one over their count. So the step the
        weights take is larger where the features are spread out, by exactly
        the factor that keeps the value moving by the same share of the error.
        """
        return sum(value * value for value in values)

    def starting_weight(self, optimism: float) -> float:
        """The weight that makes a state nothing is known about worth this.

        The optimism itself. The values of a point add to one, so a weight of
        `optimism` in every feature makes the value `optimism` times one.
        """
        return optimism

    def __repr__(self) -> str:
        return (
            f"RadialBasis(bins={self.bins}, width={self.width:g}, "
            f"kept={self.kept}, features={self.features})"
        )


__all__ = ["RadialBasis"]
