"""Partial sums that answer a weighted draw in the log of their size.

Drawing one place out of `n` in proportion to `n` weights needs the running
totals of those weights. `rel.agents.replay` builds them by adding the whole
buffer up, once for each batch: a buffer of two thousand costs two thousand
additions whether the batch is one step or eight.

That cost is paid on every update of an agent that learns from a buffer, and
it grows with the buffer while the batch does not. A buffer twenty times
larger is twenty times the work for the same eight steps.

## What a tree holds

The leaves are the weights. Every other cell is the sum of the two below it,
so the root is the total. There are twice as many cells as leaves and each
one is a float.

Changing one weight changes the leaf and every cell above it, which is one
cell per level: `log2(n)` additions rather than nothing, because a scan does
not keep anything between batches. Drawing walks down from the root, going
left when the target fits inside the left child and going right on the
remainder otherwise, which is again one step per level.

So a scan pays `n` per batch and nothing per change, and a tree pays `log2(n)`
per draw and `log2(n)` per change. The tree wins once the buffer is large
enough, and `scripts/measure_tree.py` measures where that is.

## Why the leaves are rounded up to a power of two

A tree over exactly `n` leaves can be packed into `2n` cells, but then the
leaves sit in the array in an order that is not their own: the descent from
the root reaches them rotated. Rotated leaves are fine for a total and wrong
for a draw, because a draw is a search over the leaves *in order*.

Rounding the leaf count up to a power of two keeps the descent and the leaf
order the same thing. It costs up to twice the cells, which for two thousand
floats is nothing worth the other bug.

## A tree does not agree with a scan digit for digit

Both add the same weights and neither adds them in the same order. A scan
accumulates left to right, so the sum before place `k` carries the rounding of
`k` additions in one chain. A tree adds in pairs, so the same sum is assembled
from `log2(n)` subtotals, each rounded separately.

Two different sums of the same numbers can straddle a target. When they do,
one structure returns the place before the boundary and the other returns the
place after it, from the same random number. `scripts/measure_tree.py` counts
how often that happens and checks that every disagreement is a neighbour.
"""

from __future__ import annotations


class Sums:
    """The weights of `size` places, and their totals, kept up to date."""

    __slots__ = ("_cell", "_leaves", "size")

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("A tree holds at least one place.")

        self.size = size
        #: The leaf count rounded up to a power of two, so that the descent
        #: from the root reads the leaves in their own order.
        self._leaves = 1
        while self._leaves < size:
            self._leaves *= 2
        #: Cell 1 is the root, cell `i` has children `2i` and `2i + 1`, and
        #: leaf `place` is cell `self._leaves + place`. Cell 0 is unused, which
        #: is what makes that arithmetic hold.
        self._cell = [0.0] * (2 * self._leaves)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, place: int) -> float:
        """The weight at `place`."""
        if not 0 <= place < self.size:
            raise IndexError(f"No place {place} in a tree of {self.size}.")
        return self._cell[self._leaves + place]

    def __setitem__(self, place: int, weight: float) -> None:
        """Set the weight at `place`, and every total that contains it.

        The cost is one addition per level, so a tree of two thousand pays
        eleven of them rather than rebuilding anything.
        """
        if not 0 <= place < self.size:
            raise IndexError(f"No place {place} in a tree of {self.size}.")
        if weight < 0.0:
            raise ValueError("A weight is not negative.")

        cell = self._leaves + place
        self._cell[cell] = weight
        cell //= 2
        while cell >= 1:
            self._cell[cell] = self._cell[2 * cell] + self._cell[2 * cell + 1]
            cell //= 2

    def total(self) -> float:
        """The sum of every weight, read off the root rather than added up."""
        return self._cell[1]

    def find(self, target: float) -> int:
        """The place whose running total first passes `target`.

        The descent goes left while the target fits inside the left child and
        goes right on what is left over otherwise, so it visits one cell per
        level. A target at or above the total falls off the end, and a target
        below zero falls off the front; both are clamped, because a caller who
        multiplied a fraction by the total can land on either by rounding
        alone.
        """
        if target < 0.0:
            return 0

        cell = 1
        while cell < self._leaves:
            left = 2 * cell
            if target < self._cell[left]:
                cell = left
            else:
                target -= self._cell[left]
                cell = left + 1

        place = cell - self._leaves
        return place if place < self.size else self.size - 1

    def __repr__(self) -> str:
        return f"Sums({self.size} places, total {self.total():g})"


__all__ = ["Sums"]
