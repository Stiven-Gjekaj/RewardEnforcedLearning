"""Turning a handful of real numbers into a list of switches.

A table needs a finite set of states. A cart pole has none: the angle is a real
number and no two steps of a run share one. Something has to group nearby
states together, and how it groups them decides what the agent can tell apart.

## Why not one grid

The obvious answer is one grid: cut each dimension into ten, and the state is
the cell. It works and it generalises badly. Two positions a hair apart on
opposite sides of a boundary share nothing at all, so what the agent learns at
one of them says nothing about the other, and the boundary is invisible in
every plot.

Tile coding lays several grids over the space, each shifted by a fraction of a
cell. A point falls in one cell of each grid, so it turns on one switch per
grid. Two nearby points share most of their switches and differ in a few, which
is generalisation with a dial on it: more grids means finer distinctions from
the same width of cell.

## The offsets

The grids are shifted by an odd number of units each, one odd number per
dimension: 1, 3, 5 and so on. This is the rule from Sutton and Barto, section
9.5.4, where it is credited to Miller and Glanz, and the reason given for it is
that shifting every grid by the same amount in every dimension lines the
boundaries up along a diagonal and generalises unevenly.

That reason was measured here and it did not come out. Run
`python scripts/measure_tiling_offsets.py` to see it. Two points a fixed
distance apart share a number of switches, and how much that number changes
with the direction between them is how uneven the generalisation is. In two
dimensions the odd rule was more even: a spread of 0.60 against 0.90. In four
dimensions, which is the cart pole, it was less even: a spread of 2.53 around a
mean of 1.87, against 1.13 around a mean of 1.40 for the same shift
everywhere.

The rule is kept anyway, and the reason is not that the measurement is wrong.
It is that this is the rule the literature uses, so a result from this project
can be put next to a result from a paper. A different rule would be a second
difference to argue about. What is not kept is the claim: the docstring says
what was measured rather than what is usually said.

## No hashing

The usual implementation hashes the cell into a table of fixed size, and
accepts that two different cells sometimes land on the same entry. That is
necessary when the space is large and worth avoiding when it is not.

Here the index is worked out exactly. Four dimensions at eight cells each,
across eight grids, is fifty two thousand switches: a list of floats that costs
under half a megabyte. Nothing collides, so a fault in an agent is a fault in
the agent.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.spaces import Box


#: The shift of each grid, per dimension, in units of one cell divided by the
#: number of grids. Odd numbers, as the book says.
def _displacement(dimensions: int) -> tuple[int, ...]:
    return tuple(2 * index + 1 for index in range(dimensions))


class TileCoder:
    """Turns a point in a box into the switches that are on for it."""

    __slots__ = ("_cells", "_displacement", "_per_grid", "bins", "box", "grids")

    def __init__(self, box: Box, bins: int = 8, grids: int = 8) -> None:
        if bins < 1:
            raise ValueError("A tile coder needs at least one cell per dimension.")
        if grids < 1:
            raise ValueError("A tile coder needs at least one grid.")

        self.box = box
        self.bins = bins
        self.grids = grids

        # One extra cell along each dimension, because a shifted grid reaches
        # one cell further than the box does.
        self._cells = bins + 1
        self._per_grid = self._cells**box.dimensions
        self._displacement = _displacement(box.dimensions)

    @property
    def features(self) -> int:
        """How many switches there are in total."""
        return int(self.grids * self._per_grid)

    def active(self, observation: Sequence[float]) -> tuple[int, ...]:
        """The switches that are on for this point, one for each grid."""
        scaled = self.box.scaled(self.box.clip(observation))

        switches: list[int] = []
        for grid in range(self.grids):
            index = 0
            for dimension, value in enumerate(scaled):
                shift = grid * self._displacement[dimension] / self.grids
                cell = int(value * self.bins + shift)
                cell = min(max(cell, 0), self._cells - 1)
                index = index * self._cells + cell
            switches.append(grid * self._per_grid + index)
        return tuple(switches)

    def __repr__(self) -> str:
        return (
            f"TileCoder(bins={self.bins}, grids={self.grids}, features={self.features})"
        )


__all__ = ["TileCoder"]
