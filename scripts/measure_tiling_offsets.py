#!/usr/bin/env python3
"""Measure how evenly a tile coder generalises, for two ways of shifting grids.

The rule in the literature shifts each grid by an odd number of units, one odd
number per dimension. The reason given is that shifting every grid by the same
amount in every dimension lines the boundaries up along a diagonal, so two
points that are the same distance apart share very different numbers of
switches depending on the direction between them.

This script measures that. It takes pairs of points a fixed distance apart, in
many directions, counts how many switches each pair shares, and reports how
much that count moves with the direction. A small spread means even
generalisation.

Run it with:

    python scripts/measure_tiling_offsets.py

The result is quoted in `rel/agents/tiles.py` and in `docs/algorithms.md`.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.tiles import TileCoder
from rel.rng import Rng
from rel.spaces import Box

RADIUS_IN_TILES = 0.75
DIRECTIONS = 200
ORIGINS = 30


class SameShift(TileCoder):
    """A tile coder that shifts every grid by the same amount everywhere."""

    def active(self, observation: Sequence[float]) -> tuple[int, ...]:
        scaled = self.box.scaled(self.box.clip(observation))
        switches: list[int] = []
        for grid in range(self.grids):
            index = 0
            for value in scaled:
                cell = int(value * self.bins + grid / self.grids)
                cell = min(max(cell, 0), self.bins)
                index = index * (self.bins + 1) + cell
            switches.append(grid * (self.bins + 1) ** self.box.dimensions + index)
        return tuple(switches)


def spread(coder: TileCoder, dimensions: int, radius: float) -> tuple[float, float]:
    """The mean shared count over directions, and how far it moves."""
    rng = Rng(1)
    per_direction: list[float] = []

    for _ in range(DIRECTIONS):
        raw = [rng.normal() for _ in range(dimensions)]
        length = math.sqrt(sum(value * value for value in raw))
        step = [radius * value / length for value in raw]

        total = 0
        for _ in range(ORIGINS):
            origin = [0.3 + 0.4 * rng.random() for _ in range(dimensions)]
            here = set(coder.active(origin))
            there = set(
                coder.active([o + s for o, s in zip(origin, step, strict=True)])
            )
            total += len(here & there)
        per_direction.append(total / ORIGINS)

    mean = sum(per_direction) / len(per_direction)
    return mean, max(per_direction) - min(per_direction)


def main() -> int:
    print(
        f"Shared switches between points {RADIUS_IN_TILES} tiles apart, "
        f"over {DIRECTIONS} directions and {ORIGINS} start points.\n"
    )
    print(f"{'dimensions':>11}  {'offsets':<18} {'mean':>6} {'spread':>7}")

    for dimensions in (2, 4):
        box = Box([0.0] * dimensions, [1.0] * dimensions)
        radius = RADIUS_IN_TILES / 8
        for label, coder in (
            ("odd displacement", TileCoder(box, bins=8, grids=8)),
            ("same shift", SameShift(box, bins=8, grids=8)),
        ):
            mean, width = spread(coder, dimensions, radius)
            print(f"{dimensions:>11}  {label:<18} {mean:6.2f} {width:7.2f}")

    print(
        "\nA smaller spread is more even generalisation, which is what the odd\n"
        "displacement rule is for. It gives that in both.\n"
        "\n"
        "This script once reported the opposite in four dimensions. The cause\n"
        "was a fault in TileCoder rather than in the rule: the grid shift was\n"
        "not taken modulo one cell, so most of each later grid lay outside the\n"
        "cells allocated to it and was clamped into one. See the note in\n"
        "TileCoder.active."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
