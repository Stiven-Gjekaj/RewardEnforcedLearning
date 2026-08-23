"""Tests for turning real numbers into switches."""

from __future__ import annotations

from itertools import pairwise

import pytest

from rel.agents.tiles import TileCoder
from rel.spaces import Box

UNIT = Box([0.0, 0.0], [1.0, 1.0])


class TestShape:
    def test_one_switch_for_each_grid(self) -> None:
        coder = TileCoder(UNIT, bins=8, grids=8)
        assert len(coder.active((0.5, 0.5))) == 8

    def test_every_switch_is_inside_the_range(self) -> None:
        coder = TileCoder(UNIT, bins=6, grids=4)
        for x in range(11):
            for y in range(11):
                for switch in coder.active((x / 10, y / 10)):
                    assert 0 <= switch < coder.features

    def test_no_two_grids_share_a_switch(self) -> None:
        # Each grid owns its own block of numbers. If two grids could produce
        # the same number, one weight would be updated twice for one point and
        # the step size would silently double.
        coder = TileCoder(UNIT, bins=8, grids=8)
        switches = coder.active((0.37, 0.61))
        assert len(set(switches)) == len(switches)

    def test_the_number_of_switches_is_what_it_says(self) -> None:
        coder = TileCoder(UNIT, bins=8, grids=4)
        assert coder.features == 4 * 9 * 9

    def test_a_coder_needs_a_grid_and_a_cell(self) -> None:
        with pytest.raises(ValueError, match="at least one grid"):
            TileCoder(UNIT, bins=4, grids=0)
        with pytest.raises(ValueError, match="at least one cell"):
            TileCoder(UNIT, bins=0, grids=4)


class TestBehaviour:
    def test_the_same_point_gives_the_same_switches(self) -> None:
        coder = TileCoder(UNIT, bins=8, grids=8)
        assert coder.active((0.3, 0.7)) == coder.active((0.3, 0.7))

    def test_two_points_far_apart_share_nothing(self) -> None:
        coder = TileCoder(UNIT, bins=8, grids=8)
        near = set(coder.active((0.05, 0.05)))
        far = set(coder.active((0.95, 0.95)))
        assert not near & far

    def test_two_points_close_together_share_almost_everything(self) -> None:
        # This is the whole reason for tile coding. One grid would give these
        # two points either everything or nothing, with the answer depending on
        # which side of a line they fell.
        coder = TileCoder(UNIT, bins=8, grids=8)
        here = set(coder.active((0.4, 0.4)))
        there = set(coder.active((0.401, 0.401)))
        assert len(here & there) >= 7

    def test_sharing_falls_away_as_the_points_separate(self) -> None:
        coder = TileCoder(UNIT, bins=8, grids=8)
        tile = 1.0 / 8
        counts = [
            len(
                set(coder.active((0.4, 0.4)))
                & set(coder.active((0.4 + step * tile / 8, 0.4)))
            )
            for step in range(9)
        ]
        assert counts[0] == 8
        assert counts[-1] == 0
        assert all(later <= earlier for earlier, later in pairwise(counts))

    def test_a_point_outside_the_box_lands_on_the_edge(self) -> None:
        # The cart pole reports a state past the end of its tiling box on the
        # step that ends the episode. Clipping puts it in the nearest cell,
        # which is where the agent already has weights.
        coder = TileCoder(UNIT, bins=8, grids=8)
        assert coder.active((-9.0, 9.0)) == coder.active((0.0, 1.0))

    def test_each_grid_has_its_own_boundary(self) -> None:
        # Every grid at the same offset would make the extra grids pointless:
        # they would all switch at the same place, and the coder would be one
        # grid with its numbers multiplied by eight.
        #
        # Walking across one tile crosses one boundary of each grid, so the
        # switches take a different value that many times.
        coder = TileCoder(UNIT, bins=4, grids=8)
        tile = 1.0 / 4
        seen = {coder.active((step * tile / 200, 0.5)) for step in range(200)}
        assert len(seen) >= coder.grids


class TestEveryGridKeepsItsResolution:
    """A grid has to be able to reach all of its own cells.

    Each grid is shifted by a fraction of a cell. The shift used to be taken
    without a modulo, so the later grids were shifted by several whole cells,
    and the cells they were shifted past did not exist. Those points were
    clamped into the last cell instead, so a whole band of the box near one
    edge shared a single weight in most of the grids.

    Nothing about a run reports this. The agent still learns, and it learns
    with a coder that has quietly lost most of its resolution.

    This check is exhaustive rather than sampled: two dimensions with eight
    cells has eighty one cells per grid, and a walk over the box at two
    hundred steps a side reaches every one of them.
    """

    @staticmethod
    def _cells_reached(coder: TileCoder) -> list[int]:
        reached: list[set[int]] = [set() for _ in range(coder.grids)]
        for i in range(201):
            for j in range(201):
                for grid, switch in enumerate(coder.active((i / 200, j / 200))):
                    reached[grid].add(switch)
        return [len(found) for found in reached]

    def test_every_grid_reaches_every_cell_it_owns(self) -> None:
        coder = TileCoder(UNIT, bins=8, grids=8)
        counts = self._cells_reached(coder)
        assert counts == [81] * 8, counts

    def test_no_grid_is_smaller_than_the_unshifted_one(self) -> None:
        # The claim that matters, stated so that it survives a change to the
        # number of cells or grids.
        coder = TileCoder(UNIT, bins=6, grids=8)
        counts = self._cells_reached(coder)
        assert min(counts) == max(counts), counts


class TestABoxThatIsNotTheUnitSquare:
    def test_the_bounds_are_taken_from_the_box(self) -> None:
        box = Box([-1.2, -0.07], [0.6, 0.07])
        coder = TileCoder(box, bins=8, grids=8)

        assert coder.active((-1.2, -0.07)) != coder.active((0.6, 0.07))
        # The two ends of the box are as far apart as two points can be.
        assert not set(coder.active((-1.2, -0.07))) & set(coder.active((0.6, 0.07)))

    def test_a_dimension_with_a_tiny_range_still_divides(self) -> None:
        box = Box([0.0, -0.07], [1.0, 0.07])
        coder = TileCoder(box, bins=8, grids=8)
        low = set(coder.active((0.5, -0.07)))
        high = set(coder.active((0.5, 0.07)))
        assert not low & high
