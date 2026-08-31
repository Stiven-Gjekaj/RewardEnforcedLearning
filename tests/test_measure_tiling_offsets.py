"""The measurement behind the tile coder offsets table on the algorithms page.

The load bearing tests are `TestTheControlIsReallyDifferent` and
`TestTheBrokenCoderIsReallyBroken`. Both tables compare the tile coder this
project ships against a coder written in this script to stand for what it
would otherwise be, and a stand-in that turned out to be the same thing would
make both tables read as a finding about a rule when they were a finding about
nothing.

That is not a hypothetical. The four dimensional row of the first table once
said the rule made things worse, and the cause was a fault in the shipped
coder rather than in the rule.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.agents.tiles import TileCoder
from rel.spaces import Box

SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "measure_tiling_offsets.py"
)

SQUARE = Box([0.0, 0.0], [1.0, 1.0])


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_tiling_offsets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cells_of(coder: TileCoder, point: tuple[float, float]) -> list[tuple[int, int]]:
    """The cell each grid puts the point in, one pair for each grid.

    A tile coder answers one switch per grid and the switch carries the cells
    of every dimension folded together. Unfolding them is what lets a test ask
    where a boundary in one dimension falls against a boundary in the other,
    which is the whole of what the two ways of shifting disagree about.
    """
    side = coder.bins + 1
    return [
        divmod(switch - grid * side**2, side)
        for grid, switch in enumerate(coder.active(point))
    ]


class TestTheControlIsReallyDifferent:
    """`SameShift` stands for a coder that shifts every grid alike."""

    def test_it_shifts_every_dimension_by_the_same_amount(
        self, script: ModuleType
    ) -> None:
        # Which is the whole of what it is for, and it shows on the diagonal:
        # both dimensions read the same number, so both land in the same cell
        # of every grid.
        control = script.SameShift(SQUARE, bins=8, grids=8)
        for step in range(101):
            for first, second in cells_of(control, (step / 100, step / 100)):
                assert first == second

    def test_the_shipped_coder_does_not(self, script: ModuleType) -> None:
        # The control against the thing it is a control for. Each dimension
        # gets its own odd displacement here, so on the diagonal the two
        # dimensions fall in different cells of some grids.
        shipped = TileCoder(SQUARE, bins=8, grids=8)
        apart = sum(
            first != second
            for step in range(101)
            for first, second in cells_of(shipped, (step / 100, step / 100))
        )
        assert apart > 100

    def test_it_answers_differently_from_the_shipped_coder(
        self, script: ModuleType
    ) -> None:
        # The one thing that has to be true for the table to say anything.
        # Swept rather than asked at a point: the two agree at about three
        # points in ten, and the first version of this test picked one.
        shipped = TileCoder(SQUARE, bins=8, grids=8)
        control = script.SameShift(SQUARE, bins=8, grids=8)
        agreed = sum(
            shipped.active((across / 50, up / 50))
            == control.active((across / 50, up / 50))
            for across in range(51)
            for up in range(51)
        )
        assert 0 < agreed < 51 * 51 / 2

    def test_it_still_turns_on_one_switch_for_each_grid(
        self, script: ModuleType
    ) -> None:
        # A control that answered a different number of switches would be
        # compared on the count of them rather than on where they fall.
        control = script.SameShift(SQUARE, bins=8, grids=8)
        assert len(control.active((0.31, 0.62))) == control.grids

    def test_its_switches_are_its_own_and_no_grid_shares_one(
        self, script: ModuleType
    ) -> None:
        control = script.SameShift(SQUARE, bins=8, grids=8)
        switches = control.active((0.31, 0.62))
        assert len(set(switches)) == len(switches)


class TestTheBrokenCoderIsReallyBroken:
    """`NoModulo` stands for the shipped coder before the fault was fixed."""

    def test_it_answers_differently_from_the_shipped_coder(
        self, script: ModuleType
    ) -> None:
        box = Box([0.0] * 4, [1.0] * 4)
        shipped = TileCoder(box, bins=8, grids=8)
        broken = script.NoModulo(box, bins=8, grids=8)
        assert shipped.active((0.5,) * 4) != broken.active((0.5,) * 4)

    def test_the_first_grid_is_the_one_place_they_agree(
        self, script: ModuleType
    ) -> None:
        # Grid zero has no shift at all, so the modulo has nothing to do to
        # it. A stand-in that differed there would be a different coder
        # rather than the same one without the fix.
        box = Box([0.0] * 4, [1.0] * 4)
        shipped = TileCoder(box, bins=8, grids=8)
        broken = script.NoModulo(box, bins=8, grids=8)
        assert shipped.active((0.37,) * 4)[0] == broken.active((0.37,) * 4)[0]

    def test_the_later_grids_reach_far_fewer_cells(self, script: ModuleType) -> None:
        """The claim the second table makes, at a size a test can afford.

        The reported run draws two hundred thousand points. This draws four
        thousand, so the counts are smaller, and what it checks is the shape:
        the last grid of the broken coder reaches a fraction of what the
        fixed one does.
        """
        box = Box([0.0] * 4, [1.0] * 4)
        fixed = script.reach(TileCoder(box, bins=8, grids=8), 4, 4000)
        broken = script.reach(script.NoModulo(box, bins=8, grids=8), 4, 4000)
        assert broken[-1] < fixed[-1] / 2

    def test_the_fixed_coder_keeps_every_grid_about_the_same(
        self, script: ModuleType
    ) -> None:
        # Every grid divides the same box the same number of ways, so with
        # the same points drawn against each they should reach about as many
        # cells. The broken one does not, which is what the table shows.
        box = Box([0.0] * 4, [1.0] * 4)
        counts = script.reach(TileCoder(box, bins=8, grids=8), 4, 4000)
        assert min(counts[1:]) > 0.9 * max(counts[1:])


class TestReach:
    def test_no_grid_reaches_more_cells_than_it_has(self, script: ModuleType) -> None:
        coder = TileCoder(SQUARE, bins=8, grids=8)
        for count in script.reach(coder, 2, 2000):
            assert count <= coder._per_grid

    def test_more_draws_never_reach_fewer_cells(self, script: ModuleType) -> None:
        # The count is of cells at least one point landed in, so it rises
        # with the draws. That is why the script names the number it drew.
        coder = TileCoder(SQUARE, bins=8, grids=8)
        few = script.reach(coder, 2, 200)
        many = script.reach(coder, 2, 2000)
        assert sum(many) > sum(few)

    def test_one_draw_reaches_one_cell_of_each_grid(self, script: ModuleType) -> None:
        coder = TileCoder(SQUARE, bins=8, grids=8)
        assert script.reach(coder, 2, 1) == [1] * coder.grids


class TestSpread:
    def test_the_mean_is_between_none_and_every_grid(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(script, "DIRECTIONS", 20)
        monkeypatch.setattr(script, "ORIGINS", 5)
        coder = TileCoder(SQUARE, bins=8, grids=8)
        mean, width = script.spread(coder, 2, 0.75 / 8)
        assert 0.0 <= mean <= coder.grids
        assert width >= 0.0

    def test_a_point_shares_every_switch_with_itself(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A radius of nothing moves nowhere, so both points of every pair are
        # the same point and every switch is shared. That is the one value of
        # this measurement that is known without measuring it.
        monkeypatch.setattr(script, "DIRECTIONS", 10)
        monkeypatch.setattr(script, "ORIGINS", 3)
        coder = TileCoder(SQUARE, bins=8, grids=8)
        assert script.spread(coder, 2, 0.0) == (float(coder.grids), 0.0)

    def test_points_a_whole_box_apart_share_nothing(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(script, "DIRECTIONS", 10)
        monkeypatch.setattr(script, "ORIGINS", 3)
        coder = TileCoder(SQUARE, bins=8, grids=8)
        mean, _ = script.spread(coder, 2, 5.0)
        assert mean == 0.0

    def test_it_replays(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The seed is inside the function rather than handed in, so two calls
        # draw the same directions and the same starts.
        monkeypatch.setattr(script, "DIRECTIONS", 10)
        monkeypatch.setattr(script, "ORIGINS", 3)
        coder = TileCoder(SQUARE, bins=8, grids=8)
        assert script.spread(coder, 2, 0.1) == script.spread(coder, 2, 0.1)


class TestTheReport:
    def test_it_prints_both_tables(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(script, "DIRECTIONS", 10)
        monkeypatch.setattr(script, "ORIGINS", 3)
        monkeypatch.setattr(script, "REACH_DRAWS", 500)
        assert script.main() == 0

        printed = capsys.readouterr().out
        assert "odd displacement" in printed
        assert "same shift" in printed
        assert "without the modulo" in printed
        assert "with it" in printed

    def test_it_says_how_many_points_the_reach_was_drawn_from(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The count of cells rises with the draws, so a row of counts with no
        # number of draws beside it is not a number a reader can check.
        monkeypatch.setattr(script, "DIRECTIONS", 10)
        monkeypatch.setattr(script, "ORIGINS", 3)
        monkeypatch.setattr(script, "REACH_DRAWS", 500)
        script.main()
        assert "500 random" in capsys.readouterr().out
