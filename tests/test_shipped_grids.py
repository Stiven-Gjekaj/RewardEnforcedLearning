"""Every built in grid, also written as a file, checked against the original.

The files in `grids/` are what a reader copies to write their own. A file that
had drifted from the environment it names would teach the format wrongly, and
nothing else here would notice: it would still build a grid, and the grid would
still run.

So this compares the two on their models rather than on their text. Two grids
that describe the same branches from every state and action, and carry the same
spec, are the same environment whatever their files look like.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rel.envs import ENVIRONMENTS
from rel.envs.gridfile import read
from rel.envs.gridworld import GridWorld
from rel.rng import Rng

GRIDS = Path(__file__).resolve().parent.parent / "grids"


def shipped() -> list[str]:
    return sorted(path.stem for path in GRIDS.glob("*.txt"))


def built_in() -> list[str]:
    return sorted(
        name
        for name in ENVIRONMENTS.names()
        if isinstance(ENVIRONMENTS.make(name, Rng(1).stream("env")), GridWorld)
    )


def test_there_is_a_file_for_every_built_in_grid() -> None:
    # A grid added to the registry and not written out here would leave the
    # directory looking complete while being short one.
    assert shipped() == built_in()


@pytest.mark.parametrize("name", built_in())
class TestAFileAndALiteralAreTheSameEnvironment:
    @staticmethod
    def _pair(name: str) -> tuple[GridWorld, GridWorld]:
        original = ENVIRONMENTS.make(name, Rng(1).stream("env"))
        assert isinstance(original, GridWorld)
        rebuilt = read(GRIDS / f"{name}.txt").build(Rng(1).stream("env"))
        return original, rebuilt

    def test_the_layout_is_the_same(self, name: str) -> None:
        original, rebuilt = self._pair(name)
        assert rebuilt.layout == original.layout

    def test_the_spec_is_the_same(self, name: str) -> None:
        original, rebuilt = self._pair(name)
        assert rebuilt.spec == original.spec

    def test_the_moves_are_the_same(self, name: str) -> None:
        original, rebuilt = self._pair(name)
        assert rebuilt.moves == original.moves
        assert rebuilt.action_names == original.action_names

    def test_the_wind_is_the_same(self, name: str) -> None:
        original, rebuilt = self._pair(name)
        assert rebuilt.wind == original.wind
        assert rebuilt.wind_varies == original.wind_varies

    def test_the_model_is_the_same(self, name: str) -> None:
        # The strongest check there is. Every branch of every action from
        # every state, which covers the rewards, the slip and the endings all
        # at once and cannot be satisfied by a grid that merely looks right.
        original, rebuilt = self._pair(name)
        for state in range(original.observation_space.n):
            for action in original.action_space:
                assert rebuilt.transitions(state, action) == original.transitions(
                    state, action
                ), f"{name}, state {state}, action {action}"

    def test_the_start_states_are_the_same(self, name: str) -> None:
        original, rebuilt = self._pair(name)
        assert rebuilt.start_states() == original.start_states()


def test_no_shipped_file_has_a_trailing_space() -> None:
    # A layout line is taken exactly as written, so a trailing space is a
    # floor tile that an editor would silently remove. None of these needs
    # one, and a file that did would be a file that cannot be edited safely.
    for path in GRIDS.glob("*.txt"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert line == line.rstrip(), f"{path.name}, line {number}"
