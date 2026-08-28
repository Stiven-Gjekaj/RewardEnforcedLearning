"""A grid and its settings, written in a text file rather than in Python.

Every grid in this project is already ASCII inside a module. Moving it out
means a reader can add an environment without writing any Python at all, and
the file says what it is:

    name: cliff
    summary: Twelve by four. Stepping off the edge costs a hundred.
    step_reward: -1
    pit_reward: -100

    ............
    ............
    ............
    SXXXXXXXXXXG

## The shape of a file

Everything before the first blank line is the header, and everything after it
is the layout. A file with no blank line in it is all layout and takes every
default.

The header holds `name: value` lines, and a line starting with `#` in it is a
comment. `#` is also the wall tile, which is why the two halves are separated
rather than told apart line by line: a layout that starts with a row of walls
would otherwise be read as a comment.

## Trailing spaces are floor

A layout line is taken exactly as written, because a space is a floor tile and
padding a short line would open a wall that an editor had trimmed. A file whose
rows are not all the same width is refused, and that is the error a trimmed
line gives.

## Why the settings are typed here

`step_reward` is a number, `king_moves` is either true or false and `wind` is
one number per column. A reader that guessed from the text would take
`name: 12` as an integer and hand the environment a name it cannot print.

The table below is the whole surface, and
`tests/test_gridfile.py` checks it against the arguments `GridWorld` really
takes, so a new setting cannot be added there and forgotten here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rel.envs.gridworld import GridWorld
from rel.rng import Rng

TRUE = frozenset({"true", "yes", "on", "1"})
FALSE = frozenset({"false", "no", "off", "0"})


def _bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in TRUE:
        return True
    if lowered in FALSE:
        return False
    raise ValueError(f"{text!r} is not true or false.")


def _maybe(inner: Callable[[str], Any]) -> Callable[[str], Any]:
    """A reader that also accepts `none`, for a setting whose default is one."""

    def read(text: str) -> Any:
        return None if text.lower() in ("none", "null", "") else inner(text)

    return read


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split())


#: Every setting a file may carry, and how to read its value.
SETTINGS: Mapping[str, Callable[[str], Any]] = {
    "name": str,
    "summary": str,
    "step_reward": float,
    "goal_reward": _maybe(float),
    "pit_reward": float,
    "pit_ends_episode": _bool,
    "slip": float,
    "wind": _ints,
    "wind_varies": _bool,
    "king_moves": _bool,
    "can_stay": _bool,
    "max_episode_steps": _maybe(int),
    "solved_return": _maybe(float),
    "suggested_discount": float,
}


class GridFileError(ValueError):
    """A grid file that cannot be read, with the line it went wrong on."""


@dataclass(frozen=True, slots=True)
class GridFile:
    """What a file said, before an environment is made of it."""

    layout: tuple[str, ...]
    settings: Mapping[str, Any]
    where: str = "<text>"

    def build(self, rng: Rng, **override: Any) -> GridWorld:
        """The environment this file describes.

        `override` wins over the file, which is what lets the command line
        change one setting of a grid somebody else wrote.
        """
        return GridWorld(rng, self.layout, **{**self.settings, **override})


def parse(text: str, where: str = "<text>") -> GridFile:
    """Read a grid out of the text of a file."""
    lines = text.splitlines()

    header: list[tuple[int, str]] = []
    body: list[str] = []
    split = None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            split = number
            break
        header.append((number, line))

    if split is None:
        # No blank line, so there is no header and all of it is the layout.
        body = [line for number, line in header]
        header = []
    else:
        body = lines[split:]

    while body and not body[-1].strip():
        body.pop()

    if not body:
        raise GridFileError(f"{where} has a header and no layout under it.")

    settings: dict[str, Any] = {}
    for number, line in header:
        if line.lstrip().startswith("#"):
            continue

        name, colon, value = line.partition(":")
        if not colon:
            raise GridFileError(
                f"{where}, line {number}: a header line is 'name: value' or a "
                f"comment starting with #. {line.strip()!r} is neither."
            )

        name = name.strip()
        reader = SETTINGS.get(name)
        if reader is None:
            raise GridFileError(
                f"{where}, line {number}: {name!r} is not a grid setting. "
                f"Use one of {', '.join(sorted(SETTINGS))}."
            )
        if name in settings:
            raise GridFileError(f"{where}, line {number}: {name!r} is set twice.")

        try:
            settings[name] = reader(value.strip())
        except ValueError as error:
            raise GridFileError(f"{where}, line {number}: {error}") from error

    return GridFile(layout=tuple(body), settings=settings, where=where)


def read(path: str | Path) -> GridFile:
    """Read a grid out of a file."""
    found = Path(path)
    try:
        text = found.read_text(encoding="utf-8")
    except OSError as error:
        raise GridFileError(f"{found} cannot be read: {error}") from error
    return parse(text, where=str(found))


__all__ = ["SETTINGS", "GridFile", "GridFileError", "parse", "read"]
