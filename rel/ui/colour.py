"""Colour for a terminal, switched off when there is no terminal.

Everything here returns a string. Nothing writes anywhere, so a test can ask
for a coloured line and compare it, and a run with its output redirected to a
file writes plain text.

## When colour is off

- `NO_COLOR` is set to anything at all. This is the convention at
  https://no-color.org and it is honoured whatever its value, including an
  empty one.
- `TERM` is `dumb`.
- The output is not a terminal, which is the case for a pipe and for a file.

A caller can also say so directly, which is what the command line does for
`--no-colour`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

RESET = "\x1b[0m"

#: The eight colours every terminal has had since the 1980s. Nothing here uses
#: a 256 colour or a true colour code, because the gain is small and the number
#: of terminals that get them wrong is not.
CODES = {
    "grey": "\x1b[90m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
}

#: The colours a chart hands out to series, in order.
SERIES = ("cyan", "yellow", "magenta", "green", "red", "blue")


def terminal_wants_colour(stream: object = None) -> bool:
    """Whether the terminal this is running in should be sent colour."""
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM") == "dumb":
        return False

    target = stream if stream is not None else sys.stdout
    is_a_terminal = getattr(target, "isatty", None)
    if is_a_terminal is None:
        return False
    return bool(is_a_terminal())


@dataclass(frozen=True, slots=True)
class Palette:
    """Wraps text in colour, or does not."""

    enabled: bool = True

    def paint(self, text: str, colour: str) -> str:
        if not self.enabled or not text:
            return text
        code = CODES.get(colour)
        if code is None:
            raise KeyError(f"There is no colour named {colour!r}.")
        return f"{code}{text}{RESET}"

    def series(self, text: str, index: int) -> str:
        """Paint the text in the colour that belongs to series `index`."""
        return self.paint(text, SERIES[index % len(SERIES)])

    def __bool__(self) -> bool:
        return self.enabled


def palette_for(stream: object = None, *, forced: bool | None = None) -> Palette:
    """The palette to use, deciding from the environment unless told."""
    if forced is not None:
        return Palette(enabled=forced)
    return Palette(enabled=terminal_wants_colour(stream))


def width_of(text: str) -> int:
    """How many columns the text takes, ignoring the colour codes in it."""
    columns = 0
    inside = False
    for character in text:
        if inside:
            if character == "m":
                inside = False
            continue
        if character == "\x1b":
            inside = True
            continue
        columns += 1
    return columns


def pad(text: str, width: int, align: str = "left") -> str:
    """Pad to a width, counting the columns that are really printed.

    `str.ljust` counts the escape codes as characters, so a coloured cell in a
    table comes out several columns short and the column after it walks left.
    """
    gap = width - width_of(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    if align == "centre":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


__all__ = [
    "CODES",
    "RESET",
    "SERIES",
    "Palette",
    "pad",
    "palette_for",
    "terminal_wants_colour",
    "width_of",
]
