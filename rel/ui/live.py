"""A block of lines that is redrawn in place while a run goes on.

The point is a learning curve that grows as it is learned, which is worth
watching and is not worth a scrollback full of it.

## When it is not a terminal

A pipe and a file cannot move a cursor. Writing the escape codes into one puts
them in the file, so this checks first and falls back to printing a single line
every so often. That is what continuous integration sees, and it is why
`--quiet` exists as well: a log with one line every two seconds is still a log
with a thousand lines in it.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import TextIO

#: Move the cursor up one line and clear it.
UP = "\x1b[1A"
CLEAR = "\x1b[2K"
HIDE = "\x1b[?25l"
SHOW = "\x1b[?25h"


class Live:
    """Redraws a block of lines in place."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        enabled: bool | None = None,
        every: float = 0.1,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.every = every

        if enabled is None:
            is_a_terminal = getattr(self.stream, "isatty", None)
            enabled = bool(is_a_terminal and is_a_terminal())
        self.enabled = enabled

        self._drawn = 0
        self._last = 0.0
        self._started = False

    def __enter__(self) -> Live:
        if self.enabled:
            self.stream.write(HIDE)
            self.stream.flush()
        self._started = True
        return self

    def __exit__(self, *_: object) -> None:
        if self.enabled:
            self.stream.write(SHOW)
            self.stream.flush()
        self._started = False

    def update(self, lines: Sequence[str], *, force: bool = False) -> None:
        """Redraw, unless the last redraw was too recent.

        A learning curve does not need sixty frames a second, and drawing one
        for every episode of a fast environment costs more than the run does.
        """
        now = time.monotonic()
        if not force and now - self._last < self.every:
            return
        self._last = now

        if not self.enabled:
            return

        self._erase()
        for line in lines:
            self.stream.write(f"{line}\n")
        self._drawn = len(lines)
        self.stream.flush()

    def finish(self, lines: Sequence[str]) -> None:
        """Draw one last time and leave the block on the screen."""
        if self.enabled:
            self.update(lines, force=True)
        else:
            for line in lines:
                self.stream.write(f"{line}\n")
            self.stream.flush()
        self._drawn = 0

    def _erase(self) -> None:
        if self._drawn == 0:
            return
        self.stream.write(f"{CLEAR}{UP}" * self._drawn)
        self.stream.write(CLEAR)


__all__ = ["CLEAR", "HIDE", "SHOW", "UP", "Live"]
