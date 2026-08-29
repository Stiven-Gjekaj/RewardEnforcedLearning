"""A fixed number of remembered steps, and a way to draw from them.

Learning from each step once and in the order it happened has two problems that
a table does not have and a network does.

The steps are correlated. Twenty steps of a cart pole falling to the left are
twenty samples of almost the same thing, and a network fitted to them in a row
forgets what it knew about falling to the right. A table cannot forget that
way: every cell is its own number and moving one does not move another.

The steps are also thrown away. One real step of a network is one gradient
step, and the network has capacity to learn far more from it than that.

A buffer answers both. It holds the last `capacity` steps and hands back a
sample of them, so a batch mixes experience from far apart in time and every
step is used more than once.

## Why not a deque

A deque of a thousand drops the oldest for free and reads the middle in linear
time, and drawing a batch of eight reads the middle eight times a step. A list
with a cursor writes and reads in constant time and drops the oldest by writing
over it, which is the same behaviour and the right cost.

## Drawing with replacement

One draw can come up twice in a batch. Refusing that would need a second pass
and would change the distribution being sampled, and neither is worth it for a
buffer of a thousand and a batch of eight.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic

from rel.agents.base import Transition
from rel.core import ObsT
from rel.rng import Rng


class Replay(Generic[ObsT]):
    """The last `capacity` steps, drawn from at random."""

    __slots__ = ("_cursor", "_kept", "capacity", "rng", "seen")

    def __init__(self, rng: Rng, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("A buffer holds at least one step.")

        self.rng = rng
        self.capacity = capacity
        self._kept: list[Transition[ObsT]] = []
        self._cursor = 0
        #: How many steps have ever been put in, including the ones dropped.
        self.seen = 0

    def add(self, transition: Transition[ObsT]) -> None:
        self.seen += 1
        if len(self._kept) < self.capacity:
            self._kept.append(transition)
            return

        self._kept[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, size: int) -> list[Transition[ObsT]]:
        """`size` steps drawn at random, with replacement.

        A buffer holding fewer than `size` steps gives what it has rather than
        refusing. The first few steps of a run are exactly when a buffer is
        short, and an agent that could not learn until it was full would be an
        agent that does nothing for the first thousand steps.
        """
        if size < 1:
            raise ValueError("A sample is at least one step.")
        if not self._kept:
            return []
        return [self._kept[self.rng.below(len(self._kept))] for _ in range(size)]

    def steps(self) -> Sequence[Transition[ObsT]]:
        """Everything in the buffer, oldest first is not promised."""
        return tuple(self._kept)

    def __len__(self) -> int:
        return len(self._kept)

    def __repr__(self) -> str:
        return f"Replay({len(self._kept)} of {self.capacity}, seen {self.seen})"


__all__ = ["Replay"]
