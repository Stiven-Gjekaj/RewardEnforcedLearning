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

## Drawing by priority

`priority` above zero draws a step in proportion to how wrong the agent was
about it, raised to that power. Nothing is more worth refitting than a step the
network cannot predict, and a buffer of a thousand steps the agent already has
right is a thousand gradients of nothing. At zero the draw is uniform and at
one it is straight proportional. Outside that range it is refused: below zero
the draw is backwards, and far above one the smallest priorities are so much
smaller than the largest that most of the buffer is never reached.

The uniform draw is the same code it was before. A uniform draw asks the
generator for a whole number below the size and a weighted one asks for a
fraction of a total, so the two spend different draws, and writing them as one
path would move every run this project has recorded. The branch is here for
that reason.

**A new step is given the largest priority seen.** It has no error yet, because
nothing has been fitted to it, and starting it at zero would let a step enter
the buffer and never be drawn at all.

**A drawn step keeps a floor.** A priority of exactly nothing can never be
drawn again, so an agent that once got a step right would never check it again.
`FLOOR` is small enough not to matter beside a real error and large enough that
everything stays reachable.

**The sample says where each step came from.** `Batch.places` is what
`reprioritise` needs, and without it the priorities could never be told what
the errors turned out to be. That is why the sample is an object and not a
list.

## Why drawing by priority needs a correction

Fitting to a batch is an average over the steps in it, and that average only
estimates what it is meant to estimate if the steps arrived with the right
probability. Drawing by priority changes those probabilities on purpose, so the
average is now of a different thing: the steps with large errors are counted
more often than the buffer says they happen. An agent left like that settles on
a different answer, not a faster route to the same one.

`weighting` is the correction. A step drawn `k` times more often than uniform
counts `k` to the power minus `weighting` as much, so at one the two effects
cancel exactly and at zero there is no correction at all. The trap is real and
`scripts/measure_prioritised.py` measures it against an answer that is known
exactly.

The weights are divided by the largest one the whole buffer could produce, so
every weight is at most one and the correction only ever scales a step down.
Dividing by the largest in the batch instead, which is the usual shortcut,
would give the same step a different weight depending on what happened to be
drawn beside it.

## What priority costs

Drawing by priority adds up the weights of the whole buffer once for each
batch, so a batch of eight out of two thousand costs two thousand additions
rather than eight. A sum per draw instead of per batch would cost eight times
that, and a running total kept across updates would drift. The cumulative sums
are built once and each of the eight draws is a binary search over them.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from rel.agents.base import Transition
from rel.core import ObsT
from rel.rng import Rng

#: The smallest priority a step can be left with. A step at exactly zero could
#: never be drawn again, so a step the agent once got right would never be
#: checked a second time.
FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class Batch(Generic[ObsT]):
    """A drawn sample, and where in the buffer each of its steps came from."""

    #: The steps to learn from.
    steps: tuple[Transition[ObsT, int], ...]

    #: Where each step sits in the buffer, for `Replay.reprioritise`.
    places: tuple[int, ...]

    #: How much each step counts, correcting for how often it was drawn. All
    #: one when the draw was uniform, because a uniform draw needs no
    #: correction.
    weights: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.steps)


class Replay(Generic[ObsT]):
    """The last `capacity` steps, drawn from at random or by priority."""

    __slots__ = (
        "_cursor",
        "_kept",
        "_largest",
        "_weight",
        "capacity",
        "priority",
        "rng",
        "seen",
        "weighting",
    )

    def __init__(
        self,
        rng: Rng,
        capacity: int,
        priority: float = 0.0,
        weighting: float = 0.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("A buffer holds at least one step.")
        if not 0.0 <= priority <= 1.0:
            raise ValueError("A priority power runs from zero to one.")
        if not 0.0 <= weighting <= 1.0:
            raise ValueError("A weighting power runs from zero to one.")
        if weighting > 0.0 and priority <= 0.0:
            raise ValueError("Weighting corrects a priority draw, so it needs one.")

        self.rng = rng
        self.capacity = capacity
        self.priority = priority
        self.weighting = weighting
        self._kept: list[Transition[ObsT, int]] = []
        self._weight: list[float] = []
        self._largest = 1.0
        self._cursor = 0
        #: How many steps have ever been put in, including the ones dropped.
        self.seen = 0

    def add(self, transition: Transition[ObsT, int]) -> None:
        """Put a step in, dropping the oldest once the buffer is full.

        The new step is given the largest priority the buffer has ever held, so
        it is certain to be drawn before anything is known about it.
        """
        self.seen += 1
        if len(self._kept) < self.capacity:
            self._kept.append(transition)
            self._weight.append(self._largest)
            return

        self._kept[self._cursor] = transition
        self._weight[self._cursor] = self._largest
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, size: int) -> Batch[ObsT]:
        """`size` steps drawn with replacement, and where each one sits.

        A buffer holding fewer than `size` steps gives what it has rather than
        refusing. The first few steps of a run are exactly when a buffer is
        short, and an agent that could not learn until it was full would be an
        agent that does nothing for the first thousand steps.
        """
        if size < 1:
            raise ValueError("A sample is at least one step.")
        if not self._kept:
            return Batch((), (), ())

        if self.priority > 0.0:
            places = self._by_priority(size)
            weights = self._weights_for(places)
        else:
            held = len(self._kept)
            places = [self.rng.below(held) for _ in range(size)]
            weights = [1.0] * size

        return Batch(
            tuple(self._kept[place] for place in places),
            tuple(places),
            tuple(weights),
        )

    def reprioritise(self, places: Sequence[int], errors: Sequence[float]) -> None:
        """Tell the buffer how wrong the agent turned out to be about a batch.

        The places come from the `Batch` the errors were measured on, and they
        mean nothing after the next `add`, which can write a different step
        into one of them. Call this before adding again.
        """
        if len(places) != len(errors):
            raise ValueError("Every place needs one error.")

        for place, error in zip(places, errors, strict=True):
            if not 0 <= place < len(self._kept):
                raise IndexError(f"No step sits at place {place}.")
            weight = (abs(error) + FLOOR) ** self.priority
            self._weight[place] = weight
            self._largest = max(self._largest, weight)

    def priorities(self) -> Sequence[float]:
        """The drawing weight of each held step, in the order `steps` gives."""
        return tuple(self._weight)

    def steps(self) -> Sequence[Transition[ObsT, int]]:
        """Everything in the buffer, oldest first is not promised."""
        return tuple(self._kept)

    def __len__(self) -> int:
        return len(self._kept)

    def __repr__(self) -> str:
        return (
            f"Replay({len(self._kept)} of {self.capacity}, "
            f"seen {self.seen}, priority={self.priority:g}, "
            f"weighting={self.weighting:g})"
        )

    # -- Internals -----------------------------------------------------------

    def _by_priority(self, size: int) -> list[int]:
        """`size` places, each drawn in proportion to its weight.

        The cumulative sums are built once for the whole batch, so the cost is
        one pass over the buffer and then a binary search for each draw.
        """
        running: list[float] = []
        total = 0.0
        for weight in self._weight:
            total += weight
            running.append(total)

        last = len(running) - 1
        drawn: list[int] = []
        for _ in range(size):
            # Rounding can put the target at the total itself, which would fall
            # off the end of the sums.
            place = bisect.bisect_right(running, self.rng.random() * total)
            drawn.append(place if place <= last else last)
        return drawn

    def _weights_for(self, places: Sequence[int]) -> list[float]:
        """How much each drawn step counts, at most one and never zero.

        A step drawn more often than uniform counts less, by exactly as much
        at a weighting of one. The divisor is the largest weight the whole
        buffer could produce, which is the one belonging to the least likely
        step, so the answer for a step does not depend on what was drawn
        beside it.
        """
        if self.weighting <= 0.0:
            return [1.0] * len(places)

        smallest = min(self._weight)
        return [(smallest / self._weight[place]) ** self.weighting for place in places]


__all__ = ["FLOOR", "Batch", "Replay"]
