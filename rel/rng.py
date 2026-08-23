"""The one source of chance in a run.

Every environment, every exploring agent and every weight initialiser draws
from here. A run is therefore fully described by its seed.

This is PCG-XSH-RR, written out rather than taken from `random.Random`, for
three reasons that all matter to this project.

1. `random.Random` gives the same sequence from the same seed today, but the
   promise covers `random()` and `getrandbits()` and not the methods built on
   them. `sample` changed its draw pattern in Python 3.11, and `shuffle` lost
   its `random` argument in 3.11 as well. A result that only reproduces on one
   minor version is not a reproducible result. This generator is written out
   here, so the promise is the code.
2. Its state cannot be read out and put back without pickling it, so a run
   cannot be stopped and resumed at a known point. The two integers below are
   the whole state, and `snapshot` hands them over.
3. It has one global instance that library code reaches for by habit. One
   `random.random()` in a module that forgot to take a source is enough to make
   a run irreproducible, and nothing reports it. There is no module level
   source here, so that mistake does not compile into a silent fault. It
   becomes a missing argument.

## Streams

`Rng.stream` gives an independent source from the same seed.

This is not a convenience. An environment and an agent both draw chance. If
they share one source, the number of draws the agent makes decides which
numbers the environment gets. Raising epsilon from 0.05 to 0.10 then changes
the wind on the grid as well as the exploration, and the comparison between the
two settings is no longer a comparison of one thing.

Each stream carries a different odd increment, so the two sequences never meet.
`Rng(7).stream("env")` gives the same source whatever the agent did, which is
what lets two agents be run against the same environment dice.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import MutableSequence, Sequence
from typing import TypeVar

T = TypeVar("T")

_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1
_MULTIPLIER = 6364136223846793005

# 2 to the power 53. Every value a float can hold between 0 and 1 fits in this
# many steps, and no more.
_FLOAT_STEPS = 9007199254740992


def _rotate_right(value: int, count: int) -> int:
    """Rotate a thirty two bit value right by `count` places."""
    count &= 31
    if count == 0:
        return value & _MASK32
    return ((value >> count) | (value << (32 - count))) & _MASK32


def _stream_id(name: str) -> int:
    """A stable stream number for a name.

    `hash()` is randomised per process for a string, so it cannot be used here.
    A run that named its streams with `hash()` would draw a different sequence
    on every start, which is the exact fault this module exists to prevent.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class Rng:
    """A seeded source of chance.

    Construct one per run, then take a named stream for each part that draws.
    """

    __slots__ = ("_increment", "_seed", "_spare_normal", "_state")

    def __init__(self, seed: int, stream: int = 1) -> None:
        self._seed = seed

        # The increment selects the stream and has to be odd.
        self._increment = ((stream << 1) | 1) & _MASK64
        self._state = 0
        self._spare_normal: float | None = None

        self._advance()
        self._state = (self._state + seed) & _MASK64
        self._advance()

    @property
    def seed(self) -> int:
        """The seed this source was built from."""
        return self._seed

    def __repr__(self) -> str:
        return f"Rng(seed={self._seed}, increment={self._increment})"

    # -- Deriving sources ---------------------------------------------------

    def stream(self, name: str) -> Rng:
        """An independent source from the same seed.

        The returned source draws a sequence that does not overlap this one, and
        the sequence depends on the seed and the name only. It does not depend
        on how many numbers this source has already drawn, so a caller can take
        a stream at any point and get the same one.
        """
        return Rng(self._seed, _stream_id(name))

    def restart(self) -> Rng:
        """A fresh source on the same seed and stream, back at the beginning."""
        return Rng(self._seed, self._increment >> 1)

    # -- State --------------------------------------------------------------

    def snapshot(self) -> tuple[int, int, int, float | None]:
        """The whole state of this source.

        Hand these four values to `restore` to carry on at exactly this point.
        The last one is the unused half of a normal pair, which the polar
        method produces two at a time. Leaving it out of a snapshot loses one
        draw and shifts every number after it.
        """
        return (self._seed, self._state, self._increment, self._spare_normal)

    @staticmethod
    def restore(
        seed: int, state: int, increment: int, spare_normal: float | None = None
    ) -> Rng:
        """Rebuild a source from a snapshot taken earlier."""
        if increment & 1 == 0:
            raise ValueError("The increment of a PCG stream must be odd.")

        source = Rng(seed, 0)
        source._state = state & _MASK64
        source._increment = increment & _MASK64
        source._spare_normal = spare_normal
        return source

    # -- Drawing ------------------------------------------------------------

    def next_uint32(self) -> int:
        """One step of the generator, returning the next thirty two bits."""
        previous = self._state
        self._advance()

        # Xorshift the high bits down, then rotate by a count taken from the
        # top five bits. The rotation is what removes the weak low order bits
        # that a plain linear congruential generator leaves behind.
        xorshifted = (((previous >> 18) ^ previous) >> 27) & _MASK32
        rotation = previous >> 59

        return _rotate_right(xorshifted, rotation)

    def below(self, bound: int) -> int:
        """A number from 0 up to but not including `bound`.

        The low values would come up slightly more often if the remainder were
        simply taken, so the range that would cause that is rejected and
        redrawn.
        """
        if bound <= 0:
            raise ValueError("The bound must be above zero.")

        threshold = (0x1_0000_0000 - bound) % bound
        while True:
            drawn = self.next_uint32()
            if drawn >= threshold:
                return drawn % bound

    def between(self, lower: int, upper: int) -> int:
        """A number from `lower` up to and including `upper`."""
        if upper < lower:
            raise ValueError("The upper bound must not be below the lower one.")
        return lower + self.below(upper - lower + 1)

    def random(self) -> float:
        """A number from 0 up to but not including 1."""
        high = self.next_uint32() >> 5
        low = self.next_uint32() >> 6
        return ((high << 26) | low) / _FLOAT_STEPS

    def uniform(self, lower: float, upper: float) -> float:
        """A number from `lower` up to but not including `upper`."""
        return lower + (upper - lower) * self.random()

    def chance(self, probability: float) -> bool:
        """True with the given probability."""
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return self.random() < probability

    def normal(self, mean: float = 0.0, deviation: float = 1.0) -> float:
        """A number from a normal distribution.

        This is the Marsaglia polar method. It draws pairs, so the second one
        is kept for the next call. `snapshot` carries that kept value, because
        a resume that dropped it would shift every number after the resume.
        """
        spare = self._spare_normal
        if spare is not None:
            self._spare_normal = None
            return mean + deviation * spare

        while True:
            x = 2.0 * self.random() - 1.0
            y = 2.0 * self.random() - 1.0
            square = x * x + y * y
            if 0.0 < square < 1.0:
                break

        factor = math.sqrt(-2.0 * math.log(square) / square)
        self._spare_normal = y * factor
        return mean + deviation * x * factor

    def pick(self, items: Sequence[T]) -> T:
        """One item, chosen with equal probability."""
        if not items:
            raise ValueError("Cannot pick from an empty sequence.")
        return items[self.below(len(items))]

    def weighted_index(self, weights: Sequence[float]) -> int:
        """An index, chosen with a probability proportional to its weight.

        The weights do not have to add up to one. A weight of zero is never
        chosen. Every weight must be zero or above, and at least one must be
        above zero.
        """
        total = 0.0
        for weight in weights:
            if weight < 0.0:
                raise ValueError("A weight must not be below zero.")
            total += weight

        if total <= 0.0:
            raise ValueError("At least one weight must be above zero.")

        target = self.random() * total
        running = 0.0
        for index, weight in enumerate(weights):
            running += weight
            if target < running:
                return index

        # Floating point addition can leave `running` a hair below `total`.
        # Return the last index with a weight rather than falling off the end.
        for index in range(len(weights) - 1, -1, -1):
            if weights[index] > 0.0:
                return index
        raise AssertionError("unreachable: a positive total implies a positive weight")

    def shuffle(self, items: MutableSequence[T]) -> None:
        """Reorder a sequence in place."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def sample_indices(self, count: int, population: int) -> list[int]:
        """`count` different indices below `population`, in the order drawn.

        This is a partial Fisher-Yates over a dictionary of moved entries, so
        it costs what the sample costs and not what the population costs.
        Drawing `count` different indices out of a million by shuffling the
        million would cost the million.
        """
        if count < 0:
            raise ValueError("The count must not be below zero.")
        if count > population:
            raise ValueError("The count must not be above the population.")

        moved: dict[int, int] = {}
        drawn: list[int] = []
        for i in range(count):
            j = self.between(i, population - 1)
            drawn.append(moved.get(j, j))
            moved[j] = moved.get(i, i)
        return drawn

    # -- Internals ----------------------------------------------------------

    def _advance(self) -> None:
        self._state = (self._state * _MULTIPLIER + self._increment) & _MASK64
