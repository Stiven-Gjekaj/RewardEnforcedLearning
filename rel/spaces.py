"""What an environment accepts and what it hands back.

A space says two things: which values are legal, and how to draw one at
random. That is enough for an agent to work in an environment it knows nothing
else about, and it is what the random policy is built on.

## Why an observation is immutable

An observation from this project is an `int` or a `tuple` of floats, never a
list and never an object with fields that move.

A tabular agent keys its table on the observation. An environment that returns
the same list every step, and edits it in place, gives an agent a key that
changes after the agent stored it. The table then holds one entry that means
several states, and the agent stops learning for a reason that no return value
reports. Both spaces here produce values that cannot be edited, so that fault
cannot happen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar

from rel.rng import Rng

T = TypeVar("T")


class Space(ABC, Generic[T]):
    """The set of values that a slot of an environment accepts."""

    @abstractmethod
    def contains(self, value: object) -> bool:
        """True if the value is a member of this space."""

    @abstractmethod
    def sample(self, rng: Rng) -> T:
        """One member, drawn with equal probability across the space."""

    @abstractmethod
    def __repr__(self) -> str: ...


class Discrete(Space[int]):
    """The whole numbers from `start` up to but not including `start + n`.

    An action space is almost always this. An observation space is this as well
    for every grid in the project, which is what lets a table hold one row for
    each state.
    """

    __slots__ = ("n", "start")

    def __init__(self, n: int, start: int = 0) -> None:
        if n <= 0:
            raise ValueError("A discrete space needs at least one member.")
        self.n = n
        self.start = start

    def contains(self, value: object) -> bool:
        # A bool is an int in Python. Letting `True` pass as action 1 hides a
        # caller that meant to send a flag, so it is refused here.
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return self.start <= value < self.start + self.n

    def sample(self, rng: Rng) -> int:
        return self.start + rng.below(self.n)

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.start + self.n))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Discrete):
            return NotImplemented
        return self.n == other.n and self.start == other.start

    def __hash__(self) -> int:
        return hash((Discrete, self.n, self.start))

    def __repr__(self) -> str:
        if self.start == 0:
            return f"Discrete({self.n})"
        return f"Discrete({self.n}, start={self.start})"


class Box(Space[tuple[float, ...]]):
    """A rectangle in as many dimensions as it has bounds.

    Every continuous observation in the project is one of these. The bounds are
    what a tile coder needs to divide the space, and what a report needs to say
    whether a value is near an edge.
    """

    __slots__ = ("high", "low", "names")

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        names: Sequence[str] | None = None,
    ) -> None:
        if len(low) != len(high):
            raise ValueError("A box needs one lower bound for each upper bound.")
        if not low:
            raise ValueError("A box needs at least one dimension.")
        for lower, upper in zip(low, high, strict=True):
            if not lower < upper:
                raise ValueError("Each lower bound must be below its upper bound.")

        self.low: tuple[float, ...] = tuple(float(value) for value in low)
        self.high: tuple[float, ...] = tuple(float(value) for value in high)

        if names is None:
            self.names: tuple[str, ...] = tuple(
                f"x{index}" for index in range(len(self.low))
            )
        else:
            if len(names) != len(self.low):
                raise ValueError("A box needs one name for each dimension.")
            self.names = tuple(names)

    @property
    def dimensions(self) -> int:
        return len(self.low)

    def contains(self, value: object) -> bool:
        if not isinstance(value, tuple) or len(value) != len(self.low):
            return False
        for element, lower, upper in zip(value, self.low, self.high, strict=True):
            if isinstance(element, bool) or not isinstance(element, (int, float)):
                return False
            if not lower <= element <= upper:
                return False
        return True

    def sample(self, rng: Rng) -> tuple[float, ...]:
        return tuple(
            rng.uniform(lower, upper)
            for lower, upper in zip(self.low, self.high, strict=True)
        )

    def clip(self, value: Sequence[float]) -> tuple[float, ...]:
        """The nearest member of the box to the value given."""
        return tuple(
            min(max(element, lower), upper)
            for element, lower, upper in zip(value, self.low, self.high, strict=True)
        )

    def scaled(self, value: Sequence[float]) -> tuple[float, ...]:
        """The value with each dimension moved onto the range 0 to 1.

        A tile coder and a linear agent both want this. Doing it here keeps the
        bounds in one place, so an environment that widens a bound does not
        leave a divisor behind in an agent.
        """
        return tuple(
            (element - lower) / (upper - lower)
            for element, lower, upper in zip(value, self.low, self.high, strict=True)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        return self.low == other.low and self.high == other.high

    def __hash__(self) -> int:
        return hash((Box, self.low, self.high))

    def __repr__(self) -> str:
        bounds = ", ".join(
            f"{name}=[{lower:g}, {upper:g}]"
            for name, lower, upper in zip(self.names, self.low, self.high, strict=True)
        )
        return f"Box({bounds})"


__all__ = ["Box", "Discrete", "Space"]
