"""Two car parks, and a van that moves cars between them overnight.

Sutton and Barto, example 4.2. Jack rents cars from two locations. Renting one
pays 10 and needs a car to be there. Cars come back the next day. Overnight a
van moves up to five cars from one location to the other at 2 each, and the
question is how many to move.

Requests and returns are Poisson, which is the reason this environment is
here. Every other model in this project is a handful of branches written out;
this one is a distribution over how many people turn up, and the model has to
be built rather than listed.

## The states and the actions

A state is how many cars stand at each location at the end of a day, before
the van moves, folded into one number as `first * (capacity + 1) + second`. An
action is how many cars the van then moves, from five to the second location
through zero to five to the first, so the action space is eleven wide and its
middle is doing nothing.

A move that cannot happen is clipped to the largest one that can, in the same
way as `Gambler` clips a stake. There is no car to move that is not there, and
no room at a location that is full. The book lets a car moved to a full
location disappear and charges for it; clipping does not charge for a car that
never arrives, which is the same policy at a different price.

The middle action moves nothing, and it is the only action whose move is
nothing at every state. Several actions clip to nothing at the edges of the
board, so a table of action numbers is not a table of moves, and anything that
reports a policy here reads `moved` rather than the action.

## It never ends

There is no goal and no ruin. The reward is the day's takings and the state is
where the cars are, so this is a continuing task, and running it undiscounted
would ask for the return of a run that goes on for ever. The suggested
discount is 0.9, which is the book's.

## What the model costs

`(capacity + 1)` squared states and eleven actions, and each of those pairs is
a sum over how many cars were asked for and how many came back at each
location. At the book's numbers that is 441 states and 4851 pairs, and the
day's takings at a location depend only on the cars there after the move, so
they are worked out once for each count rather than once for each pair. The
whole model is then about a second rather than about a minute.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import cache

from rel.core import NO_INFO, EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete

#: How much of a Poisson tail to leave out. A count past this is folded into
#: the last one kept, so the branches still add up to one.
TAIL = 1e-6


@cache
def poisson(rate: float, cap: int) -> tuple[float, ...]:
    """The chance of each count from nothing to `cap`, with the tail folded in.

    A Poisson distribution has no largest count and a car park does. Folding
    the tail into the last kept count keeps the branches adding up to one,
    which is what the model contract asks for, and it is right rather than
    merely tidy: more requests than there are cars is the same day as exactly
    as many requests as there are cars.
    """
    shares = []
    for count in range(cap):
        shares.append(math.exp(-rate) * rate**count / math.factorial(count))
    shares.append(max(0.0, 1.0 - sum(shares)))
    return tuple(shares)


class CarRental(TabularEnv):
    """Move cars between two locations overnight, and rent out what is there."""

    def __init__(
        self,
        rng: Rng,
        capacity: int = 20,
        van: int = 5,
        rent: float = 10.0,
        move_cost: float = 2.0,
        asked: tuple[float, float] = (3.0, 4.0),
        returned: tuple[float, float] = (3.0, 2.0),
    ) -> None:
        super().__init__(rng)

        if capacity < 1:
            raise ValueError("A location holds at least one car.")
        if not 0 <= van <= capacity:
            raise ValueError("A van moves between nothing and a full location.")

        self.capacity = capacity
        self.van = van
        self.rent = rent
        self.move_cost = move_cost
        self.asked = asked
        self.returned = returned

        self.side = capacity + 1
        self.observation_space = Discrete(self.side * self.side)
        #: From `van` cars moved to the second location, through nothing, to
        #: `van` moved to the first. The middle action moves nothing.
        self.action_space = Discrete(2 * van + 1)
        self.action_names = tuple(f"{count:+d}" for count in range(-van, van + 1))
        self.spec = EnvSpec(
            name="rental",
            summary=(
                f"Two car parks holding {capacity} each. Renting a car pays "
                f"{rent:g} and the van moves up to {van} overnight at "
                f"{move_cost:g} each."
            ),
            max_episode_steps=100,
            ends=False,
            suggested_discount=0.9,
        )

        # A location's day depends on the cars on it and on nothing else, so
        # both tables are worked out once for each count rather than once for
        # each state and action. At the book's numbers that is 21 entries a
        # side rather than 4851 of them, and it is the difference between a
        # model that takes about a second to build and one that takes a
        # minute.
        self._ending = tuple(
            tuple(
                self._one_day(here, self.asked[side], self.returned[side])
                for here in range(self.side)
            )
            for side in (0, 1)
        )
        self._takings = tuple(
            tuple(
                self._one_takings(here, self.asked[side]) for here in range(self.side)
            )
            for side in (0, 1)
        )

        self.at = self.fold(capacity // 2, capacity // 2)
        self._rented = 0
        self._moved = 0
        self._days = 0

    # -- Reading a state ----------------------------------------------------

    def fold(self, first: int, second: int) -> int:
        """The state number for this many cars at each location."""
        return first * self.side + second

    def unfold(self, state: int) -> tuple[int, int]:
        """How many cars stand at each location in this state."""
        return divmod(state, self.side)

    def moved(self, state: int, action: int) -> int:
        """How many cars the van really moves, positive towards the first.

        Clipped to what can happen: never more cars than are at the location
        they leave, and never more than there is room for where they arrive.
        `Gambler` clips a stake the same way and for the same reason.
        """
        first, second = self.unfold(state)
        want = action - self.van
        if want > 0:
            return min(want, second, self.capacity - first)
        return -min(-want, first, self.capacity - second)

    # -- The model ----------------------------------------------------------

    def _one_day(self, here: int, asked: float, returned: float) -> tuple[float, ...]:
        """Where one location ends the day, given the cars it started it with.

        Indexed by the cars it ends with, and the value is the chance of that
        ending. The takings come back separately because they depend on how
        many were asked for rather than on where the cars ended up.
        """
        wants = poisson(asked, here)
        backs = poisson(returned, self.capacity)

        ending = [0.0] * self.side
        for taken, want in enumerate(wants):
            if want == 0.0:
                continue
            left = here - taken
            for back, chance in enumerate(backs):
                if chance == 0.0:
                    continue
                ending[min(left + back, self.capacity)] += want * chance
        return tuple(ending)

    def _one_takings(self, here: int, asked: float) -> float:
        """What one location is expected to take, with this many cars on it."""
        wants = poisson(asked, here)
        return self.rent * sum(taken * want for taken, want in enumerate(wants))

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        shifted = self.moved(state, action)
        first, second = self.unfold(state)
        first, second = first + shifted, second - shifted

        paid = (
            self._takings[0][first]
            + self._takings[1][second]
            - self.move_cost * abs(shifted)
        )

        left = self._ending[0][first]
        right = self._ending[1][second]

        branches = []
        for ending_first, one in enumerate(left):
            if one == 0.0:
                continue
            for ending_second, other in enumerate(right):
                if other == 0.0:
                    continue
                branches.append(
                    Outcome(
                        one * other,
                        self.fold(ending_first, ending_second),
                        paid,
                        False,
                    )
                )
        return tuple(branches)

    def start_states(self) -> Sequence[tuple[float, int]]:
        half = self.capacity // 2
        return ((1.0, self.fold(half, half)),)

    def terminal_states(self) -> frozenset[int]:
        """None. There is no goal here and no ruin, only another day."""
        return frozenset()

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        half = self.capacity // 2
        self.at = self.fold(half, half)
        self._rented = 0
        self._moved = 0
        self._days = 0
        return self.at

    def _step(self, action: int) -> Step[int]:
        shifted = self.moved(self.at, action)
        first, second = self.unfold(self.at)
        first, second = first + shifted, second - shifted

        paid = -self.move_cost * abs(shifted)
        rented = 0
        ends = []
        for side, here in ((0, first), (1, second)):
            # Drawn from the same tables the model lists, so that stepping and
            # the model cannot drift apart. The request table is already cut
            # at the cars that are there, which is what makes more requests
            # than cars the same day as exactly as many.
            taken = self.rng.weighted_index(poisson(self.asked[side], here))
            back = self.rng.weighted_index(poisson(self.returned[side], self.capacity))
            paid += self.rent * taken
            rented += taken
            ends.append(min(here - taken + back, self.capacity))

        self._days += 1
        self._rented += rented
        self._moved += abs(shifted)

        self.at = self.fold(ends[0], ends[1])
        return Step(observation=self.at, reward=paid, terminated=False, truncated=False)

    # -- What the run really did --------------------------------------------

    def audit(self) -> Mapping[str, float]:
        """Cars rented and cars moved, which the takings roll into one number.

        A day that rents ten cars and moves five pays the same as a day that
        rents nine and moves none. The reward cannot tell those apart and this
        can.
        """
        if self._days == 0:
            return NO_INFO
        return {
            "rented_a_day": self._rented / self._days,
            "moved_a_day": self._moved / self._days,
        }

    def render(self) -> str:
        first, second = self.unfold(self.at)
        return (
            f"first  [{'#' * first}{'.' * (self.capacity - first)}] {first:>3}\n"
            f"second [{'#' * second}{'.' * (self.capacity - second)}] {second:>3}"
        )


__all__ = ["TAIL", "CarRental", "poisson"]
