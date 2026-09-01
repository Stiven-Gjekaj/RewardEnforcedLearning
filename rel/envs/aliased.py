"""Three cells that all look the same, one of which has its actions reversed.

Sutton and Barto, example 13.1. Every environment before this one hands the
agent an observation that names the state. This one does not: the agent sees
the same thing everywhere, and one of the places it cannot tell apart works
backwards.

## The shape

    start   0 --- 1 --- 2 --- goal
                  ^
                  the actions here are swapped

Right moves right and left moves left, except in the middle cell where right
moves left and left moves right. Left at the start walks into a wall and stays.
Every step pays -1, so the return is minus the number of steps.

**The observation is the same number in all three cells.** A tabular agent
therefore has one row of action values for the whole corridor, and a policy
network has one set of weights that cannot depend on where it is standing.

## Why this environment is here

Both deterministic policies never finish.

    always right   0 goes to 1, 1 goes back to 0, for ever
    always left    the start's wall, for ever

An agent that ranks its actions and takes the best one is choosing between
those two, and the best it can do is spend its exploration on the other one.
Nothing about the ranking is wrong. The problem is that the answer is not a
ranking.

The best a policy of this shape can do is take right with probability
`2 - sqrt 2`, which is 0.5858, and that reaches the goal in `6 + 4 sqrt 2`
steps, which is 11.6569. Both are exact and `best_share` and `best_steps`
return them.

## Where the arithmetic comes from

Write `J(s)` for the expected steps from cell `s` under a policy that goes
right with probability `p`, and `q` for `1 - p`.

    J(0) = 1 + p J(1) + q J(0)
    J(1) = 1 + p J(0) + q J(2)
    J(2) = 1 + q J(1)

The first gives `J(0) = 1/p + J(1)`. Substituting that into the second and the
third into it as well leaves `J(1) (1 - p - q squared) = 2 + q`, and
`1 - p - q squared` is `q - q squared`, which is `p q`. So

    J(0) = 2 (2 - p) / (p (1 - p))

`steps_from_start` is that expression. It runs off to infinity at both ends,
which is the two deterministic policies never finishing, and its smallest value
is where `p squared - 4 p + 2` is zero.

## What an epsilon-greedy agent can reach

An agent that ranks its actions and explores with probability `epsilon` takes
its favourite `1 - epsilon / 2` of the time. That is a policy of this shape
with `p` at one end of the range rather than in the middle, and
`steps_from_start` says what it costs. At an epsilon of 0.1 the best of the two
is 44.2 steps against the 11.7 the best policy of this shape reaches.

**That gap is not an agent tuned badly.** It is the whole family of methods
that produce a ranking, measured against the best thing a policy without a
ranking can do.

## It is not a `TabularEnv`

Every other small environment here writes out its model so that dynamic
programming can answer it exactly. This one cannot: dynamic programming works
over states, the agent is given observations, and here they are not the same
thing. Writing a model over the observation would be writing down the very
confusion the environment exists to demonstrate.

The exact answer is here instead, as arithmetic over the true cells, and
`tests/test_aliased.py` checks it against both a solve of the three equations
and a run of the environment itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from rel.core import DiscreteEnv, EnvSpec, Step
from rel.rng import Rng
from rel.spaces import Discrete

#: The two actions. `RIGHT` moves right everywhere but the middle cell.
LEFT, RIGHT = 0, 1

#: The cell whose actions are swapped, and the one the goal sits past.
SWITCHED = 1
CELLS = 3

#: The one observation. Every cell gives it, which is the point.
ANYWHERE = 0


class AliasedCorridor(DiscreteEnv[int]):
    """A corridor whose cells cannot be told apart, with one of them reversed."""

    def __init__(self, rng: Rng, steps: int = 1000) -> None:
        super().__init__(rng)

        self.observation_space = Discrete(1)
        self.action_space = Discrete(2)
        self.action_names = ("left", "right")
        self.spec = EnvSpec(
            name="aliased",
            summary=(
                "Three cells that look the same and one of them works "
                "backwards. No fixed choice ever finishes."
            ),
            max_episode_steps=steps,
            suggested_discount=1.0,
        )

        self.at = 0
        self._reached = 0.0

    # -- The answer ---------------------------------------------------------

    @staticmethod
    def steps_from_start(share: float) -> float:
        """Expected steps to the goal, going right with probability `share`.

        `2 (2 - p) / (p (1 - p))`, worked out in the module docstring. It runs
        off to infinity at both ends, which is the two fixed choices never
        finishing.
        """
        if not 0.0 < share < 1.0:
            raise ValueError("A fixed choice never reaches the goal at all.")
        return 2.0 * (2.0 - share) / (share * (1.0 - share))

    @staticmethod
    def best_share() -> float:
        """The probability of going right that finishes soonest.

        The root of `p squared - 4 p + 2` inside the range, which is
        `2 - sqrt 2`. The other root is above one and is not a policy.
        """
        return 2.0 - math.sqrt(2.0)

    @classmethod
    def best_steps(cls) -> float:
        """How long the best policy of this shape takes, which is `6 + 4 sqrt 2`."""
        return 6.0 + 4.0 * math.sqrt(2.0)

    @classmethod
    def best_ranking_share(cls, epsilon: float) -> float:
        """The best an agent that ranks its two actions can reach.

        Such an agent takes its favourite `1 - epsilon / 2` of the time, so its
        share of going right is at one end of the range or the other. This
        returns whichever end finishes sooner.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("An exploring share runs from zero to one.")
        favourite = 1.0 - epsilon / 2.0
        other = epsilon / 2.0
        if cls.steps_from_start(favourite) <= cls.steps_from_start(other):
            return favourite
        return other

    # -- What the reward does not say ---------------------------------------

    def audit(self) -> Mapping[str, float]:
        """Whether the episode reached the goal or ran out of steps.

        The return cannot say. Every step pays -1, so an episode that walked
        into the goal on its thousandth step and one that was cut off at a
        thousand are both worth -1000.
        """
        return {"reached": self._reached}

    # -- Acting -------------------------------------------------------------

    def _reset(self) -> int:
        self.at = 0
        self._reached = 0.0
        return ANYWHERE

    def _step(self, action: int) -> Step[int]:
        forward = action == RIGHT
        if self.at == SWITCHED:
            forward = not forward

        if forward:
            self.at += 1
        elif self.at > 0:
            self.at -= 1

        if self.at >= CELLS:
            self._reached = 1.0
            return Step(ANYWHERE, -1.0, True, False)
        return Step(ANYWHERE, -1.0, False, False)

    def render(self) -> str:
        cells = "".join("A" if cell == self.at else "." for cell in range(CELLS))
        return f"{cells}| goal, the middle one is reversed"

    def __repr__(self) -> str:
        return f"AliasedCorridor(at cell {self.at})"


def aliased_corridor(rng: Rng) -> AliasedCorridor:
    """The corridor as the book states it."""
    return AliasedCorridor(rng)


__all__ = [
    "ANYWHERE",
    "CELLS",
    "LEFT",
    "RIGHT",
    "SWITCHED",
    "AliasedCorridor",
    "aliased_corridor",
]
