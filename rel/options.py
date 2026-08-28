"""An action that lasts several steps, and how to build one from a model.

Every agent in this project chooses one primitive action and is told what
happened. On the four rooms grid that means the value of the goal has to be
carried back one cell at a time, across two rooms, through a doorway one cell
wide.

An option is the other thing an agent can choose. It is a policy, a rule for
when it stops, and the set of states it can start from. "Go to the doorway on
your left" is one, and choosing it once covers the twenty steps that follow.

## The three parts

The policy says which action to take while the option runs. The stopping rule
says when it hands control back. The states it can start from are the states
its policy has an answer for, which is one set rather than two and so cannot
disagree with itself.

An option always takes at least one step. The stopping rule is tested at the
state it arrives in, never at the state it started in, so an option whose
stopping rule holds everywhere is a single action wearing this shape.

## Where the hallway options come from

`GridWorld.gaps` reads the doorways off the layout. Take those out of the grid
and what is left falls into rooms, and a doorway that touches two of them is a
hallway. For each hallway and each room it touches there is one option: from
anywhere in that room, walk to that hallway and stop.

Nothing here is written down for the four rooms grid in particular. The same
construction on the Dyna maze builds no options at all, because its gaps are
passages inside one room rather than doors between two, and on the cliff walk
there is nothing to find because there is no wall.

## A stochastic grid

The policy is built by walking backwards from the hallway over the model, and
where an action can land in more than one place the most likely landing is the
one followed. On a slippery grid the option is then a plan that the environment
will not always carry out, which is the ordinary case for an option and not a
fault in one.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from rel.spaces import Discrete


@dataclass(frozen=True, slots=True)
class Option:
    """A policy, a rule for when it stops, and the states it can start from.

    Not hashable: the policy is a mapping. Options are held in a list and
    referred to by their place in it.
    """

    name: str
    policy: Mapping[int, int]
    stops: frozenset[int]

    def can_start(self, state: int) -> bool:
        """Whether this option has an answer here."""
        return state in self.policy

    def act(self, state: int) -> int:
        """The action to take while this option is running."""
        action = self.policy.get(state)
        if action is None:
            raise ValueError(f"{self.name} has no action for state {state}.")
        return action

    def stops_at(self, state: int) -> bool:
        """Whether the option hands control back on arriving here.

        A state the policy has no answer for stops it. That is what makes an
        option that walks out of its own room end there rather than carry on
        with an action chosen for somewhere else.
        """
        return state not in self.policy or state in self.stops

    def __len__(self) -> int:
        """How many states this option can start from."""
        return len(self.policy)

    def __repr__(self) -> str:
        return f"Option({self.name!r}, over {len(self.policy)} states)"


def primitive(action: int, states: Collection[int], name: str = "") -> Option:
    """One action, everywhere, stopping after every step.

    This is the shape of a primitive action written as an option, and it is
    what lets an agent over options hold both kinds in one table.
    """
    return Option(
        name=name or f"action {action}",
        policy=dict.fromkeys(states, action),
        stops=frozenset(states),
    )


def primitives(actions: Discrete, states: Collection[int]) -> list[Option]:
    """One option for each primitive action, in action order."""
    return [primitive(action, states) for action in actions]


__all__ = ["Option", "primitive", "primitives"]
