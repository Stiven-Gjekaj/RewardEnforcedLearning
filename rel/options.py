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

from rel.core import TabularEnv
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

    @property
    def is_primitive(self) -> bool:
        """Whether this option stops after every step, so it is an action.

        The collapse in the class docstring, made checkable. An agent that
        holds only these is the agent it came from, and a measurement that
        wants to know whether the long options are being used can ask.
        """
        return all(state in self.stops for state in self.policy)

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


# -- Reading the rooms off a model -------------------------------------------


def steps_between(env: TabularEnv) -> dict[int, set[int]]:
    """Which states each state can reach in one step, ignoring itself.

    Built from the model rather than from a layout, so nothing here has to
    know what shape the environment is. A branch that lands where it started
    is dropped, because walking into a wall is not a way to somewhere.
    """
    edges: dict[int, set[int]] = {}
    terminal = env.terminal_states()

    for state in range(env.observation_space.n):
        if state in terminal:
            continue
        landings = {
            outcome.observation
            for action in env.action_space
            for outcome in env.transitions(state, action)
            if outcome.observation != state and outcome.observation not in terminal
        }
        if landings:
            edges[state] = landings

    return edges


def reachable(env: TabularEnv) -> set[int]:
    """Every state the agent can actually be in, walking out from the start.

    Not every state in the model is one of these. Walking into the cliff sends
    the agent back to the start, so the eleven cliff cells are described by the
    model and never stood in, and counting each of them as a room of its own
    would be counting places nobody goes.
    """
    edges = steps_between(env)
    found = {state for _, state in env.start_states()}
    stack = list(found)

    while stack:
        state = stack.pop()
        for landed in edges.get(state, ()):
            if landed not in found:
                found.add(landed)
                stack.append(landed)

    return found


def rooms(env: TabularEnv, doors: Collection[int]) -> list[frozenset[int]]:
    """What the model falls into when the doors are taken out of it.

    The pieces come back in the order of their lowest numbered state, so two
    runs on the same grid name the same room the same way.
    """
    edges = steps_between(env)
    inside = (set(edges) & reachable(env)) - set(doors)

    found: list[frozenset[int]] = []
    seen: set[int] = set()

    for start in sorted(inside):
        if start in seen:
            continue
        piece: set[int] = set()
        stack = [start]
        seen.add(start)
        while stack:
            state = stack.pop()
            piece.add(state)
            for landed in edges[state]:
                if landed in inside and landed not in seen:
                    seen.add(landed)
                    stack.append(landed)
        found.append(frozenset(piece))

    return found


# -- Building the hallway options --------------------------------------------


def _likeliest(env: TabularEnv, state: int, action: int) -> int | None:
    """Where this action most likely lands, or `None` if it goes nowhere."""
    outcomes = env.transitions(state, action)
    if not outcomes:
        return None
    landed = max(outcomes, key=lambda outcome: outcome.probability).observation
    return None if landed == state else landed


def walk_to(env: TabularEnv, target: int, inside: Collection[int]) -> dict[int, int]:
    """The action from each state of `inside` that steps towards `target`.

    A breadth first walk backwards from the target, so every state gets the
    first action on a shortest path to it. A state of `inside` that cannot
    reach the target without leaving is left out, which is what makes the
    option's start set the room it can really cross.
    """
    ahead: dict[int, int] = {}
    reached = {target}
    frontier = [target]

    while frontier:
        following: list[int] = []
        for landing in frontier:
            for state in inside:
                if state in reached:
                    continue
                for action in env.action_space:
                    if _likeliest(env, state, action) == landing:
                        ahead[state] = action
                        reached.add(state)
                        following.append(state)
                        break
        frontier = following

    return ahead


def hallway_options(env: TabularEnv, doors: Collection[int]) -> list[Option]:
    """One option for each room and each doorway on its edge.

    Cross this room and stop at that doorway. A doorway that touches only one
    room is not a way out of anywhere, so it gets no options: on the Dyna maze
    that leaves the list empty.

    The start set is the room plus the other doorways on its edge, and not the
    one being walked to. So an agent that has just arrived at one doorway can
    choose to cross to another, which is the whole point of having them. An
    option that could start where it stops would take an arbitrary first step
    and then walk back.
    """
    edges = steps_between(env)
    parts = rooms(env, doors)

    def touches(door: int, part: frozenset[int]) -> bool:
        return bool(edges.get(door, set()) & part)

    built: list[Option] = []
    for index, part in enumerate(parts):
        boundary = sorted(
            door
            for door in doors
            if touches(door, part) and sum(touches(door, other) for other in parts) >= 2
        )

        for door in boundary:
            inside = set(part) | (set(boundary) - {door})
            policy = walk_to(env, door, inside)
            if not policy:
                continue
            built.append(
                Option(
                    name=f"room {index} to gap {door}",
                    policy=policy,
                    stops=frozenset({door}),
                )
            )

    return built


__all__ = [
    "Option",
    "hallway_options",
    "primitive",
    "primitives",
    "reachable",
    "rooms",
    "steps_between",
    "walk_to",
]
