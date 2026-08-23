"""Drawing what an agent has learned about a grid.

Three pictures, and each one answers a different question.

The policy says what the agent would do in each cell. It is the picture that
shows the cliff walk result at a glance: SARSA runs along the top and
Q-learning runs along the edge, and no table of returns says that as directly.

The value map says how good the agent thinks each cell is. It shows how far the
learning has spread, which the policy does not: a cell can have a sensible
arrow and a value of nearly nothing, because one visit is enough to order four
actions and not enough to know what the cell is worth.

The difference map says where a learned policy and the best possible policy
disagree. Two policies can score almost the same and differ in half the cells,
and the map says which half.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.agents.base import Agent
from rel.envs.gridworld import GOAL, PIT, START, WALL, GridWorld
from rel.ui.colour import Palette

ARROWS = {
    (-1, 0): "^",
    (0, 1): ">",
    (1, 0): "v",
    (0, -1): "<",
    (-1, 1): "/",
    (1, 1): "\\",
    (1, -1): "/",
    (-1, -1): "\\",
    (0, 0): "o",
}

#: Light to dark. A value map is read as a picture rather than as numbers, so
#: the ramp has to be visible in a terminal with no colour at all.
SHADES = " .:-=+*%@"


def policy_map(
    env: GridWorld, agent: Agent[int], palette: Palette | None = None
) -> str:
    """The grid with the agent's greedy action drawn in each cell."""
    palette = palette or Palette(enabled=False)
    lines = []

    for row in range(env.height):
        drawn = []
        for column in range(env.width):
            state = env.state_of(row, column)
            tile = env.tile(state)

            if tile == WALL:
                drawn.append(palette.paint("#", "grey"))
            elif tile == GOAL:
                drawn.append(palette.paint("G", "green"))
            elif tile == PIT:
                drawn.append(palette.paint("X", "red"))
            else:
                action = agent.greedy(state)
                mark = ARROWS.get(env.moves[action - env.action_space.start], "?")
                if tile == START:
                    drawn.append(palette.paint(mark, "cyan"))
                else:
                    drawn.append(mark)
        lines.append("".join(drawn))

    return "\n".join(lines)


def value_map(
    env: GridWorld, values: Sequence[float], palette: Palette | None = None
) -> str:
    """The grid shaded by value, light for low and dark for high.

    `values` holds one number for each state, which is what `value_iteration`
    returns and what `best_values` takes off an agent.
    """
    palette = palette or Palette(enabled=False)

    walkable = [
        values[state]
        for state in env.walkable()
        if not env.is_wall(state) and _finite(values[state])
    ]
    if not walkable:
        return env.render()

    low = min(walkable)
    high = max(walkable)
    span = high - low

    lines = []
    for row in range(env.height):
        drawn = []
        for column in range(env.width):
            state = env.state_of(row, column)
            if env.is_wall(state):
                drawn.append(palette.paint("#", "grey"))
                continue
            if not _finite(values[state]):
                drawn.append(palette.paint("?", "grey"))
                continue

            share = 0.5 if span <= 0 else (values[state] - low) / span
            index = min(len(SHADES) - 1, max(0, round(share * (len(SHADES) - 1))))
            drawn.append(SHADES[index])
        lines.append("".join(drawn))

    unknown = sum(1 for state in env.walkable() if not _finite(values[state]))
    caption = f"  low {low:.2f}   high {high:.2f}"
    if unknown:
        caption += f"   ? = {unknown} cells it has never stood in"
    lines.append(caption)
    return "\n".join(lines)


def difference_map(
    env: GridWorld,
    agent: Agent[int],
    best: Sequence[int],
    palette: Palette | None = None,
) -> str:
    """The grid marked where the agent and the best possible policy differ.

    A cell where they agree is drawn as a dot. A cell where they differ carries
    the action the agent chose, so the map says what it does instead rather
    than only that it does something else.
    """
    palette = palette or Palette(enabled=False)
    lines = []
    differing = 0
    counted = 0

    for row in range(env.height):
        drawn = []
        for column in range(env.width):
            state = env.state_of(row, column)
            tile = env.tile(state)

            if tile == WALL:
                drawn.append(palette.paint("#", "grey"))
                continue
            if tile == GOAL:
                drawn.append(palette.paint("G", "green"))
                continue

            counted += 1
            chosen = agent.greedy(state)
            if chosen == best[state]:
                drawn.append(palette.paint(".", "grey"))
            else:
                differing += 1
                mark = ARROWS.get(env.moves[chosen - env.action_space.start], "?")
                drawn.append(palette.paint(mark, "yellow"))
        lines.append("".join(drawn))

    lines.append(f"  {differing} of {counted} cells differ from the best policy")
    return "\n".join(lines)


def best_values(env: GridWorld, agent: Agent[int]) -> list[float]:
    """What the agent thinks each state is worth, taken off its own numbers.

    An agent that keeps no numbers has none of these, and this raises rather
    than inventing zeros. A map of zeros looks like a value map of a state
    nothing has learned, which is a different thing entirely.
    """
    values = []
    for state in range(env.observation_space.n):
        if not agent.knows(state):
            # Not zero. Zero is the value of a state worth nothing, and on a
            # grid where every step costs something that is the best value on
            # the map. A run of the cliff walk drew the whole cliff as the
            # finest place to stand, because the agent is put back at the start
            # the moment it enters one and so has never stood there.
            values.append(float("nan"))
            continue

        row = agent.action_values(state)
        if row is None:
            raise TypeError(
                f"{type(agent).__name__} keeps no value for an action, so "
                f"there is no value map to draw."
            )
        values.append(max(row))
    return values


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


__all__ = [
    "ARROWS",
    "SHADES",
    "best_values",
    "difference_map",
    "policy_map",
    "value_map",
]
