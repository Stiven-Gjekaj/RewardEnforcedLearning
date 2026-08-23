"""A grid of cells, described by a picture of it.

One class covers the cliff walk, the windy grid, the four rooms and the frozen
lake. The differences between those four are a layout and five numbers, so they
are five numbers here rather than four classes that drift apart.

## The layout is the specification

An environment is built from a list of strings that looks like the grid:

    ##########
    #S......G#
    #..XXXX..#
    ##########

A test that needs a shape writes the shape. It does not ask a generator for one
and then assert something about what came back. This is the same reason
`RogueBit` gives its map tests an ASCII builder: a test that says what it means
keeps saying it after somebody tunes the thing that used to produce it.

## Reward replaces, it does not add

Walking into the cliff in Sutton and Barto pays -100, not -101. Reaching the
goal on the cliff walk pays the ordinary -1, because the chapter counts steps.
So a tile reward replaces the step reward rather than adding to it, and the
goal reward is `None` where the goal is worth nothing special.

Getting this backwards is quiet. Every curve still goes up, the shape looks
right, and the numbers cannot be compared with the book.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rel.core import EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete

WALL = "#"
FLOOR = "."
START = "S"
GOAL = "G"
PIT = "X"

_PASSABLE = frozenset({FLOOR, START, GOAL, PIT})
_LEGEND = frozenset({WALL, FLOOR, START, GOAL, PIT, " "})

# Up, right, down, left. This order is the one the literature uses, and an
# agent that prints its policy as arrows depends on it.
CARDINAL: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
CARDINAL_NAMES = ("up", "right", "down", "left")

# The eight king moves, in a ring. The ring order is what lets a slip land on a
# direction next to the one intended, whatever the number of directions is.
KING: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
KING_NAMES = (
    "up",
    "up right",
    "right",
    "down right",
    "down",
    "down left",
    "left",
    "up left",
)

STAY = (0, 0)


class GridWorld(TabularEnv):
    """An agent on a grid, moving one cell at a time.

    The state is the cell. Nothing else changes, so the cell is the whole
    Markov state and a table holds one row for each of them.
    """

    def __init__(
        self,
        rng: Rng,
        layout: Sequence[str],
        *,
        step_reward: float = -1.0,
        goal_reward: float | None = None,
        pit_reward: float = -100.0,
        pit_ends_episode: bool = False,
        slip: float = 0.0,
        wind: Sequence[int] | None = None,
        wind_varies: bool = False,
        king_moves: bool = False,
        can_stay: bool = False,
        max_episode_steps: int | None = None,
        name: str = "grid",
        summary: str = "An agent on a grid.",
        solved_return: float | None = None,
        suggested_discount: float = 1.0,
    ) -> None:
        super().__init__(rng)

        self.layout = tuple(layout)
        self._check_layout()

        self.height = len(self.layout)
        self.width = len(self.layout[0])

        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.pit_reward = pit_reward
        self.pit_ends_episode = pit_ends_episode
        self.slip = slip

        if wind is None:
            self.wind: tuple[int, ...] = (0,) * self.width
        else:
            if len(wind) != self.width:
                raise ValueError("The wind needs one strength for each column.")
            self.wind = tuple(wind)
        self.wind_varies = wind_varies

        moves = list(KING if king_moves else CARDINAL)
        names = list(KING_NAMES if king_moves else CARDINAL_NAMES)
        self._ring = len(moves)
        if can_stay:
            moves.append(STAY)
            names.append("stay")
        self.moves: tuple[tuple[int, int], ...] = tuple(moves)
        self.action_names: tuple[str, ...] = tuple(names)

        if slip < 0.0 or slip > 1.0:
            raise ValueError("A slip probability is between 0 and 1.")

        self.observation_space = Discrete(self.height * self.width)
        self.action_space = Discrete(len(self.moves))
        self.spec = EnvSpec(
            name=name,
            summary=summary,
            max_episode_steps=max_episode_steps,
            solved_return=solved_return,
            suggested_discount=suggested_discount,
        )

        self._starts = tuple(self._cells_marked(START))
        if not self._starts:
            raise ValueError("A grid needs at least one start cell, marked S.")
        self._goals = frozenset(self._cells_marked(GOAL))
        if not self._goals:
            raise ValueError("A grid needs at least one goal cell, marked G.")
        self._pits = frozenset(self._cells_marked(PIT))

        self.at = self._starts[0]
        self._pit_entries = 0

    # -- Building -----------------------------------------------------------

    def _check_layout(self) -> None:
        if not self.layout:
            raise ValueError("A grid needs at least one row.")
        widths = {len(row) for row in self.layout}
        if len(widths) != 1:
            raise ValueError("Every row of a grid has to be the same width.")
        for row in self.layout:
            for character in row:
                if character not in _LEGEND:
                    raise ValueError(
                        f"{character!r} is not a grid tile. "
                        f"Use one of {''.join(sorted(_LEGEND))!r}."
                    )

    def _cells_marked(self, mark: str) -> list[int]:
        return [
            row * self.width + column
            for row, line in enumerate(self.layout)
            for column, character in enumerate(line)
            if character == mark
        ]

    # -- Reading the grid ---------------------------------------------------

    def cell_of(self, state: int) -> tuple[int, int]:
        """The row and column of a state."""
        return divmod(state, self.width)

    def state_of(self, row: int, column: int) -> int:
        return row * self.width + column

    def tile(self, state: int) -> str:
        row, column = self.cell_of(state)
        character = self.layout[row][column]
        return FLOOR if character == " " else character

    def is_wall(self, state: int) -> bool:
        return self.tile(state) == WALL

    def is_goal(self, state: int) -> bool:
        return state in self._goals

    def is_pit(self, state: int) -> bool:
        return state in self._pits

    def walkable(self) -> list[int]:
        """Every state an agent can stand on, in reading order."""
        return [
            state
            for state in range(self.observation_space.n)
            if self.tile(state) in _PASSABLE
        ]

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        self._pit_entries = 0
        self.at = self.rng.pick(self._starts)
        return self.at

    def _step(self, action: int) -> Step[int]:
        # Stepping draws the slip and the gust and then asks `_resolve` what
        # happens, which is the same call the model makes. Sampling the model
        # instead would work, and it would let the two drift the day one of
        # them gained a case. Keeping the movement in one place means a
        # difference between them is not possible rather than merely unlikely.
        taken = self._draw(self._slip_over(action))
        gust = self._draw(self._gusts())
        landed, reward, terminated, entered_pit = self._resolve(self.at, taken, gust)

        if entered_pit:
            self._pit_entries += 1

        self.at = landed
        return Step(landed, reward, terminated, False)

    def _draw(self, options: Sequence[tuple[float, int]]) -> int:
        """One option, or the only one without drawing.

        A grid with no slip and no varying wind draws nothing at all after the
        reset. That is worth arranging: it means two agents on the cliff walk
        meet exactly the same environment, and it means the environment stream
        of a deterministic grid stays where it is when an agent explores more.
        """
        if len(options) == 1:
            return options[0][1]
        index = self.rng.weighted_index([share for share, _ in options])
        return options[index][1]

    def audit(self) -> Mapping[str, float]:
        return {"pit_entries": float(self._pit_entries)}

    def render(self) -> str:
        lines = []
        for row, line in enumerate(self.layout):
            drawn = []
            for column, character in enumerate(line):
                if self.state_of(row, column) == self.at:
                    drawn.append("@")
                else:
                    drawn.append(character)
            lines.append("".join(drawn))
        return "\n".join(lines)

    # -- The model ----------------------------------------------------------

    def start_states(self) -> Sequence[tuple[float, int]]:
        share = 1.0 / len(self._starts)
        return [(share, state) for state in self._starts]

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if self.is_goal(state) or (self.pit_ends_episode and self.is_pit(state)):
            return [Outcome(1.0, state, 0.0, terminated=True)]
        if self.is_wall(state):
            # Nothing stands here, so nothing happens here. The entry exists so
            # that a table can be indexed by every state without a gap.
            return [Outcome(1.0, state, 0.0, terminated=True)]

        gathered: dict[tuple[int, float, bool], float] = {}
        for probability, chosen in self._slip_over(action):
            for wind_share, gust in self._gusts():
                landed, reward, terminated, _ = self._resolve(state, chosen, gust)
                key = (landed, reward, terminated)
                gathered[key] = gathered.get(key, 0.0) + probability * wind_share

        return [
            Outcome(share, landed, reward, terminated)
            for (landed, reward, terminated), share in gathered.items()
        ]

    def _slip_over(self, action: int) -> list[tuple[float, int]]:
        """The action that is really taken, and how likely each one is."""
        if self.slip <= 0.0 or action >= self._ring:
            return [(1.0, action)]

        left = (action - 1) % self._ring
        right = (action + 1) % self._ring
        return [
            (1.0 - self.slip, action),
            (self.slip / 2.0, left),
            (self.slip / 2.0, right),
        ]

    def _gusts(self) -> list[tuple[float, int]]:
        """How much the wind is shifted from its column strength."""
        if not self.wind_varies:
            return [(1.0, 0)]
        third = 1.0 / 3.0
        return [(third, -1), (third, 0), (third, 1)]

    def _resolve(
        self, state: int, action: int, gust: int
    ) -> tuple[int, float, bool, bool]:
        """Where the agent lands, what it is paid, and whether it stopped.

        The fourth value says whether the agent walked into a pit. The first
        three cannot say it: the cliff sends the agent back to the start, so the
        state it lands in is the start, and the reward is the only trace left.
        Counting pits by reward breaks on a grid where a pit pays what a step
        pays, which the frozen lake does.
        """
        row, column = self.cell_of(state)
        move_row, move_column = self.moves[action]

        strength = self.wind[column]
        lift = strength + gust if strength > 0 else 0

        # The edge of the grid holds each axis on its own rather than cancelling
        # the whole move. The difference shows twice. A wind of two at the top
        # of the windy grid would otherwise cancel every move made in that
        # column, and the grid would have a row nothing can cross. A king move
        # into a corner would stop dead rather than sliding along the wall.
        landed_row = min(max(row + move_row - lift, 0), self.height - 1)
        landed_column = min(max(column + move_column, 0), self.width - 1)

        candidate = self.state_of(landed_row, landed_column)
        landed = state if self.is_wall(candidate) else candidate

        if self.is_goal(landed):
            reward = self.step_reward if self.goal_reward is None else self.goal_reward
            return landed, reward, True, False

        if self.is_pit(landed):
            if self.pit_ends_episode:
                return landed, self.pit_reward, True, True
            # The cliff sends the agent back to where it started, and the
            # episode carries on. With one start cell this is deterministic,
            # which is why the model can name a single outcome.
            return self._starts[0], self.pit_reward, False, True

        return landed, self.step_reward, False, False


__all__ = [
    "CARDINAL",
    "CARDINAL_NAMES",
    "FLOOR",
    "GOAL",
    "KING",
    "KING_NAMES",
    "PIT",
    "START",
    "WALL",
    "GridWorld",
]
