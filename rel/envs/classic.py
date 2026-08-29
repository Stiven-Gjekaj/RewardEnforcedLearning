"""The grids that the literature uses, written as pictures of themselves.

Each function here is one call to `GridWorld` with a layout and a few numbers.
Nothing is subclassed. A reader who wants to know what the cliff walk is looks
at eleven lines and sees the whole of it.

The numbers come from Sutton and Barto, *Reinforcement Learning: An
Introduction*, second edition. Where this project changes one, the docstring
says so and says why.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.core import EnvSpec, Outcome, Step, TabularEnv
from rel.envs.gridworld import GridWorld
from rel.rng import Rng
from rel.spaces import Discrete

# Example 6.6. Four rows and twelve columns. Every step pays -1. Walking into
# the cliff pays -100 and puts the agent back at the start, and the episode
# carries on.
#
# The best possible return is -13: one step up, eleven right, one down. The
# path along the top pays -15 and never risks the cliff, which is the whole
# point of the example. SARSA learns the safe path and Q-learning learns the
# edge, and the difference between them is a difference in what each one
# assumes about the policy that follows.
CLIFF = (
    "............",
    "............",
    "............",
    "SXXXXXXXXXXG",
)

# Example 6.5. Seven rows and ten columns. A column with wind lifts the agent
# upward by its strength, on top of whatever move it made.
WINDY = (
    "..........",
    "..........",
    "..........",
    "S......G..",
    "..........",
    "..........",
    "..........",
)
WINDY_STRENGTH = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)

# The four rooms of Sutton, Precup and Singh, 1999. Eleven by eleven inside a
# wall, cut into four rooms by two walls with one gap each.
#
# This grid is here because it separates agents that the cliff walk does not.
# A hallway is a bottleneck, so the value of reaching one is high for a long
# stretch of the run, and an agent that plans with a model finds the second
# hallway far sooner than one that only backs up along the path it walked.
FOUR_ROOMS = (
    "#############",
    "#     #    G#",
    "#     #     #",
    "#           #",
    "#     #     #",
    "#     #     #",
    "## ####     #",
    "#     ### ###",
    "#     #     #",
    "#     #     #",
    "#           #",
    "#S    #     #",
    "#############",
)

# Figure 8.2, the maze that Dyna is shown on. Six rows and nine columns, with
# seven blocked cells. Every step pays nothing and reaching the goal pays 1, so
# a run says nothing at all until the goal is found once.
DYNA_MAZE = (
    ".......#G",
    "..#....#.",
    "S.#....#.",
    "..#......",
    ".....#...",
    ".........",
)

# The frozen lake, four by four and eight by eight. A hole ends the episode
# with nothing paid. The goal pays 1. There is no step cost at all, so the only
# thing that separates two policies is how often each one arrives.
LAKE_4 = (
    "S...",
    ".X.X",
    "...X",
    "X..G",
)
LAKE_8 = (
    "S.......",
    "........",
    "...X....",
    ".....X..",
    "...X....",
    ".XX...X.",
    ".X..X.X.",
    "...X...G",
)


def cliff_walk(rng: Rng, slip: float = 0.0) -> GridWorld:
    """The cliff walk. Every step pays -1 and the cliff pays -100."""
    return GridWorld(
        rng,
        CLIFF,
        step_reward=-1.0,
        pit_reward=-100.0,
        pit_ends_episode=False,
        slip=slip,
        max_episode_steps=500,
        name="cliff",
        summary="Four by twelve. The short path runs along a cliff that pays -100.",
    )


def windy_grid(
    rng: Rng, king_moves: bool = False, wind_varies: bool = False
) -> GridWorld:
    """The windy grid. Three columns lift by one and two lift by two."""
    kind = "king moves" if king_moves else "four moves"
    gusts = ", wind that varies" if wind_varies else ""
    return GridWorld(
        rng,
        WINDY,
        step_reward=-1.0,
        wind=WINDY_STRENGTH,
        wind_varies=wind_varies,
        king_moves=king_moves,
        can_stay=king_moves,
        max_episode_steps=1000,
        name="windy",
        summary=f"Seven by ten, with an upward wind in six columns ({kind}{gusts}).",
    )


def four_rooms(rng: Rng, slip: float = 0.0) -> GridWorld:
    """Four rooms joined by four gaps. Every step pays -1."""
    return GridWorld(
        rng,
        FOUR_ROOMS,
        step_reward=-1.0,
        slip=slip,
        max_episode_steps=1000,
        name="rooms",
        summary="Eleven by eleven, cut into four rooms with one gap between each pair.",
    )


def dyna_maze(rng: Rng) -> GridWorld:
    """The Dyna maze. Nothing is paid until the goal, which pays 1."""
    return GridWorld(
        rng,
        DYNA_MAZE,
        step_reward=0.0,
        goal_reward=1.0,
        max_episode_steps=2000,
        name="maze",
        summary="Six by nine with seven blocked cells. Only the goal pays.",
        suggested_discount=0.95,
    )


def frozen_lake(rng: Rng, size: int = 4, slip: float = 2.0 / 3.0) -> GridWorld:
    """A slippery lake. A hole ends the episode and the goal pays 1.

    The default slip of two thirds is the classic setting: the move the agent
    asked for happens one time in three, and each of the two moves at right
    angles to it happens one time in three as well.
    """
    layout: tuple[str, ...]
    if size == 4:
        layout = LAKE_4
    elif size == 8:
        layout = LAKE_8
    else:
        raise ValueError("The lake comes in size 4 and size 8.")

    return GridWorld(
        rng,
        layout,
        step_reward=0.0,
        goal_reward=1.0,
        pit_reward=0.0,
        pit_ends_episode=True,
        slip=slip,
        max_episode_steps=200,
        name=f"lake{size}",
        summary=(
            f"{size} by {size}, slippery. A hole ends the episode and pays nothing."
        ),
    )


class RandomWalk(TabularEnv):
    """A line of cells with a terminal at each end. Left pays nothing, right pays 1.

    This is Sutton and Barto, example 6.2, and it is here for one reason: it is
    the only environment in this project whose values are known in closed form.

    Under a policy that goes each way half the time, the value of the k-th cell
    from the left is exactly k / (size + 1). Every other measurement here
    compares an agent against dynamic programming over the same model, which is
    a strong check and still a check against another computation. This is a
    check against arithmetic.

    ## Why it is not a grid

    A one row grid would have four actions, two of which walk into a wall, and
    a policy that goes each way half the time would then be standing still half
    the time. That is a different problem with a different answer, and the
    answer above is the reason this environment exists.

    ## The states

    `0` and `size + 1` are the two endings and the cells between them are the
    walk. The episode starts in the middle, which is why an odd `size` is the
    usual choice: with an even one there is no middle and the start is left of
    centre.
    """

    LEFT = 0
    RIGHT = 1

    def __init__(self, rng: Rng, size: int = 5) -> None:
        super().__init__(rng)

        if size < 1:
            raise ValueError("A walk needs at least one cell between its endings.")
        self.size = size

        self.observation_space = Discrete(size + 2)
        self.action_space = Discrete(2)
        self.action_names = ("left", "right")
        self.spec = EnvSpec(
            name="walk",
            summary=(
                f"{size} cells in a line between two endings. "
                f"The right one pays 1 and nothing else pays anything."
            ),
            max_episode_steps=1000,
        )

        self.start = (size + 1) // 2
        self.at = self.start

    # -- What it is worth ---------------------------------------------------

    def true_values(self) -> tuple[float, ...]:
        """What each state is worth under a policy that goes each way equally.

        Worked out rather than estimated. A walk on a line is a fair game, so
        the chance of reaching the right ending from the k-th cell is k over
        the number of gaps, and the value is that chance because the right
        ending pays 1 and nothing else pays anything.
        """
        gaps = self.size + 1
        return tuple(
            0.0 if state in (0, self.size + 1) else state / gaps
            for state in range(self.size + 2)
        )

    def is_ending(self, state: int) -> bool:
        return state in (0, self.size + 1)

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        self.at = self.start
        return self.at

    def _step(self, action: int) -> Step[int]:
        landed, reward, ended = self._resolve(self.at, action)
        self.at = landed
        return Step(
            observation=landed, reward=reward, terminated=ended, truncated=False
        )

    def _resolve(self, state: int, action: int) -> tuple[int, float, bool]:
        if self.is_ending(state):
            return state, 0.0, True

        landed = state - 1 if action == self.LEFT else state + 1
        if landed == self.size + 1:
            return landed, 1.0, True
        return landed, 0.0, landed == 0

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        landed, reward, ended = self._resolve(state, action)
        return (Outcome(1.0, landed, reward, ended),)

    def start_states(self) -> Sequence[tuple[float, int]]:
        return ((1.0, self.start),)

    def render(self) -> str:
        cells = []
        for state in range(self.size + 2):
            if self.is_ending(state):
                cells.append("|")
            elif state == self.at:
                cells.append("o")
            else:
                cells.append("-")
        return "".join(cells)


def random_walk(rng: Rng, size: int = 5) -> RandomWalk:
    """The random walk of Sutton and Barto, example 6.2."""
    return RandomWalk(rng, size=size)


__all__ = [
    "CLIFF",
    "DYNA_MAZE",
    "FOUR_ROOMS",
    "LAKE_4",
    "LAKE_8",
    "WINDY",
    "WINDY_STRENGTH",
    "RandomWalk",
    "cliff_walk",
    "dyna_maze",
    "four_rooms",
    "frozen_lake",
    "random_walk",
    "windy_grid",
]
