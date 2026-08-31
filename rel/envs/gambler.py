"""A bet on a biased coin, repeated until the money runs out or the goal is met.

Sutton and Barto, example 4.3. A gambler holds some capital and stakes part of
it on a coin. Heads pays the stake, tails takes it. Reaching the goal pays 1
and everything else pays nothing, so the value of a capital is the chance of
reaching the goal from it.

It is here for two reasons.

**It has a closed form, and only at one setting.** A fair coin makes this a
fair game, so the chance of reaching the goal from a capital is that capital
over the goal whatever the gambler stakes. That is a check against arithmetic
rather than against another computation, which is a stronger thing to have, and
`RandomWalk` is the only other environment here that offers one.

**Its famous picture is mostly an artefact.** The optimal policy of this
problem is drawn in the literature as a jagged staircase, and the jaggedness is
real at a biased coin. At a fair coin every stake is optimal, so whatever a
solver draws there is its own tie breaking rule and nothing about the problem.
`scripts/measure_gambler.py` measures how much of the picture survives a
different rule.

## The states and the actions

A state is the capital, from nothing to the goal. Nothing and the goal are the
two endings. An action is the stake, from one up to half the goal, which is the
largest stake that is ever legal: a larger one would win past the goal, and
winning past the goal is winning.

A stake larger than the capital, or larger than what is left to win, is clipped
to the largest legal one rather than refused. Every environment here has one
action space for all of its states, and clipping is what `RandomWalk` does with
a step that would pass an ending.

## Why staking nothing is not an action

The problem as stated allows it, and this was built with it. It pays nothing
and leaves the capital where it was, which makes it a loop of one state, and a
loop of one state at a discount of one is worth exactly what the state is
worth. So it ties with the best real stake at every capital once the values
have settled, and a solver that breaks ties towards the lower action takes it.

That is not a small effect. At a goal of a hundred and a coin that lands heads
0.4 of the time, value iteration staked nothing at 35 of the 99 capitals, and
`rel.agents.dp.evaluate_policy` scored the resulting policy at minus infinity,
because a gambler following it never reaches an ending from anywhere.

The alternative was to keep the action and break ties elsewhere. Tie breaking
is a property of the solver rather than of this problem, so the fix belongs
here: the action that is never useful and always ties is the one to leave out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rel.core import NO_INFO, EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete


class Gambler(TabularEnv):
    """Stake part of a capital on a coin, until the goal or nothing is reached."""

    def __init__(self, rng: Rng, goal: int = 100, heads: float = 0.4) -> None:
        super().__init__(rng)

        if goal < 2:
            raise ValueError("A goal is at least two, so that there is a bet to make.")
        if not 0.0 < heads < 1.0:
            raise ValueError(f"A coin lands heads some of the time. {heads} does not.")

        self.goal = goal
        self.heads = heads

        self.observation_space = Discrete(goal + 1)
        #: Every stake from one to half the goal. Half is the largest that is
        #: ever legal, and one is the smallest that is ever useful. Action `a`
        #: stakes `a + 1`, so there is no action that does nothing.
        self.action_space = Discrete(goal // 2)
        self.spec = EnvSpec(
            name="gambler",
            summary=(
                f"Stake part of a capital on a coin that lands heads "
                f"{heads:g} of the time. Reaching {goal} pays 1."
            ),
            max_episode_steps=1000,
            solved_return=None,
        )

        self.start = goal // 2
        self.at = self.start
        self._flips = 0
        self._won = 0

    # -- What it is worth ---------------------------------------------------

    def true_values(self) -> tuple[float, ...]:
        """The chance of reaching the goal from each capital, played well.

        Worked out rather than swept, and only for a fair coin. A fair game
        stopped at either end is worth the same however it is played: the
        chance of reaching the goal from a capital is that capital over the
        goal, for every stake that is not nothing.

        A biased coin has no such form. The value there is the fixed point of
        the Bellman equation and nothing simpler, so this refuses rather than
        answering a little wrongly. `rel.agents.dp.value_iteration` gives it.
        """
        if self.heads != 0.5:
            raise ValueError(
                f"The closed form is for a fair coin and this one lands heads "
                f"{self.heads:g} of the time. `rel.agents.dp.value_iteration` "
                f"gives the values from the model instead."
            )
        return tuple(state / self.goal for state in range(self.goal + 1))

    def values_to_score(self) -> dict[int, float]:
        """The true values of the capitals a gambler can actually hold.

        The two endings are left out. A gambler is never in one, so an
        estimate there never moves off what it started at, and scoring it
        would measure the starting value rather than the learning.
        """
        return {
            state: worth
            for state, worth in enumerate(self.true_values())
            if not self.is_ending(state)
        }

    def is_ending(self, state: int) -> bool:
        return state in (0, self.goal)

    def terminal_states(self) -> frozenset[int]:
        """Nothing and the goal, whatever the model looks like.

        The default reads the model and calls a state terminal when every
        branch out of it says so. At a goal of two there is one capital, both
        of whose branches reach an ending, and the default would call it an
        ending too. `RandomWalk` has the same trap at its smallest size and
        the same answer to it.
        """
        return frozenset({0, self.goal})

    def stake(self, state: int, action: int) -> int:
        """The stake this action really makes from this capital.

        Action `a` stakes `a + 1`, clipped to what is legal: never more than
        the capital, and never more than what is left to win. At an ending it
        is nothing, because there is nothing left to stake.
        """
        if self.is_ending(state):
            return 0
        return min(action + 1, state, self.goal - state)

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        self.at = self.start
        self._flips = 0
        self._won = 0
        return self.at

    def _step(self, action: int) -> Step[int]:
        staked = self.stake(self.at, action)
        won = self.rng.chance(self.heads)

        self._flips += 1
        self._won += int(won)

        self.at = self.at + staked if won else self.at - staked
        reward = 1.0 if self.at == self.goal else 0.0
        return Step(
            observation=self.at,
            reward=reward,
            terminated=self.is_ending(self.at),
            truncated=False,
        )

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if self.is_ending(state):
            return (Outcome(1.0, state, 0.0, True),)

        staked = self.stake(state, action)
        up = state + staked
        down = state - staked
        return (
            Outcome(self.heads, up, 1.0 if up == self.goal else 0.0, up == self.goal),
            Outcome(1.0 - self.heads, down, 0.0, down == 0),
        )

    def start_states(self) -> Sequence[tuple[float, int]]:
        return ((1.0, self.start),)

    # -- What the run really did --------------------------------------------

    def audit(self) -> Mapping[str, float]:
        """How the coin fell, which is what a short run is really about.

        A gambler who reached the goal on a coin that came up heads nine times
        of ten did not find a good policy. Reading the return alone on a
        handful of episodes cannot tell that from one that did.
        """
        if self._flips == 0:
            return NO_INFO
        return {
            "flips": float(self._flips),
            "heads_share": self._won / self._flips,
        }

    def render(self) -> str:
        width = 40
        filled = round(width * self.at / self.goal)
        bar = "#" * filled + "." * (width - filled)
        return f"[{bar}] {self.at} of {self.goal}"


__all__ = ["Gambler"]
