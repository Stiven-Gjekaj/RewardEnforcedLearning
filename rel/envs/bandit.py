"""A row of levers, and no state at all.

This is the ten armed testbed of Sutton and Barto, chapter 2. There is one
state, so nothing an agent does changes what it faces next. Everything left is
the tension between pulling the lever that has paid best so far and pulling one
that has not been tried enough to know.

## An episode is one problem

`reset` draws a new set of true values. One episode is therefore one bandit
problem, run for a thousand pulls, and averaging over episodes is the average
over problems that the chapter plots.

## Why the reward is not the measure

A run that pulls the best lever every time and one that pulls a lever half a
point worse every time both produce a smooth rising curve. The reward tells
them apart slowly and badly, because the noise on one pull is as large as the
gap between the best two levers.

`audit` reports two things the reward does not: how often the agent chose the
lever that was really best, and how much it gave up by not choosing it. Both
need the true values, which the agent never sees.
"""

from __future__ import annotations

from collections.abc import Mapping

from rel.core import Env, EnvSpec, Step
from rel.rng import Rng
from rel.spaces import Discrete

ONE_STATE = 0


class KArmedBandit(Env[int]):
    """`arms` levers. Each pays a noisy reward around its own true value."""

    def __init__(
        self,
        rng: Rng,
        arms: int = 10,
        spread: float = 1.0,
        noise: float = 1.0,
        walk: float = 0.0,
        pulls: int = 1000,
    ) -> None:
        super().__init__(rng)

        if arms < 2:
            raise ValueError("A bandit needs at least two levers.")
        if noise < 0.0:
            raise ValueError("The noise must not be below zero.")
        if walk < 0.0:
            raise ValueError("The walk must not be below zero.")

        self.arms = arms
        self.spread = spread
        self.noise = noise
        self.walk = walk

        self.observation_space = Discrete(1)
        self.action_space = Discrete(arms)
        self.spec = EnvSpec(
            name="bandit",
            summary=(
                f"{arms} levers with no state. Each pays a noisy reward around "
                f"its own true value."
            ),
            max_episode_steps=pulls,
        )

        self.values: list[float] = [0.0] * arms
        self._chosen_best = 0
        self._pulls = 0
        self._regret = 0.0

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        # A run of this environment is an average over problems, so a new
        # episode is a new problem. The walking version starts every lever
        # equal instead, because there the interest is in following a target
        # that moves rather than in finding one that is fixed.
        if self.walk > 0.0:
            self.values = [0.0] * self.arms
        else:
            self.values = [self.rng.normal(0.0, self.spread) for _ in range(self.arms)]

        self._chosen_best = 0
        self._pulls = 0
        self._regret = 0.0
        return ONE_STATE

    def _step(self, action: int) -> Step[int]:
        best = self.best_value()
        if self.values[action] >= best:
            self._chosen_best += 1
        self._regret += best - self.values[action]
        self._pulls += 1

        reward = self.values[action]
        if self.noise > 0.0:
            reward += self.rng.normal(0.0, self.noise)

        # The walk happens after the pull, so the reward just paid belongs to
        # the values the agent was choosing between.
        if self.walk > 0.0:
            self.values = [
                value + self.rng.normal(0.0, self.walk) for value in self.values
            ]

        return Step(ONE_STATE, reward, terminated=False, truncated=False)

    # -- What the reward does not say ---------------------------------------

    def best_arm(self) -> int:
        return max(range(self.arms), key=lambda arm: self.values[arm])

    def best_value(self) -> float:
        return max(self.values)

    def audit(self) -> Mapping[str, float]:
        if self._pulls == 0:
            return {"best_arm_share": 0.0, "regret": 0.0}
        return {
            "best_arm_share": self._chosen_best / self._pulls,
            "regret": self._regret,
        }

    def render(self) -> str:
        best = self.best_arm()
        lines = []
        for arm in range(self.arms):
            mark = "*" if arm == best else " "
            lines.append(f"{mark} {arm:2d}  {self.values[arm]:+.3f}")
        return "\n".join(lines)


def stationary_bandit(rng: Rng, arms: int = 10, pulls: int = 1000) -> KArmedBandit:
    """The ten armed testbed. New true values every episode, and they stay put."""
    return KArmedBandit(rng, arms=arms, pulls=pulls)


def drifting_bandit(
    rng: Rng, arms: int = 10, walk: float = 0.01, pulls: int = 10_000
) -> KArmedBandit:
    """Exercise 2.5. Every lever starts equal, and all of them wander.

    An agent that averages every reward it has ever seen from a lever cannot
    follow a target that moves, because the old rewards outvote the new ones
    and never stop doing so. An agent with a fixed step size can. This is the
    environment where those two come apart, and the reward curve alone shows it
    only faintly. The share of best choices shows it plainly.
    """
    return KArmedBandit(rng, arms=arms, walk=walk, pulls=pulls)


__all__ = ["KArmedBandit", "drifting_bandit", "stationary_bandit"]
