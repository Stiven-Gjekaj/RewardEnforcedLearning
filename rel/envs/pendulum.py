"""A weight on a rod, and a motor too weak to lift it in one go.

Every other environment here takes an action from a short list. This one takes
a number: a torque anywhere between minus two and two, and the difference is
the whole reason it is here.

    action space   Box([-2.0], [2.0])
    observation    cos of the angle, sin of the angle, angular speed
    reward         zero at the top and nothing else, minus everywhere else
    ending         none, so every episode runs to the step limit

## Why the angle is two numbers

An angle wraps. Feed it to an approximator as one number and pi and minus pi,
which are the same place, sit at opposite ends of the range, so anything
learned about one says nothing about the other. Its cosine and its sine are two
numbers that agree there and everywhere else.

That costs one extra input and it removes a seam that a tile coder would put a
boundary across and a network would have to learn its way around.

## Why it cannot be lifted in one go

The motor at full torque cannot hold the weight up from horizontal, let alone
raise it. The only way to the top is to swing: drive one way, let it fall back,
drive the other way, and add a little energy each time.

That makes it the same shape of problem as the mountain car, and a greedy step
by step reading of it fails for the same reason. What it adds is that the
torque is a dial rather than three buttons, so a policy has to say how hard as
well as which way.

## The reward is a cost

Nothing pays anything above zero. The best a step can be is zero, which is the
weight balanced at the top with no speed and no torque, and everything else is
negative, so a return is a measure of how much was spent getting there and
staying there. A policy that spends no torque at all scores -1187 over
two hundred steps, averaged over twenty starts, and one that draws its torque
at random scores -1219. Both are what a run looks like when nothing has been
learned, and the gap between them is the point: **thrashing is not better than
doing nothing here.**

The angle in that cost is wrapped onto minus pi to pi first. Without the wrap a
weight that has gone all the way round is charged for the whole journey, and
the cheapest policy would be one that never turns.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from rel.core import Env, EnvSpec, Step
from rel.rng import Rng
from rel.spaces import Box

Observation = tuple[float, ...]
Torque = tuple[float, ...]


def wrapped(angle: float) -> float:
    """The same angle, moved onto minus pi to pi.

    A weight that has gone twice round is in the same place as one that has
    not, and the cost of where it is has to say so.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Pendulum(Env[Observation, Torque]):
    """A weight on a rod, driven by a motor that cannot lift it directly."""

    GRAVITY = 10.0
    MASS = 1.0
    LENGTH = 1.0
    STEP = 0.05
    TOP_SPEED = 8.0
    TOP_TORQUE = 2.0

    #: What each part of the cost is worth. The angle dominates, the speed
    #: keeps a policy from spinning through the top, and the torque is charged
    #: a thousandth so that two policies that reach the top are separated by
    #: which one used less motor.
    SPEED_COST = 0.1
    TORQUE_COST = 0.001

    def __init__(self, rng: Rng, max_episode_steps: int = 200) -> None:
        super().__init__(rng)

        self.observation_space = Box(
            [-1.0, -1.0, -self.TOP_SPEED],
            [1.0, 1.0, self.TOP_SPEED],
            names=["cos angle", "sin angle", "speed"],
        )

        # Everything the observation holds is clamped by the physics, so the
        # box a tile coder divides is the box the environment reports.
        self.tiling_space = self.observation_space
        self.action_space = Box([-self.TOP_TORQUE], [self.TOP_TORQUE], names=["torque"])
        self.spec = EnvSpec(
            name="pendulum",
            summary="A weight on a rod and a motor too weak to lift it in one go.",
            max_episode_steps=max_episode_steps,
            ends=False,
            suggested_discount=0.99,
        )

        self.angle = math.pi
        self.speed = 0.0
        self.highest = -1.0
        self.torque_spent = 0.0

    # -- The contract -------------------------------------------------------

    def _reset(self) -> Observation:
        # Anywhere at all, which is the start the literature uses. Half of the
        # starts are already near the top, so a run that never learned to
        # swing still scores better than the worst case, and a report that
        # reads only the mean return hides that.
        self.angle = self.rng.uniform(-math.pi, math.pi)
        self.speed = self.rng.uniform(-1.0, 1.0)
        self.highest = math.cos(self.angle)
        self.torque_spent = 0.0
        return self._observation()

    def _step(self, action: Torque) -> Step[Observation]:
        # Not clamped. `Env.step` refuses an action the space does not hold,
        # so a torque outside the box never reaches here, and a clamp would be
        # a branch nothing could take.
        torque = action[0]
        cost = (
            wrapped(self.angle) ** 2
            + self.SPEED_COST * self.speed**2
            + self.TORQUE_COST * torque**2
        )

        pull = 3.0 * self.GRAVITY / (2.0 * self.LENGTH) * math.sin(self.angle)
        push = 3.0 / (self.MASS * self.LENGTH**2) * torque

        # Semi-implicit Euler, as in `rel.envs.control`: the speed is advanced
        # first and the angle is advanced with the new speed.
        self.speed += (pull + push) * self.STEP
        self.speed = min(max(self.speed, -self.TOP_SPEED), self.TOP_SPEED)
        self.angle += self.speed * self.STEP

        self.highest = max(self.highest, math.cos(self.angle))
        self.torque_spent += abs(torque)

        return Step(self._observation(), -cost, terminated=False, truncated=False)

    def _observation(self) -> Observation:
        return (math.cos(self.angle), math.sin(self.angle), self.speed)

    # -- What the reward does not say ---------------------------------------

    def audit(self) -> Mapping[str, float]:
        """How near the top it got, and how much motor it spent getting there.

        The return adds the cost of every step, so a policy that reaches the
        top late and one that never reaches it can score alike over two hundred
        steps. The highest point reached tells them apart.
        """
        return {
            "highest_point": self.highest,
            "torque_spent": self.torque_spent,
        }

    def render(self) -> str:
        """The pivot and the weight, with up as up.

        Eleven rows and twenty one columns, so the weight moves about the same
        distance on the screen for a given turn in either direction. The sine
        and the cosine are between minus one and one and the reaches are half
        of each side, so the weight lands on the edge of the box at the
        furthest and never outside it, and nothing has to be clamped.
        """
        across, down = 21, 11
        middle_across, middle_down = across // 2, down // 2

        grid = [[" "] * across for _ in range(down)]
        grid[middle_down][middle_across] = "+"

        column = middle_across + round(middle_across * math.sin(self.angle))
        row = middle_down - round(middle_down * math.cos(self.angle))
        grid[row][column] = "o"

        return "\n".join("".join(line) for line in grid)


def pendulum(rng: Rng, steps: int = 200) -> Pendulum:
    """The weight on a rod, with an episode of `steps` steps."""
    return Pendulum(rng, max_episode_steps=steps)


__all__ = ["Pendulum", "pendulum", "wrapped"]
