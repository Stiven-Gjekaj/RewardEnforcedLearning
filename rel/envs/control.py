"""Two classic control problems, with the physics written out.

Neither one has a table of states. The observation is a handful of real
numbers, so an agent has to approximate, and these are the two environments
that separate a tile coder from a table.

## The integrator

Both use semi-implicit Euler: the velocity is advanced first, then the position
is advanced using the new velocity.

The widely copied cart pole uses plain Euler instead, advancing the position
with the old velocity. The difference is one line and it is not a rounding
error. Plain Euler adds energy to a swinging system, so a pole that is only
just balanced drifts outward on its own. Semi-implicit Euler does not, which is
why every physics engine uses it.

`docs/algorithms.md` holds the measurement of how far the two come apart on the
same policy, so the choice can be checked rather than argued about.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from rel.core import DiscreteEnv, EnvSpec, Step
from rel.rng import Rng
from rel.spaces import Box, Discrete

Observation = tuple[float, ...]


class CartPole(DiscreteEnv[Observation]):
    """A pole hinged on a cart. Push the cart left or right to keep it up.

    The state is the position and speed of the cart and the angle and angular
    speed of the pole. Every step pays 1, and the episode ends when the pole
    passes twelve degrees or the cart runs off the track.

    The numbers are from Barto, Sutton and Anderson, 1983.
    """

    GRAVITY = 9.8
    CART_MASS = 1.0
    POLE_MASS = 0.1
    #: Half the pole, which is where its centre of mass is.
    POLE_HALF_LENGTH = 0.5
    FORCE = 10.0
    TIMESTEP = 0.02

    #: Twelve degrees, in radians.
    ANGLE_LIMIT = 12.0 * math.pi / 180.0
    TRACK_LIMIT = 2.4

    #: How near upright counts as steady, for the audit. About three degrees.
    STEADY_ANGLE = 0.05
    STEADY_POSITION = 0.5

    def __init__(
        self,
        rng: Rng,
        max_episode_steps: int = 500,
        start_spread: float = 0.05,
    ) -> None:
        super().__init__(rng)

        self.start_spread = start_spread

        # Wide enough to hold every observation the environment can produce.
        # The episode ends the moment the position or the angle passes its
        # limit, so the observation that reports the ending is the only one
        # that goes past, and it goes past by one step of movement. The speed
        # and the spin have no limit of their own, and a run of four hundred
        # random episodes reached 2.53 and 3.11. These bounds hold that with
        # room to spare, and `test_every_observation_is_inside_the_box` drives
        # the environment to check it.
        self.observation_space = Box(
            [-self.TRACK_LIMIT * 2, -6.0, -self.ANGLE_LIMIT * 2, -6.0],
            [self.TRACK_LIMIT * 2, 6.0, self.ANGLE_LIMIT * 2, 6.0],
            names=["position", "speed", "angle", "spin"],
        )

        # The range a tile coder divides. It is narrower than the observation
        # space on purpose. Almost every step of a run happens inside these
        # bounds, and spreading the tiles over the wider box instead would
        # spend most of them on states that a balanced pole never reaches.
        # The tile coder clips, so a rare step outside lands in the edge tile.
        self.tiling_space = Box(
            [-self.TRACK_LIMIT, -3.0, -self.ANGLE_LIMIT, -3.5],
            [self.TRACK_LIMIT, 3.0, self.ANGLE_LIMIT, 3.5],
            names=["position", "speed", "angle", "spin"],
        )
        self.action_space = Discrete(2)
        self.spec = EnvSpec(
            name="cartpole",
            summary="Keep a hinged pole upright by pushing the cart it stands on.",
            max_episode_steps=max_episode_steps,
            # The usual bar: an average of 475 over a hundred episodes, out of
            # a possible 500.
            solved_return=475.0,
        )

        self.position = 0.0
        self.speed = 0.0
        self.angle = 0.0
        self.spin = 0.0
        self._steady = 0
        self._steps = 0

    @property
    def total_mass(self) -> float:
        return self.CART_MASS + self.POLE_MASS

    @property
    def pole_moment(self) -> float:
        """The mass of the pole times the distance to its centre of mass."""
        return self.POLE_MASS * self.POLE_HALF_LENGTH

    def _reset(self) -> Observation:
        spread = self.start_spread
        self.position = self.rng.uniform(-spread, spread)
        self.speed = self.rng.uniform(-spread, spread)
        self.angle = self.rng.uniform(-spread, spread)
        self.spin = self.rng.uniform(-spread, spread)
        self._steady = 0
        self._steps = 0
        return self._observation()

    def _step(self, action: int) -> Step[Observation]:
        force = self.FORCE if action == 1 else -self.FORCE

        cosine = math.cos(self.angle)
        sine = math.sin(self.angle)

        # The two accelerations of a cart and pole, solved together. The pole
        # pushes the cart while the cart accelerates under the pole, so neither
        # can be worked out without the other.
        shared = (force + self.pole_moment * self.spin**2 * sine) / self.total_mass
        angular = (self.GRAVITY * sine - cosine * shared) / (
            self.POLE_HALF_LENGTH
            * (4.0 / 3.0 - self.POLE_MASS * cosine**2 / self.total_mass)
        )
        linear = shared - self.pole_moment * angular * cosine / self.total_mass

        # Semi-implicit Euler. The speed moves first, then the position moves
        # with the speed it now has.
        self.speed += self.TIMESTEP * linear
        self.position += self.TIMESTEP * self.speed
        self.spin += self.TIMESTEP * angular
        self.angle += self.TIMESTEP * self.spin

        self._steps += 1
        if (
            abs(self.angle) < self.STEADY_ANGLE
            and abs(self.position) < self.STEADY_POSITION
        ):
            self._steady += 1

        fallen = (
            abs(self.angle) > self.ANGLE_LIMIT or abs(self.position) > self.TRACK_LIMIT
        )
        return Step(self._observation(), 1.0, terminated=fallen, truncated=False)

    def _observation(self) -> Observation:
        return (self.position, self.speed, self.angle, self.spin)

    def audit(self) -> Mapping[str, float]:
        """How well it was held up, not merely whether it fell.

        The reward pays 1 for every step the pole has not fallen, and a pole
        wobbling at eleven degrees earns exactly what a pole standing still
        earns. This reports the share of the episode spent near upright and
        near the middle of the track, which is what a person watching would
        call balanced.
        """
        if self._steps == 0:
            return {"steady_share": 0.0}
        return {"steady_share": self._steady / self._steps}

    def render(self) -> str:
        width = 41
        middle = width // 2
        spot = middle + round(self.position / self.TRACK_LIMIT * middle)
        spot = min(max(spot, 0), width - 1)

        track = ["-"] * width
        track[spot] = "#"

        lean = round(self.angle / self.ANGLE_LIMIT * 4)
        pole = [" "] * width
        for height in range(1, 5):
            column = spot + round(lean * height / 4)
            if 0 <= column < width:
                pole[column] = "|"

        return "\n".join(
            [
                "".join(pole),
                "".join(track),
                f"x={self.position:+.2f}  angle={math.degrees(self.angle):+.1f} deg",
            ]
        )


class MountainCar(DiscreteEnv[Observation]):
    """An underpowered car in a valley. Rock back and forth to climb out.

    The engine cannot climb the hill directly. The only way up is to drive
    backward first and use the slope on the other side, which is why a greedy
    step by step view of the problem fails on it.

    Every step pays -1, so the return is the number of steps taken and shorter
    is better.
    """

    LOWEST = -1.2
    HIGHEST = 0.6
    GOAL = 0.5
    TOP_SPEED = 0.07
    ENGINE = 0.001
    SLOPE = 0.0025

    def __init__(self, rng: Rng, max_episode_steps: int = 1000) -> None:
        super().__init__(rng)

        self.observation_space = Box(
            [self.LOWEST, -self.TOP_SPEED],
            [self.HIGHEST, self.TOP_SPEED],
            names=["position", "speed"],
        )

        # Both are clamped by the physics, so the box a tile coder divides is
        # the box the environment reports.
        self.tiling_space = self.observation_space
        self.action_space = Discrete(3)
        self.spec = EnvSpec(
            name="mountaincar",
            summary="Rock a car out of a valley that its engine cannot climb.",
            max_episode_steps=max_episode_steps,
            solved_return=-110.0,
        )

        self.position = -0.5
        self.speed = 0.0
        self.peak = self.LOWEST
        self._arrived = False

    def _reset(self) -> Observation:
        self.position = self.rng.uniform(-0.6, -0.4)
        self.speed = 0.0
        self.peak = self.position
        self._arrived = False
        return self._observation()

    def _step(self, action: int) -> Step[Observation]:
        push = action - 1  # -1 back, 0 nothing, 1 forward

        self.speed += push * self.ENGINE - self.SLOPE * math.cos(3.0 * self.position)
        self.speed = min(max(self.speed, -self.TOP_SPEED), self.TOP_SPEED)

        self.position += self.speed
        self.position = min(max(self.position, self.LOWEST), self.HIGHEST)

        # The wall at the far end of the valley takes the speed away. Without
        # this the car bounces off it and keeps the energy, which turns a hard
        # problem into an easy one.
        if self.position <= self.LOWEST and self.speed < 0.0:
            self.speed = 0.0

        self.peak = max(self.peak, self.position)
        arrived = self.position >= self.GOAL
        self._arrived = self._arrived or arrived

        return Step(self._observation(), -1.0, terminated=arrived, truncated=False)

    def _observation(self) -> Observation:
        return (self.position, self.speed)

    def audit(self) -> Mapping[str, float]:
        """How far up the hill it got, and whether it arrived.

        The return of a truncated episode is the step limit, whatever the car
        did. A car that reached 0.49 and one that never left the bottom look
        the same in the reward and are not the same at all.
        """
        return {
            "peak_position": self.peak,
            "arrived": 1.0 if self._arrived else 0.0,
        }

    def render(self) -> str:
        width = 45
        span = self.HIGHEST - self.LOWEST
        spot = round((self.position - self.LOWEST) / span * (width - 1))
        goal = round((self.GOAL - self.LOWEST) / span * (width - 1))

        hill = []
        for column in range(width):
            position = self.LOWEST + column * span / (width - 1)
            hill.append("_" if math.sin(3.0 * position) < 0 else "/")
        hill[goal] = "G"

        car = [" "] * width
        car[min(max(spot, 0), width - 1)] = "#"

        return "\n".join(
            [
                "".join(car),
                "".join(hill),
                f"position={self.position:+.3f}  speed={self.speed:+.4f}",
            ]
        )


__all__ = ["CartPole", "MountainCar", "Observation"]
