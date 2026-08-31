"""A box of actions cut into a short list of them.

Every agent in this project but one chooses from a list. An environment whose
action is a number cannot be handed to any of them, and the usual answer is to
cut the box into levels and let the agent choose a level.

    levels=2   the two ends, which is a switch
    levels=3   the two ends and the middle
    levels=9   eight equal gaps between the ends

That is a real method and not a workaround, and this is here so that what it
costs can be measured rather than assumed. `scripts/measure_levels.py` runs the
same agent over the same environment at several counts and reads where the
curve stops moving.

## Two things it cannot do

**It grows as a power.** A box of `d` dimensions cut into `n` levels is `n` to
the power `d` actions, so a two dimensional box at nine levels is 81 and a
four dimensional one is 6561. Every one of them is a column of a table or an
output of a network, so past two dimensions this stops being a method.

**It cannot be finer than its levels.** A policy that wants a torque of 0.3
where the nearest level is 0.5 has to take 0.5 and pay for it. Whether that
matters is a question about the environment, and on the pendulum it turns out
to matter less than the count of levels would suggest.

## What it forwards

Everything but the action. The observation space, the box a tile coder
divides, the spec, the drawing and the audit all come from the environment
inside, so a run of the cut version is a run of the same problem.

The step limit is on both, and they agree because one is copied from the
other, so the two truncate on the same step rather than one of them stopping
the other early.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rel.core import DiscreteEnv, Env, EnvSpec, ObsT, Step
from rel.envs.pendulum import Pendulum
from rel.rng import Rng
from rel.spaces import Box, Discrete


class Levels(DiscreteEnv[ObsT]):
    """An environment whose box of actions has been cut into a list of them."""

    def __init__(self, inside: Env[ObsT, tuple[float, ...]], levels: int = 9) -> None:
        super().__init__(inside.rng)

        if levels < 2:
            raise ValueError("A cut needs at least the two ends.")
        box = inside.action_space
        if not isinstance(box, Box):
            raise TypeError(
                f"{inside.spec.name} already takes an action from "
                f"{box!r}, so there is nothing to cut."
            )

        self.inside = inside
        self.levels = levels
        self.box = box

        self.observation_space = inside.observation_space
        self.action_space = Discrete(levels**box.dimensions)
        self.spec = EnvSpec(
            name=f"{inside.spec.name}-levels",
            summary=inside.spec.summary,
            max_episode_steps=inside.spec.max_episode_steps,
            solved_return=inside.spec.solved_return,
            ends=inside.spec.ends,
            suggested_discount=inside.spec.suggested_discount,
        )

        tiling = getattr(inside, "tiling_space", None)
        if tiling is not None:
            self.tiling_space = tiling

    # -- The cut ------------------------------------------------------------

    def torque(self, action: int) -> tuple[float, ...]:
        """The point in the box that this action stands for.

        The action is read as a number in base `levels`, one digit for each
        dimension, lowest dimension first. At two levels the digits pick out
        the corners of the box and at more they pick out an even grid over it,
        with a level on each bound rather than inside them: an environment
        driven by a motor needs full power to be reachable.
        """
        if not self.action_space.contains(action):
            raise IndexError(f"{action} is not one of {self.action_space!r}.")

        chosen = []
        left = action
        for low, high in zip(self.box.low, self.box.high, strict=True):
            digit = left % self.levels
            left //= self.levels
            chosen.append(low + (high - low) * digit / (self.levels - 1))
        return tuple(chosen)

    def every_torque(self) -> Sequence[tuple[float, ...]]:
        """Every point the cut can reach, in the order the actions are in."""
        return [self.torque(action) for action in self.action_space]

    # -- The contract, forwarded --------------------------------------------

    def _reset(self) -> ObsT:
        return self.inside.reset()

    def _step(self, action: int) -> Step[ObsT]:
        return self.inside.step(self.torque(action))

    def audit(self) -> Mapping[str, float]:
        return self.inside.audit()

    def render(self) -> str:
        return self.inside.render()

    def __repr__(self) -> str:
        return f"Levels({self.inside!r}, levels={self.levels})"


def levelled_pendulum(rng: Rng, levels: int = 9, steps: int = 200) -> Levels[Any]:
    """The pendulum with its torque cut into `levels` settings."""
    return Levels(Pendulum(rng, max_episode_steps=steps), levels=levels)


__all__ = ["Levels", "levelled_pendulum"]
