"""Every environment the command line can build, declared once.

Adding an environment means one entry here and one module. From the entry come
the name that `--env` accepts, the line in `--list`, the group that a benchmark
runs, and the table in the documentation.

## The tags

- `tabular` says the environment has a finite state set and can describe its
  own model. Dynamic programming works on these, so a run can be compared with
  the best possible return rather than with a number somebody remembered.
- `continuous` says the observation is real numbers. A table does not fit, and
  an agent needs a tile coder or a network.
- `bandit` says there is one state. The whole problem is which lever to pull.
- `gaming` says the reward the environment pays is not the thing it wants. Each
  one carries a `reward` or a penalty setting that switches between the
  specification that can be gamed and one that cannot, so the difference can be
  run rather than argued about. See `docs/specification-gaming.md`.

Two of the gaming environments never end. A discount below one is needed to
solve or to learn them, because a run of them that goes on for ever is worth
an unbounded amount and nothing converges. `value_iteration` says so rather
than returning a number.
"""

from __future__ import annotations

from typing import Any

from rel.core import Env
from rel.envs.bandit import KArmedBandit, drifting_bandit, stationary_bandit
from rel.envs.classic import (
    cliff_walk,
    dyna_maze,
    four_rooms,
    frozen_lake,
    windy_grid,
)
from rel.envs.control import CartPole, MountainCar
from rel.envs.gaming import (
    BoatRace,
    Thermostat,
    VaseRoom,
    boat_race,
    thermostat,
    vase_room,
)
from rel.envs.gridworld import GridWorld
from rel.registry import Entry, Registry

ENVIRONMENTS: Registry[Env[Any]] = Registry(
    "environment",
    [
        Entry(
            "bandit",
            "Ten levers with no state. New true values every episode.",
            stationary_bandit,
            tags=("bandit",),
        ),
        Entry(
            "bandit-drift",
            "Ten levers that all start equal and then wander.",
            drifting_bandit,
            tags=("bandit",),
        ),
        Entry(
            "cliff",
            "Four by twelve. The short path runs along a cliff that pays -100.",
            cliff_walk,
            tags=("tabular", "grid"),
        ),
        Entry(
            "windy",
            "Seven by ten, with an upward wind in six of the columns.",
            windy_grid,
            tags=("tabular", "grid"),
        ),
        Entry(
            "rooms",
            "Four rooms with one gap between each pair.",
            four_rooms,
            tags=("tabular", "grid"),
        ),
        Entry(
            "maze",
            "Six by nine with seven blocked cells. Only the goal pays.",
            dyna_maze,
            tags=("tabular", "grid"),
        ),
        Entry(
            "lake",
            "A slippery lake. A hole ends the episode and pays nothing.",
            frozen_lake,
            tags=("tabular", "grid"),
        ),
        Entry(
            "cartpole",
            "Keep a hinged pole upright by pushing the cart it stands on.",
            CartPole,
            tags=("continuous",),
        ),
        Entry(
            "mountaincar",
            "Rock a car out of a valley that its engine cannot climb.",
            MountainCar,
            tags=("continuous",),
        ),
        Entry(
            "boatrace",
            "A boat paid for touching checkpoints, two of which sit side by side.",
            boat_race,
            tags=("tabular", "gaming"),
        ),
        Entry(
            "vase",
            "A room with a goal at the other end and a vase on the short way.",
            vase_room,
            tags=("tabular", "gaming"),
        ),
        Entry(
            "thermostat",
            "A room to keep warm, with a dial that makes the sensor report comfort.",
            thermostat,
            tags=("tabular", "gaming"),
        ),
    ],
)

__all__ = [
    "ENVIRONMENTS",
    "BoatRace",
    "CartPole",
    "GridWorld",
    "KArmedBandit",
    "MountainCar",
    "Thermostat",
    "VaseRoom",
]
