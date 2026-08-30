#!/usr/bin/env python3
"""What the exploration bonus is worth when the world changes under an agent.

Sutton and Barto, figure 8.5. A wall with a gap on the left, and after sixty
episodes a second gap opens on the right that is much shorter. Plain Dyna-Q has
a route that works and no reason to look for another, so it keeps the long one.
`dyna-q-plus` adds `kappa` times the square root of how long it has been since
a state and action were tried, which is a reason to look.

    python scripts/measure_shortcut.py
    python scripts/measure_shortcut.py --runs 10
    python scripts/measure_shortcut.py --kappas 0.001 0.002

The second thing it measures is the size of the bonus. It is added to a
remembered reward, so it has to be small against the rewards the environment
really pays. Here the goal pays one and nothing else pays anything, so a
`kappa` of 0.01 passes that after a thousand steps: the planning stops being
about the environment and the agent wanders.

`docs/algorithms.md` has the table. It had the table before it had this script,
which is what `scripts/check_numbers.py` was written to find.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.dyna import DynaQ, DynaQPlus
from rel.envs.gridworld import GridWorld
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

#: The wall with one gap, on the left.
BEFORE = (
    "........G",
    ".########",
    ".........",
    ".........",
    ".........",
    "...S.....",
)

#: The same wall with a second gap, on the right, which is the shorter way.
AFTER = (
    "........G",
    ".#######.",
    ".........",
    ".........",
    ".........",
    "...S.....",
)

#: Everything both agents share. The step limit is high because an agent that
#: has been wrecked by too large a bonus wanders for hundreds of steps, and a
#: limit that cut it off would report the limit rather than the wandering.
SETTINGS: dict[str, float] = {
    "planning_steps": 20,
    "step_size": 0.5,
    "epsilon": 0.1,
    "discount": 0.95,
}
STEPS = 3000
#: Episodes before the shortcut opens, and after.
LEARNING = 60
AFTERWARDS = 120
#: How many of the last episodes the mean is taken over.
TAIL = 40


def one_run(
    seed: int,
    kappa: float | None,
    learning: int = LEARNING,
    afterwards: int = AFTERWARDS,
) -> float:
    """The mean episode length over the last `TAIL` episodes after the change.

    The agent carries its table and its model across the change, which is the
    whole point: it has already learned a route, and the question is whether
    it ever finds out that a better one exists.
    """
    rng = Rng(seed)
    grid = GridWorld(
        rng.stream("env"),
        BEFORE,
        step_reward=0.0,
        goal_reward=1.0,
        max_episode_steps=STEPS,
    )
    if kappa is None:
        agent: DynaQ[int] = DynaQ(rng.stream("agent"), grid.action_space, **SETTINGS)
    else:
        agent = DynaQPlus(
            rng.stream("agent"), grid.action_space, kappa=kappa, **SETTINGS
        )
    train(grid, agent, learning)

    opened = GridWorld(
        Rng(seed).stream("moved"),
        AFTER,
        step_reward=0.0,
        goal_reward=1.0,
        max_episode_steps=STEPS,
    )
    record = train(opened, agent, afterwards)

    # Divided by how many there are rather than by `TAIL`. A run shorter than
    # the tail would otherwise report a mean over forty episodes when it had
    # twenty, which is half the truth and looks like a result.
    tail = record.lengths[-TAIL:]
    return sum(tail) / len(tail)


def measure(
    kappa: float | None,
    runs: int,
    learning: int = LEARNING,
    afterwards: int = AFTERWARDS,
) -> list[float]:
    return [one_run(seed, kappa, learning, afterwards) for seed in range(1, runs + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="seeds per setting")
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    parser.add_argument("--before", type=int, default=LEARNING)
    parser.add_argument("--after", type=int, default=AFTERWARDS)
    args = parser.parse_args()

    started = time.perf_counter()
    print(
        f"A wall with one gap, then a second gap opens. {args.runs} seeds, "
        f"{args.before} episodes\nbefore the shortcut and {args.after} after. "
        f"The mean is over the last {min(TAIL, args.after)}.\n"
    )

    rows: list[list[str]] = []
    for name, kappa in [("dyna-q", None)] + [
        (f"dyna-q-plus, kappa {kappa:g}", kappa) for kappa in args.kappas
    ]:
        lengths = measure(kappa, args.runs, args.before, args.after)
        rows.append(
            [
                name,
                f"{statistics.mean(lengths):.1f}",
                " ".join(f"{length:.0f}" for length in lengths),
            ]
        )

    for line in table(
        ["agent", "mean episode length after a shortcut opens", "every seed"],
        rows,
        align=["left", "right", "left"],
    ):
        print(f"  {line}")

    print(
        "\nThe shortest way after the change is through the new gap. A bonus\n"
        "large against the rewards the environment pays wrecks the planning:\n"
        "the goal pays one here, so a kappa of 0.01 passes that after a\n"
        "thousand steps and the agent wanders."
    )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
