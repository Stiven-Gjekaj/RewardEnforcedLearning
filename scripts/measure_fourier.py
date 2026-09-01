#!/usr/bin/env python3
"""Waves over the whole box, and the one setting the literature attaches.

A Fourier basis is `(order + 1)` to the power of the dimensions cosine waves
over the box, and an order is the whole design. No bins, no widths, no
centres, no offsets between grids. That is what it is for.

    python scripts/measure_fourier.py
    python scripts/measure_fourier.py --orders 3 5 --runs 3
    python scripts/measure_fourier.py --env mountaincar cartpole

The one thing said about it that is not the basis itself is that the step size
should be divided for each feature by how fast that feature waves. This runs
the same problem with the division and without it.

**Both are swept.** Every scale is at most one, so scaling makes every step
smaller as well as uneven. A comparison at one step size would report "smaller
steps are better here" and call it "uneven steps are better here". So each
side is swept over the same step sizes and each is read at its own best.

**And the sweep says when it did not bracket a side.** Reading a side at its
own best is only fair if that best is really in the range. A side whose best
sits at an end of the swept step sizes might do better outside it, so the
report names those rows rather than letting them read as a finding.

**The gap between the two is tested.** A mean beside a mean says which is
larger and says nothing about whether the seeds agree, so each row carries the
paired interval and the p value from `rel.compare`.

`docs/algorithms.md` has the tables.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.fourier import FourierBasis
from rel.compare import compare
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.spaces import Box
from rel.training import evaluate, train
from rel.ui.table import table

#: The orders to run. Order 1 is four features on a two dimensional box and
#: order 7 is sixty four, which is the growth this basis has and the reason it
#: does not go past a few dimensions.
ORDERS: tuple[int, ...] = (1, 3, 5, 7)

#: The step sizes each side is swept over. Both sides see all of them, and the
#: range runs past what either side wants so that neither is read at an end.
STEP_SIZES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)

ENVIRONMENT = "mountaincar"
EPISODES = 200
RUNS = 5
DISCOUNT = 1.0


def one_setting(
    order: int,
    scaled: bool,
    step_size: float,
    runs: int,
    episodes: int,
    env_name: str = ENVIRONMENT,
) -> list[float]:
    """The greedy return of each seed after training it at this setting.

    Every seed rather than their mean, because the two sides are compared
    paired: seed 3 of one against seed 3 of the other, on the same walk from
    the same start.
    """
    got = []
    for seed in range(1, runs + 1):
        env = ENVIRONMENTS.make(env_name, Rng(seed).stream("env"))
        agent = AGENTS.make(
            "fourier-sarsa",
            Rng(seed).stream("agent"),
            env,
            order=order,
            scaled_steps=scaled,
            step_size=step_size,
            discount=DISCOUNT,
            epsilon=0.0,
        )
        train(env, agent, episodes, discount=DISCOUNT)
        got.append(statistics.mean(evaluate(env, agent, 10, discount=DISCOUNT).returns))
    return got


def best_of(
    order: int,
    scaled: bool,
    steps: tuple[float, ...],
    runs: int,
    episodes: int,
    env_name: str = ENVIRONMENT,
) -> tuple[list[float], float]:
    """Every seed at the best of the swept step sizes, and the step that won.

    Best is the largest mean return. These returns are negative, so a sweep
    that took the smallest would report the worst setting of each side.
    """
    got = {
        step: one_setting(order, scaled, step, runs, episodes, env_name)
        for step in steps
    }
    best = max(got, key=lambda step: statistics.mean(got[step]))
    return got[best], best


def at_an_end(step: float, steps: tuple[float, ...]) -> bool:
    """Whether a best step size sits at either end of what was swept.

    A side whose best is at an end might do better outside the range, so the
    comparison at that row is between one side at its best and the other at
    the edge of what it was allowed. One value swept is both ends at once.
    """
    return step in (min(steps), max(steps))


def features_of(order: int, env_name: str = ENVIRONMENT) -> int:
    env = ENVIRONMENTS.make(env_name, Rng(1).stream("env"))
    box = getattr(env, "tiling_space", env.observation_space)
    assert isinstance(box, Box)
    return FourierBasis(box, order=order).features


def scales_section(
    orders: tuple[int, ...],
    steps: tuple[float, ...],
    runs: int,
    episodes: int,
    rng: Rng,
    env_name: str = ENVIRONMENT,
) -> None:
    print(
        f"Dividing the step size by the speed of each wave, against not "
        f"doing it.\nBoth sides swept over {len(steps)} step sizes and read "
        f"at their own best.\n{env_name}, {runs} seeds, {episodes} "
        f"episodes.\n"
    )

    rows = []
    unbracketed = []
    for order in orders:
        scaled, at_scaled = best_of(order, True, steps, runs, episodes, env_name)
        flat, at_flat = best_of(order, False, steps, runs, episodes, env_name)
        answer = compare(scaled, flat, rng)
        rows.append(
            [
                str(order),
                str(features_of(order, env_name)),
                f"{statistics.mean(scaled):.1f}",
                f"{at_scaled:g}",
                f"{statistics.mean(flat):.1f}",
                f"{at_flat:g}",
                f"{answer.difference:+.1f}",
                f"[{answer.low:+.1f}, {answer.high:+.1f}]",
                f"{answer.p_value:.3f}",
            ]
        )
        for side, step in (("scaled", at_scaled), ("flat", at_flat)):
            if at_an_end(step, steps):
                unbracketed.append(f"order {order}, {side}, at {step:g}")

    for line in table(
        [
            "order",
            "features",
            "scaled",
            "at step",
            "flat",
            "at step",
            "scaled minus flat",
            "95 percent interval",
            "p",
        ],
        rows,
        align=["right"] * 7 + ["right", "right"],
    ):
        print(f"  {line}")

    if unbracketed:
        print(
            "\n  Best at an end of the swept step sizes, so the sweep did not"
            "\n  bracket it and the number above is a floor rather than a best:"
        )
        for line in unbracketed:
            print(f"    {line}")

    floor = 2.0 / 2.0**runs
    if floor > 0.05:
        print(
            f"\n  {runs} seeds cannot give a p below {floor:.4f} however large"
            f" the difference is,\n  because a paired test over {runs} seeds"
            f" has only {2**runs} sign patterns in it.\n  Six seeds is the"
            f" fewest that can reach 0.05."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=list(ORDERS))
    parser.add_argument("--step-sizes", type=float, nargs="+", default=list(STEP_SIZES))
    parser.add_argument("--runs", type=int, default=RUNS, help="seeds per setting")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument(
        "--env",
        nargs="+",
        default=[ENVIRONMENT],
        help="the environments to run, one table each",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    for index, env_name in enumerate(args.env):
        if index:
            print()
        # A fresh comparison source for each environment, so the table for one
        # is the same table whether or not another was asked for beside it.
        scales_section(
            tuple(args.orders),
            tuple(args.step_sizes),
            args.runs,
            args.episodes,
            Rng(9).stream("compare"),
            env_name,
        )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
