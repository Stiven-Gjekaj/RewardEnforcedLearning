#!/usr/bin/env python3
"""What a replay buffer and a target network each buy a value network.

`deep-q` is Q-learning with a network in place of the table. A table has one
number per state and action and moving one never moves another. A network has
one set of weights for all of them, so two things that are harmless with a
table become the reason a value network diverges: the steps arrive correlated,
and the target moves with the estimate because the same weights produce both.

A buffer answers the first and a second copy of the weights answers the second.
Both are settings of one agent, so the four rows below are the same code with
two numbers changed.

    python scripts/measure_value_network.py --env cliff
    python scripts/measure_value_network.py --env cartpole --episodes 400

`replay=0` learns from the step just taken and nothing else. `target_refresh=0`
takes the target from the live network, so it moves with every update.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.dp import DidNotSettleError, value_iteration
from rel.core import TabularEnv
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table


def measure(
    grid: str,
    episodes: int,
    runs: int,
    replay: int,
    refresh: int,
    settings: dict[str, object],
) -> tuple[list[float], float]:
    """What every seed ended at, and the seconds the whole row took."""
    finals: list[float] = []
    started = time.perf_counter()

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(grid, root.stream("env"))
        discount = env.spec.suggested_discount
        agent = AGENTS.make(
            "deep-q",
            root.stream("agent"),
            env,
            replay=replay,
            target_refresh=refresh,
            **settings,
        )
        record = train(env, agent, episodes, discount=discount)
        finals.append(statistics.mean(record.returns[-50:]))

    return finals, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="cliff")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--replay", type=int, default=2000)
    parser.add_argument("--refresh", type=int, default=200)
    parser.add_argument(
        "--set",
        action="append",
        metavar="NAME=VALUE",
        help="a setting held fixed across all four rows, repeatable",
    )
    parser.add_argument(
        "--each-seed",
        action="store_true",
        help="print the number every seed ended at, under the table",
    )
    args = parser.parse_args()

    from rel.cli import parse_settings

    settings = parse_settings(args.set)

    probe = ENVIRONMENTS.make(args.env, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount
    best = "-"
    if isinstance(probe, TabularEnv):
        try:
            best = f"{value_iteration(probe, discount=discount).start_value:.2f}"
        except DidNotSettleError:
            best = "-"

    print(
        f"{args.env}, {args.episodes} episodes, seeds 1 to {args.runs}, "
        f"discount {discount:g}.\nThe best possible return is {best}."
    )
    if settings:
        print("Held fixed: " + ", ".join(f"{k}={v}" for k, v in settings.items()))

    rows = []
    every: list[tuple[str, list[float]]] = []
    for replay in (args.replay, 0):
        for refresh in (args.refresh, 0):
            finals, seconds = measure(
                args.env, args.episodes, args.runs, replay, refresh, settings
            )
            label = f"{'on' if replay else 'off'} / {'on' if refresh else 'off'}"
            every.append((label, finals))
            rows.append(
                (
                    "on" if replay else "off",
                    "on" if refresh else "off",
                    f"{statistics.median(finals):.1f}",
                    f"{statistics.mean(finals):.1f}",
                    f"{min(finals):.1f}",
                    f"{max(finals):.1f}",
                    f"{seconds:.0f}s",
                )
            )

    print()
    for line in table(
        ["replay", "target", "median", "mean", "worst", "best", "time"],
        rows,
        align=["right"] * 7,
    ):
        print(f"  {line}")

    if args.each_seed:
        print()
        for label, finals in every:
            numbers = " ".join(f"{one:8.1f}" for one in finals)
            print(f"  {label:9s} {numbers}")

    print(
        "\n'median' rather than 'mean' first, and 'worst' beside it. A value\n"
        "network that diverges on one seed of ten moves a mean by more than\n"
        "the difference between any two of these rows, so the mean is the\n"
        "number least worth reading here. --each-seed prints them all."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
