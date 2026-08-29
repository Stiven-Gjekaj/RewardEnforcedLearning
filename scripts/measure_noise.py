#!/usr/bin/env python3
"""How large a difference two identical agents show, which is the floor.

Every comparison in this project is between two agents on a handful of seeds. A
difference smaller than what one agent shows against a copy of itself is not a
difference, and the only way to know where that line is, is to measure it.

    python scripts/measure_noise.py
    python scripts/measure_noise.py --agent sarsa --env maze --trials 200
    python scripts/measure_noise.py --runs 10

The construction is the whole thing. One agent, one environment, and two runs
per seed that differ only in the agent's own source of chance. The environment
seed is shared, so the pairing is real, and the agents are the same code with
the same settings, so **every difference this prints is noise**.

Each run gets its own agent seed. Giving one agent seed to a whole comparison
makes the paired differences lean together, which answers "is this agent seed
better than that one" instead. The first version of this did that and reported
an interval excluding zero on every comparison it made.

What comes out is three numbers a reader of any other table on this project
needs: how large a difference turns up by chance, how often a bootstrap
interval excludes zero when there is nothing there, and how often the
permutation test says the difference is real when it is not.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.compare import compare
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table


def one_run(
    grid: str, agent_name: str, env_seed: int, agent_seed: int, episodes: int
) -> float:
    """What the agent got over the last hundred episodes of one run.

    The two seeds are separate on purpose. Sharing the environment seed is
    what pairs the runs, and differing in the agent seed is what makes the
    pair a measurement of noise rather than of nothing.
    """
    env = ENVIRONMENTS.make(grid, Rng(env_seed).stream("env"))
    discount = env.spec.suggested_discount
    agent = AGENTS.make(agent_name, Rng(agent_seed).stream("agent"), env)
    return train(env, agent, episodes, discount=discount).final(100)


def one_trial(
    grid: str, agent_name: str, runs: int, episodes: int, trial: int, rng: Rng
) -> tuple[float, bool, bool]:
    """One whole comparison of an agent against a copy of itself."""
    ours: list[float] = []
    theirs: list[float] = []

    for seed in range(1, runs + 1):
        # A fresh agent seed for every run of every trial. The first version
        # of this gave one agent seed to a whole trial, so the five paired
        # differences all leaned whichever way that seed leaned, and the
        # interval excluded zero on twenty comparisons of twenty. That is a
        # correct answer to "is agent seed X better than agent seed Y" and it
        # is not the question.
        mine = 10_000 + trial * runs + seed
        yours = 500_000 + trial * runs + seed
        ours.append(one_run(grid, agent_name, seed, mine, episodes))
        theirs.append(one_run(grid, agent_name, seed, yours, episodes))

    answer = compare(ours, theirs, rng)
    return abs(answer.difference), answer.certain, answer.p_value < 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="cliff")
    parser.add_argument("--agent", default="q-learning")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5, help="seeds per comparison")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    rng = Rng(7).stream("compare")
    gaps: list[float] = []
    certain = 0
    significant = 0
    started = time.perf_counter()

    for trial in range(args.trials):
        gap, was_certain, was_significant = one_trial(
            args.env, args.agent, args.runs, args.episodes, trial, rng
        )
        gaps.append(gap)
        certain += was_certain
        significant += was_significant

    floor = 2.0 / 2.0**args.runs
    print(
        f"{args.agent} against a copy of itself on {args.env}, "
        f"{args.trials} comparisons of {args.runs} seeds, "
        f"{args.episodes} episodes each.\n"
        f"Everything below is noise by construction: the two sides are the "
        f"same code with the same settings."
    )

    rows = [
        ("difference, median", f"{statistics.median(gaps):.2f}"),
        ("difference, nine in ten under", f"{sorted(gaps)[int(0.9 * len(gaps))]:.2f}"),
        ("difference, largest seen", f"{max(gaps):.2f}"),
        ("interval excluded zero", f"{certain} of {args.trials}"),
        ("p below 0.05", f"{significant} of {args.trials}"),
        ("smallest p these seeds allow", f"{floor:.4f}"),
        ("time", f"{time.perf_counter() - started:.0f}s"),
    ]

    print()
    for line in table(["", "value"], rows, align=["left", "right"]):
        print(f"  {line}")

    print(
        "\nA difference smaller than the median line is not a difference. The\n"
        "'interval excluded zero' line is how often a reader taking the\n"
        "interval as a verdict would have been wrong, and it should sit near\n"
        "one in twenty rather than at zero: an interval that never excludes\n"
        "zero on noise is too wide to be useful."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
