#!/usr/bin/env python3
"""How close each way of estimating a fixed policy gets, on the one problem
whose answer is known.

Every other measurement in this project compares an agent against dynamic
programming over the same model. On the random walk the values are arithmetic:
the k-th cell is worth k over the number of gaps under a policy that goes each
way half the time. So the error of an estimate is a number rather than a
comparison.

    python scripts/measure_prediction.py
    python scripts/measure_prediction.py --episodes 200 --runs 100

This is Sutton and Barto, example 6.2, and the question is the one that example
asks: is it better to wait and see what the return was, or to use what you
already believe about where you ended up?

The step size is a constant here on purpose. Every one of these methods tracks
rather than converges at a constant step size, settling into a band around the
answer whose width is proportional to it, so a single step size would measure
that step size. The ladder is the measurement.

`error` is the root mean square error over the five states an agent can be in,
averaged over the runs. The two endings are left out: an agent is never in one,
so its entry there never moves and scoring it would measure the starting value.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.prediction import (
    MonteCarloPrediction,
    NStepTD,
    Predictor,
    TDLambda,
    TemporalDifference,
)
from rel.envs.classic import random_walk
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

#: Where the estimates start. Half way is the classic choice and it is not a
#: neutral one: it happens to be the exact value of the middle state, so the
#: middle starts right and everything else starts wrong in a symmetric way.
START = 0.5

LADDER = (0.01, 0.02, 0.05, 0.1, 0.2)

METHODS = {
    "td": lambda rng, env, step: TemporalDifference(
        rng, env.action_space, step_size=step, start_value=START
    ),
    "3-step td": lambda rng, env, step: NStepTD(
        rng, env.action_space, n=3, step_size=step, start_value=START
    ),
    "td-lambda 0.8": lambda rng, env, step: TDLambda(
        rng, env.action_space, trace_decay=0.8, step_size=step, start_value=START
    ),
    "mc, first visit": lambda rng, env, step: MonteCarloPrediction(
        rng, env.action_space, step_size=step, start_value=START
    ),
    "mc, every visit": lambda rng, env, step: MonteCarloPrediction(
        rng, env.action_space, first_visit=False, step_size=step, start_value=START
    ),
}


def measure(name: str, size: int, episodes: int, runs: int, step: float) -> float:
    errors: list[float] = []
    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = random_walk(root.stream("env"), size=size)
        agent = METHODS[name](root.stream("agent"), env, step)
        train(env, agent, episodes)
        errors.append(agent.error_against(env.values_to_score()))
    return statistics.mean(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--steps", type=float, nargs="+", default=list(LADDER))
    args = parser.parse_args()

    probe = random_walk(Rng(1), size=args.size)
    truth = probe.values_to_score()
    untrained = Predictor(Rng(1), probe.action_space, start_value=START).error_against(
        truth
    )

    print(
        f"walk of {args.size}, {args.episodes} episodes, "
        f"{args.runs} runs, estimates starting at {START:g}.\n"
        f"The true values are {', '.join(f'{worth:.3f}' for worth in truth.values())}, "
        f"and an untrained table scores {untrained:.4f}."
    )

    rows = []
    for name in METHODS:
        row = [name]
        row += [
            f"{measure(name, args.size, args.episodes, args.runs, step):.4f}"
            for step in args.steps
        ]
        rows.append(row)

    headings = ["method"] + [f"step {step:g}" for step in args.steps]
    print()
    for line in table(headings, rows, align=["left"] + ["right"] * len(args.steps)):
        print(f"  {line}")

    print(
        "\nEach cell is the root mean square error against the true values,\n"
        "over the states an agent can be in, averaged over the runs. Lower is\n"
        "closer to the answer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
