#!/usr/bin/env python3
"""Which loop a discount chooses, and what that choice collects per step.

The two loops pay 1 a step and 2 a step. Below a discount of about 0.74 the
first of them has the higher discounted value, so a discounted agent takes it,
and it is right to: the discount was part of the question.

    python scripts/measure_average_reward.py
    python scripts/measure_average_reward.py --episodes 40 --runs 5
    python scripts/measure_average_reward.py --length 8

Two rows of the table come from the model and not from a run. `value iteration`
is the exactly optimal policy under each discount, so the flip is a property of
the question rather than something an agent failed to learn. The learned rows
say whether `q-learning` finds the same answer, which it should and does.

The last row has no discount in it at all.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.dp import average_reward, value_iteration
from rel.envs.continuing import LONG, TwoLoops
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

DISCOUNTS = (0.5, 0.7, 0.735, 0.745, 0.8, 0.9, 0.99)


def loop_name(action: int) -> str:
    return "long" if action == LONG else "short"


def solved(env: TwoLoops, discount: float) -> tuple[str, float]:
    """Which loop the exactly optimal policy takes, and what it collects."""
    chosen = list(value_iteration(env, discount=discount).policy)
    collected = average_reward(env, chosen)
    return loop_name(chosen[0]), 0.0 if collected is None else collected


def learned(
    length: int,
    agent_name: str,
    settings: dict[str, object],
    episodes: int,
    runs: int,
) -> tuple[str, float]:
    """Which loop the seeds mostly took, and the median they collected."""
    taken: list[int] = []
    rates: list[float] = []

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = TwoLoops(root.stream("env"), length=length)
        agent = AGENTS.make(agent_name, root.stream("agent"), env, **settings)
        train(env, agent, episodes, discount=1.0)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        taken.append(policy[0])
        collected = average_reward(env, policy)
        rates.append(0.0 if collected is None else collected)

    long_ones = sum(1 for one in taken if one == LONG)
    if long_ones == len(taken):
        which = "long"
    elif long_ones == 0:
        which = "short"
    else:
        which = f"{long_ones} of {len(taken)} long"
    return which, statistics.median(rates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=0.1)
    args = parser.parse_args()

    probe = TwoLoops(Rng(1).stream("env"), length=args.length)
    crossover = probe.crossover()

    print(
        f"loops of length {args.length}, {args.episodes} episodes of "
        f"{probe.spec.max_episode_steps} steps, seeds 1 to {args.runs}.\n"
        f"The short loop pays {probe.per_step(0):.2f} a step and the long one "
        f"pays {probe.per_step(LONG):.2f}.\n"
        f"The two have equal discounted value at a discount of "
        f"{crossover:.4f}."
    )

    rows = []
    for discount in DISCOUNTS:
        exact_loop, exact_rate = solved(probe, discount)
        found_loop, found_rate = learned(
            args.length,
            "q-learning",
            {"discount": discount, "step_size": args.step_size},
            args.episodes,
            args.runs,
        )
        rows.append(
            (
                f"{discount:g}",
                exact_loop,
                f"{exact_rate:.2f}",
                found_loop,
                f"{found_rate:.2f}",
            )
        )

    found_loop, found_rate = learned(
        args.length,
        "differential-q",
        {"step_size": args.step_size},
        args.episodes,
        args.runs,
    )
    rows.append(("none", "-", "-", found_loop, f"{found_rate:.2f}"))

    print()
    for line in table(
        ["discount", "exact", "per step", "q-learning", "per step"],
        rows,
        align=["right", "right", "right", "right", "right"],
    ):
        print(f"  {line}")

    print(
        "\n'exact' is the loop the optimal policy under that discount takes,\n"
        "worked out from the model, so the flip is a property of the question\n"
        "rather than something an agent failed to learn. 'per step' is what\n"
        "the chosen policy really collects, which has no discount in it. The\n"
        "last row is `differential-q`, which has no discount to get wrong."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
