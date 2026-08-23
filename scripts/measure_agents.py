#!/usr/bin/env python3
"""Run every tabular agent on a grid and report what each one reached.

The tables in `docs/algorithms.md` come from here. Run it with:

    python scripts/measure_agents.py
    python scripts/measure_agents.py --env windy --episodes 800

Three numbers are reported for each agent, and each answers a different
question.

`while learning` is the mean return of the last hundred episodes, which
includes the cost of exploring. It is what the chapters plot and it is not what
the agent knows.

`greedy, exactly` is the value of the greedy policy, worked out from the model
rather than sampled. It has no noise in it at all, and it says outright when a
policy never reaches the goal instead of reporting the step limit.

`stuck` counts the seeds whose greedy policy never reaches an ending. A policy
can be caught on a cell where the best action does not move it, and it then
circles for ever. That happens, it is not rare, and one averaged number would
hide it entirely.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.dp import evaluate_policy, value_iteration
from rel.cli import parse_settings
from rel.core import TabularEnv
from rel.envs import ENVIRONMENTS
from rel.metrics import summarise
from rel.rng import Rng
from rel.training import train

#: The agents this table compares. All of them keep a table, so the comparison
#: is between learning rules rather than between representations. The agents
#: over a network are measured on their own in `docs/algorithms.md`, because
#: they are far slower and far more sensitive to their settings, and averaging
#: them into this table would say nothing about either.
AGENT_NAMES = (
    "random",
    "monte-carlo",
    "sarsa",
    "expected-sarsa",
    "q-learning",
    "double-q",
    "n-step-sarsa",
    "dyna-q",
    "dyna-q-plus",
)


def measure(
    name: str,
    environment: str,
    episodes: int,
    runs: int,
    first_seed: int,
    settings: dict[str, object] | None = None,
) -> tuple[str, ...]:
    learned: list[float] = []
    exact: list[float] = []
    stuck = 0
    started = time.perf_counter()

    for run in range(runs):
        seed = first_seed + run
        root = Rng(seed)
        env = ENVIRONMENTS.make(environment, root.stream("env"))
        agent = AGENTS.make(name, root.stream("agent"), env, **(settings or {}))

        discount = env.spec.suggested_discount
        record = train(env, agent, episodes, discount=discount)
        learned.append(record.final(100))

        if isinstance(env, TabularEnv):
            policy = [agent.greedy(state) for state in range(env.observation_space.n)]
            report = evaluate_policy(env, policy, discount=discount)
            if report.reaches_end:
                exact.append(report.start_value)
            else:
                stuck += 1

    seconds = time.perf_counter() - started
    while_learning = summarise(learned)
    greedy = f"{sum(exact) / len(exact):.2f}" if exact else "-"

    return (
        name,
        f"{while_learning.mean:.2f} +/- {while_learning.error:.2f}",
        greedy,
        str(stuck) if stuck else "",
        f"{seconds:.1f}s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="cliff")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENT_NAMES),
        help="which agents to measure",
    )
    parser.add_argument(
        "--set",
        nargs="+",
        dest="agent_set",
        metavar="NAME=VALUE",
        help="a setting for every agent measured, such as entropy=0.05",
    )
    args = parser.parse_args()
    settings = parse_settings(args.agent_set)

    env = ENVIRONMENTS.make(args.env, Rng(args.seed).stream("env"))
    discount = env.spec.suggested_discount

    best = "-"
    if isinstance(env, TabularEnv):
        solved = value_iteration(env, discount=discount)
        best = f"{solved.start_value:.2f}"

    print(
        f"{args.env}, {args.episodes} episodes, seeds {args.seed} to "
        f"{args.seed + args.runs - 1}, discount {discount:g}"
    )
    print(f"the best possible return is {best}")
    if settings:
        named = ", ".join(f"{name}={value!r}" for name, value in settings.items())
        print(f"every agent is built with {named}")
    print()

    headings = ("agent", "while learning", "greedy, exactly", "stuck", "time")
    rows = [
        measure(name, args.env, args.episodes, args.runs, args.seed, settings)
        for name in args.agents
    ]

    widths = [
        max(len(heading), *(len(row[index]) for row in rows))
        for index, heading in enumerate(headings)
    ]
    print(
        "  ".join(
            head.ljust(width) for head, width in zip(headings, widths, strict=True)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            )
        )

    print()
    print(
        "'while learning' is the mean of the last hundred episodes, averaged "
        f"over {args.runs} seeds, with the standard error of that mean."
    )
    print(
        "'greedy, exactly' is the value of the greedy policy from the model, "
        "over the seeds whose policy reaches an ending."
    )
    print("'stuck' counts the seeds whose greedy policy never reaches one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
