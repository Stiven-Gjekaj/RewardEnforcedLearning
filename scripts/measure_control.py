#!/usr/bin/env python3
"""Run the agents that approximate on the two control problems.

These environments have no table of states and no model, so there is no exact
value to compare against. What is reported is the return of the greedy policy
over several seeds, and every one of them, because the interesting thing about
these agents is how much they vary.

    python scripts/measure_control.py
    python scripts/measure_control.py --env cartpole --episodes 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import evaluate, train

AGENT_NAMES = ("random", "tile-sarsa", "tile-q", "reinforce", "actor-critic")


def measure(
    name: str, environment: str, episodes: int, runs: int, first_seed: int
) -> tuple[str, ...]:
    greedy: list[float] = []
    started = time.perf_counter()

    for run in range(runs):
        seed = first_seed + run
        root = Rng(seed)
        env = ENVIRONMENTS.make(environment, root.stream("env"))
        agent = AGENTS.make(name, root.stream("agent"), env)

        train(env, agent, episodes)
        watched = evaluate(
            ENVIRONMENTS.make(environment, Rng(seed + 500).stream("env")), agent, 10
        )
        greedy.append(watched.final())

    seconds = time.perf_counter() - started
    mean = sum(greedy) / len(greedy)
    each = " ".join(f"{value:.0f}" for value in greedy)
    return name, f"{mean:.1f}", each, f"{seconds:.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="mountaincar")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--agents", nargs="+", default=list(AGENT_NAMES))
    args = parser.parse_args()

    print(
        f"{args.env}, {args.episodes} episodes, seeds {args.seed} to "
        f"{args.seed + args.runs - 1}"
    )
    print()

    headings = ("agent", "greedy, mean", "each seed", "time")
    rows = [
        measure(name, args.env, args.episodes, args.runs, args.seed)
        for name in args.agents
    ]

    widths = [
        max(len(heading), *(len(row[index]) for row in rows))
        for index, heading in enumerate(headings)
    ]
    print("  ".join(h.ljust(w) for h, w in zip(headings, widths, strict=True)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))

    print()
    print(
        "Every seed is shown. These agents vary far more than the tabular ones "
        "do, and a mean on its own would hide it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
