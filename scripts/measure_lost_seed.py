#!/usr/bin/env python3
"""The cliff walk seed where REINFORCE never sees the goal, and what it needs.

A policy gradient learns by pushing up whatever led to a good return. On the
cliff walk nothing leads to a good return until the goal is reached once, so a
seed whose early policy never wanders that far has nothing to learn from. No
rule for sharing out a return can help: there is no return to share.

The milestones left three possibilities open, and this measures them: does that
seed want more entropy, more episodes, or something that is not a knob at all.

    python scripts/measure_lost_seed.py
    python scripts/measure_lost_seed.py --runs 12 --budgets 500 1000 2000

This one is slow. A policy gradient is seven hundred times slower than
Q-learning on the same grid, and the whole point here is a budget long enough
to see what a short one hides, so there is no fast version of the question.

The first table is which seeds have not reached the goal by each budget. The
second takes the seed that holds out longest and varies the entropy bonus,
which is the knob that holds the policy wide enough to wander.

`--ladder-all` runs that ladder over every seed instead of the holdout, which
is what says whether the default should move rather than whether one seed can
be rescued. It is slow, which is why it is not the default.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table


def run(
    seed: int, episodes: int, entropy: float | None
) -> tuple[int | None, int, float]:
    """When this seed first reached the goal, how often, and what it ended at."""
    root = Rng(seed)
    env = ENVIRONMENTS.make("cliff", root.stream("env"))
    settings = {} if entropy is None else {"entropy": entropy}
    agent = AGENTS.make("reinforce", root.stream("agent"), env, **settings)

    record = train(env, agent, episodes, discount=1.0)
    first = next(
        (number for number, ended in enumerate(record.terminated) if ended), None
    )
    return (
        None if first is None else first + 1,
        sum(record.terminated),
        record.final(100),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--budgets", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument(
        "--entropies", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4]
    )
    parser.add_argument(
        "--ladder-episodes",
        type=int,
        default=1000,
        help="how long each run of the entropy ladder gets",
    )
    parser.add_argument(
        "--ladder-all",
        action="store_true",
        help="run the entropy ladder over every seed rather than the holdout",
    )
    args = parser.parse_args()

    longest = max(args.budgets)
    print(
        f"reinforce on cliff, seeds 1 to {args.runs}, at the default entropy.\n"
        f"Each seed is run for {longest} episodes once and read at every budget."
    )

    first_goal: dict[int, int | None] = {}
    for seed in range(1, args.runs + 1):
        first_goal[seed], _, _ = run(seed, longest, None)

    rows = []
    for budget in args.budgets:
        late = [
            str(seed)
            for seed, first in first_goal.items()
            if first is None or first > budget
        ]
        rows.append(
            (
                str(budget),
                str(len(late)),
                ", ".join(late) if late else "",
            )
        )
    print()
    for line in table(
        ["episodes", "not there yet", "which seeds"],
        rows,
        align=["right", "right", "left"],
    ):
        print(f"  {line}")

    holdout = max(
        first_goal,
        key=lambda seed: (
            first_goal[seed] is None,
            first_goal[seed] or 0,
        ),
    )
    seeds = list(range(1, args.runs + 1)) if args.ladder_all else [holdout]
    print(
        f"\n{'Every seed' if args.ladder_all else f'Seed {holdout} holds out longest'}"
        f", against the entropy bonus:"
    )

    ladder = []
    for entropy in args.entropies:
        results = [run(seed, args.ladder_episodes, entropy) for seed in seeds]
        arrived = [first for first, _, _ in results if first is not None]
        lost = len(results) - len(arrived)
        finals = [final for _, _, final in results]
        ladder.append(
            (
                f"{entropy:g}",
                f"{statistics.median(arrived):.0f}" if arrived else "never",
                str(lost) if lost else "",
                f"{statistics.mean(finals):.1f}",
            )
        )
    print()
    for line in table(
        ["entropy", "first goal", "never got there", "last 100"],
        ladder,
        align=["right", "right", "right", "right"],
    ):
        print(f"  {line}")

    print(
        f"\n'first goal' is the median episode the goal was first reached in, "
        f"over\n{args.ladder_episodes} episodes, and 'last 100' is the mean "
        f"return of the last hundred.\nA seed that never got there is left "
        f"out of the first and counted in the second."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
