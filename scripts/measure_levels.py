#!/usr/bin/env python3
"""What it costs to cut a box of actions into a short list of them.

The pendulum takes a torque anywhere between minus two and two. Every agent
here but one takes an action from a list, so the usual answer is to cut the box
into levels, and the question this answers is how many levels are worth having.

    python scripts/measure_levels.py
    python scripts/measure_levels.py --runs 8
    python scripts/measure_levels.py --levels 2 4 8 --episodes 200 600

Two things are measured.

**The ladder.** The same agent over the same problem at several counts of
levels, at two budgets. The two columns disagree about which count is best,
and that disagreement is the answer.

**The two kinds of agent.** What learns this problem at all: the agents that
keep a value and rank it, against the ones that learn a policy directly, with
the cut version and the box side by side.

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
from rel.core import Env
from rel.envs.levels import levelled_pendulum
from rel.envs.pendulum import Pendulum
from rel.rng import Rng
from rel.training import evaluate, train
from rel.ui.table import table

DISCOUNT = 0.99
#: How many episodes of learning, and then how many watched greedily.
EPISODES = (300, 900)
WATCHED = 10

#: The counts of levels the ladder walks. Two is a switch.
LEVELS: tuple[int, ...] = (2, 3, 5, 9, 17)

#: Which agent walks the ladder. It has to be one that learns this problem, or
#: the ladder measures the agent rather than the cut.
LADDER_AGENT = "tile-q"

#: The agents on the cut version, and the one on the box itself.
ON_A_LIST: tuple[str, ...] = (
    "random",
    "tile-q",
    "tile-sarsa",
    "deep-q",
    "actor-critic",
)
ON_A_BOX = "gaussian-actor-critic"

#: The levels the second table uses. Nine is the middle of the ladder and the
#: registry default.
MIDDLE = 9


def one_run(
    agent_name: str, env: Env[tuple[float, ...], object], seed: int, episodes: int
) -> float:
    """The mean return of a greedy run after learning."""
    agent = AGENTS.make(agent_name, Rng(seed).stream("agent"), env)
    train(env, agent, episodes, discount=DISCOUNT)
    watched = evaluate(env, agent, WATCHED, discount=DISCOUNT)
    return statistics.mean(watched.returns)


def on_levels(agent_name: str, levels: int, seed: int, episodes: int) -> float:
    env = levelled_pendulum(Rng(seed).stream("env"), levels=levels)
    return one_run(agent_name, env, seed, episodes)


def on_the_box(seed: int, episodes: int) -> float:
    env = Pendulum(Rng(seed).stream("env"))
    return one_run(ON_A_BOX, env, seed, episodes)


def ladder_section(
    levels: tuple[int, ...], runs: int, budgets: tuple[int, ...]
) -> None:
    print(
        f"The same agent, the same problem, the torque cut into different\n"
        f"numbers of levels. {LADDER_AGENT}, {runs} seeds, the mean of "
        f"{WATCHED} greedy episodes.\n"
    )

    got: dict[tuple[int, int], float] = {}
    for count in levels:
        for episodes in budgets:
            seeds = [
                on_levels(LADDER_AGENT, count, seed, episodes)
                for seed in range(1, runs + 1)
            ]
            got[count, episodes] = statistics.mean(seeds)

    for line in table(
        ["levels", *(f"{episodes} episodes" for episodes in budgets)],
        [
            [str(count), *(f"{got[count, episodes]:.1f}" for episodes in budgets)]
            for count in levels
        ],
        align=["right", *("right" for _ in budgets)],
    ):
        print(f"  {line}")

    # Which row won each column, worked out rather than written down. The
    # first reading of this table was from one budget, and the sentence under
    # it said that a finer cut is never better. The second budget disagreed.
    print()
    for episodes in budgets:
        best = max(levels, key=lambda count: got[count, episodes])
        print(
            f"  Best at {episodes} episodes: {best} levels, {got[best, episodes]:.1f}"
        )


def agents_section(runs: int, episodes: int, levels: int) -> None:
    print(
        f"\nWhat learns this problem at all. {runs} seeds, {episodes} episodes,\n"
        f"the mean of {WATCHED} greedy episodes. The last row acts on the box\n"
        f"itself and every other row acts on it cut into {levels} levels.\n"
    )

    rows: list[list[str]] = []
    for name in ON_A_LIST:
        got = [on_levels(name, levels, seed, episodes) for seed in range(1, runs + 1)]
        rows.append([name, f"{levels}", f"{statistics.mean(got):.1f}"])

    got = [on_the_box(seed, episodes) for seed in range(1, runs + 1)]
    rows.append([ON_A_BOX, "the box", f"{statistics.mean(got):.1f}"])

    for line in table(
        ["agent", "actions", "mean return"],
        rows,
        align=["left", "right", "right"],
    ):
        print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="seeds per setting")
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--episodes", type=int, nargs="+", default=list(EPISODES))
    parser.add_argument("--middle", type=int, default=MIDDLE)
    args = parser.parse_args()

    started = time.perf_counter()
    budgets = tuple(args.episodes)
    ladder_section(tuple(args.levels), args.runs, budgets)
    agents_section(args.runs, budgets[0], args.middle)

    print(
        "\nA coarse cut learns faster and a fine one learns better. Every extra "
        "level\nis another column to learn, so a small budget is spent better on "
        "fewer of\nthem, and a switch is enough to swing this problem up at all. "
        "Which count is\nright is a question about the budget rather than about "
        "the environment."
    )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
