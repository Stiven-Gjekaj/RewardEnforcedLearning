#!/usr/bin/env python3
"""What a staircase can say about a straight line, and what an agent reaches.

State aggregation is the smallest approximation there is: put the states into
groups and keep one number for each group. On a walk of a thousand cells that
is ten numbers where a table has a thousand, and every state in a group shares
all of its learning with the others.

    python scripts/measure_aggregation.py
    python scripts/measure_aggregation.py --runs 10 --episodes 2000
    python scripts/measure_aggregation.py --groups 4 10 40

The value function it can write down is a staircase, so it cannot be right
everywhere, and how wrong it has to be is arithmetic rather than a run. Two
things are measured against each other.

**The floor.** The best a staircase of `n` steps can do, which is the group
mean of the true values. Nothing that learns can beat it and nothing about
learning appears in it.

**What an agent reaches.** `linear-td` over the same grouping, from the same
walk, measured the same way. The gap between the two is what the learning has
left to give, and the floor is what no amount of learning can.

The true values come from `rel.agents.dp.evaluate_shares`, which sweeps the
model under the policy the walk is run with. On the small walk that sweep is
equal to the closed form, which is what says it is the arithmetic.

`docs/algorithms.md` has the tables.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.dp import evaluate_shares, uniform_shares
from rel.agents.linear_prediction import LinearPredictor
from rel.envs.classic import RandomWalk, long_walk
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

#: How many groups the ladder walks. One is a single number for the whole
#: walk and a thousand is a table with extra steps.
GROUPS: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)

EPISODES = 1000
STEP_SIZE = 0.2
DISCOUNT = 1.0


@cache
def true_values() -> tuple[float, ...]:
    """What each cell of the long walk is worth under the policy it is run at.

    Cached, because the sweep is over a thousand states with a hundred
    branches each and every row of both tables wants the same answer.
    """
    walk = long_walk(Rng(1).stream("env"))
    return evaluate_shares(walk, uniform_shares(walk), discount=DISCOUNT)


def cells() -> range:
    """The states an agent can be in, which leaves out the two endings."""
    return range(1, len(true_values()) - 1)


def group_of(state: int, groups: int, states: int) -> int:
    """The same split `rel.agents.lookup.aggregated` makes."""
    return state * groups // states


def floor_for(groups: int) -> float:
    """The best a staircase of this many steps can do, worked out.

    Each step of the staircase is one number for every state in its group, and
    the number that makes the squared error smallest is their mean. So this is
    arithmetic over the true values and nothing about learning is in it.
    """
    values = true_values()
    states = len(values)

    totals: dict[int, list[float]] = {}
    for state in cells():
        totals.setdefault(group_of(state, groups, states), []).append(values[state])

    best = {group: sum(worth) / len(worth) for group, worth in totals.items()}
    squares = sum(
        (values[state] - best[group_of(state, groups, states)]) ** 2
        for state in cells()
    )
    return float((squares / len(cells())) ** 0.5)


def one_run(groups: int, seed: int, episodes: int, step_size: float) -> float:
    """The error of what `linear-td` learned, against the true values."""
    walk = long_walk(Rng(seed).stream("env"))
    agent = AGENTS.make(
        "linear-td",
        Rng(seed).stream("agent"),
        walk,
        groups=groups,
        step_size=step_size,
        discount=DISCOUNT,
        start_value=0.0,
    )
    assert isinstance(agent, LinearPredictor)

    train(walk, agent, episodes, discount=DISCOUNT)
    values = true_values()
    return agent.error_against({state: values[state] for state in cells()})


def ladder_section(
    groups: tuple[int, ...], runs: int, episodes: int, step: float
) -> None:
    print(
        f"A thousand cells, grouped. The floor is arithmetic over the true "
        f"values\nand the agent is linear-td, {runs} seeds, {episodes} "
        f"episodes, step size {step:g}.\n"
    )

    got: dict[int, float] = {}
    for count in groups:
        reached = [one_run(count, seed, episodes, step) for seed in range(1, runs + 1)]
        got[count] = statistics.mean(reached)

    for line in table(
        ["groups", "best a staircase can do", "what linear-td reached"],
        [
            [str(count), f"{floor_for(count):.4f}", f"{got[count]:.4f}"]
            for count in groups
        ],
        align=["right", "right", "right"],
    ):
        print(f"  {line}")

    # Which row the agent did best on, worked out rather than written down.
    # The floor falls all the way and what an agent reaches does not, so the
    # two columns have their best in different places and only one of them is
    # a fact about the approximation.
    best = min(groups, key=lambda count: got[count])
    print(
        f"\n  Lowest floor: {max(groups)} groups, {floor_for(max(groups)):.4f}"
        f"\n  Lowest reached: {best} groups, {got[best]:.4f}"
    )


def closed_form_section() -> None:
    print(
        "\nThe true values come from a sweep of the model rather than a "
        "formula.\nOn the small walk the same sweep is equal to the formula, "
        "which is what\nsays it is the arithmetic:\n"
    )
    walk = RandomWalk(Rng(1).stream("env"), size=5)
    swept = evaluate_shares(walk, uniform_shares(walk))
    rows = [
        [str(state), f"{walk.true_values()[state]:.6f}", f"{swept[state]:.6f}"]
        for state in range(1, 6)
    ]
    for line in table(
        ["cell", "closed form", "swept"], rows, align=["right", "right", "right"]
    ):
        print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="seeds per setting")
    parser.add_argument("--groups", type=int, nargs="+", default=list(GROUPS))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--step-size", type=float, default=STEP_SIZE)
    args = parser.parse_args()

    started = time.perf_counter()
    ladder_section(tuple(args.groups), args.runs, args.episodes, args.step_size)
    closed_form_section()

    print(
        "\nMore groups is a lower floor and a slower walk down to it. The "
        "floor falls\nall the way, because a staircase of more steps can "
        "always say more. What an\nagent reaches turns back up, because the "
        "same episodes are spread over more\nnumbers to learn, and the best "
        "count is wherever those two meet."
    )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
