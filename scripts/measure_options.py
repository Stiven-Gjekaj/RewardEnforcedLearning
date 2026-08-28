#!/usr/bin/env python3
"""What an action that lasts several steps costs, and where the cost is.

`options-q` can choose to walk to a doorway and stop there. On the four rooms
grid it has eight of those and its four primitive actions, and `hallways=off`
leaves it the four alone, which is Q-learning. Both sides are then the same
code and the difference between them is the options.

    python scripts/measure_options.py
    python scripts/measure_options.py --episodes 1000 --runs 20

The cost is measured against a ladder of exploration rates, because that is
what says where the cost is. An exploratory choice that lands on a long option
commits several steps in one direction, so it is paid for several times over,
and a cost that comes from that has to fall with epsilon. A cost that came from
the learning would not.

`while learning` is the mean return of the last hundred episodes. `long` is the
share of choices that were an option lasting more than a step, and `length` is
the mean number of steps an option ran for.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.dp import evaluate_policy, value_iteration
from rel.agents.options import OptionsQ
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

LADDER = (0.2, 0.1, 0.05, 0.02)


def measure(
    grid: str, hallways: bool, epsilon: float, episodes: int, runs: int
) -> tuple[float, float, int, float, float]:
    returns: list[float] = []
    values: list[float] = []
    longs: list[float] = []
    lengths: list[float] = []
    stuck = 0

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(grid, root.stream("env"))
        discount = env.spec.suggested_discount
        agent = AGENTS.make(
            "options-q",
            root.stream("agent"),
            env,
            hallways=hallways,
            epsilon=epsilon,
        )
        assert isinstance(agent, OptionsQ)

        record = train(env, agent, episodes, discount=discount)
        returns.append(statistics.mean(record.returns[-100:]))

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=discount)
        if report.reaches_end:
            values.append(report.start_value)
        else:
            stuck += 1

        longs.append(agent.long_chosen / max(agent.finished, 1))
        lengths.append(agent.steps_in_options / max(agent.finished, 1))

    return (
        statistics.mean(returns),
        statistics.mean(values) if values else float("nan"),
        stuck,
        statistics.mean(longs),
        statistics.mean(lengths),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="rooms")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(LADDER))
    args = parser.parse_args()

    probe = ENVIRONMENTS.make(args.grid, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount
    best = value_iteration(probe, discount=discount).start_value
    print(
        f"{args.grid}, {args.episodes} episodes, seeds 1 to {args.runs}, "
        f"discount {discount:g}.\nThe best possible return is {best:.2f}."
    )

    rows: list[tuple[str, ...]] = []
    for epsilon in args.epsilons:
        without = measure(args.grid, False, epsilon, args.episodes, args.runs)
        with_them = measure(args.grid, True, epsilon, args.episodes, args.runs)
        rows.append(
            (
                f"{epsilon:g}",
                f"{without[0]:.2f}",
                f"{with_them[0]:.2f}",
                f"{without[0] - with_them[0]:.2f}",
                f"{(without[0] - with_them[0]) / epsilon:.1f}",
                f"{with_them[3]:.0%}",
                f"{with_them[4]:.2f}",
                str(with_them[2]) if with_them[2] else "",
            )
        )

    headings = [
        "epsilon",
        "actions only",
        "with options",
        "cost",
        "cost/epsilon",
        "long",
        "length",
        "stuck",
    ]
    print()
    for line in table(headings, rows, align=["right"] * len(headings)):
        print(f"  {line}")

    print(
        "\n'cost' is what the options took off the return while learning, and\n"
        "'cost/epsilon' is that divided by the exploration rate. A column that\n"
        "stays flat says the cost is the price of exploring rather than\n"
        "anything about what was learned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
