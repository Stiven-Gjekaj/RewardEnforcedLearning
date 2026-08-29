#!/usr/bin/env python3
"""What crediting every state an option passed through buys, and what it costs.

`options-q` waits for an option to stop and credits the state it started in.
Three steps inside one option move one cell. `intra-option-q` credits every
option that agreed with each step, so the states passed through learn about the
option as well.

    python scripts/measure_intra_option.py
    python scripts/measure_intra_option.py --episodes 800 --runs 20

The question is whether that recovers what the options cost. The algorithms
page measures that cost at 2.57 return on four rooms, and a ladder of
exploration rates says the cost is the price of exploring rather than anything
about what was learned. If that reading is right, fixing the credit assignment
will move the early episodes and leave the late ones alone. That is a
prediction, and this is the test of it.

The last two rows run with `hallways=off`, which leaves both agents their
primitive actions. A primitive option stops after every step, so both are
Q-learning there and both have to give the same answer.
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

AGENT_NAMES = ("options-q", "intra-option-q")


def measure(
    name: str, grid: str, episodes: int, runs: int, block: int, hallways: bool
) -> tuple[list[float], float, float, int]:
    """The curve, the updates per step, the last hundred, and the stuck count."""
    blocks: list[list[float]] = [[] for _ in range(episodes // block)]
    per_step: list[float] = []
    finals: list[float] = []
    stuck = 0

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(grid, root.stream("env"))
        discount = env.spec.suggested_discount
        agent = AGENTS.make(name, root.stream("agent"), env, hallways=hallways)
        assert isinstance(agent, OptionsQ)

        record = train(env, agent, episodes, discount=discount)
        for index in range(len(blocks)):
            blocks[index].append(
                statistics.mean(record.lengths[index * block : (index + 1) * block])
            )

        per_step.append(agent.updates / max(agent.steps, 1))
        finals.append(statistics.mean(record.returns[-100:]))

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        if not evaluate_policy(env, policy, discount=discount).reaches_end:
            stuck += 1

    return (
        [statistics.mean(one) for one in blocks],
        statistics.mean(per_step),
        statistics.mean(finals),
        stuck,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="rooms")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--block", type=int, default=50)
    args = parser.parse_args()

    probe = ENVIRONMENTS.make(args.grid, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount
    best = value_iteration(probe, discount=discount).start_value
    print(
        f"{args.grid}, {args.episodes} episodes, seeds 1 to {args.runs}, "
        f"discount {discount:g}.\nThe best possible return is {best:.2f}, "
        f"and the shortest path is {-best:.0f} steps."
    )

    rows: list[tuple[str, ...]] = []
    for hallways in (True, False):
        for name in AGENT_NAMES:
            curve, per_step, final, stuck = measure(
                name, args.grid, args.episodes, args.runs, args.block, hallways
            )
            rows.append(
                (
                    name if hallways else f"{name}, no options",
                    *[f"{point:.1f}" for point in curve],
                    f"{per_step:.2f}",
                    f"{final:.2f}",
                    str(stuck) if stuck else "",
                )
            )

    headings = ["agent"]
    headings += [
        f"{index * args.block + 1}-{(index + 1) * args.block}"
        for index in range(args.episodes // args.block)
    ]
    headings += ["updates/step", "last 100", "stuck"]

    print()
    for line in table(headings, rows, align=["left"] + ["right"] * (len(headings) - 1)):
        print(f"  {line}")

    print(
        "\nThe numbered columns are the mean episode length over that block of\n"
        "episodes. 'updates/step' is how many times the learning rule was\n"
        "applied to a cell for each step taken. The last two rows hold only\n"
        "primitive options, where both agents are Q-learning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
