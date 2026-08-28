#!/usr/bin/env python3
"""What ordering the replays buys, counted in updates rather than episodes.

`dyna-q` and `prioritised-sweeping` both learn from a model, and both spend
their time replaying remembered steps. Comparing them by episode hides the
whole question, because the question is how many replays it takes.

    python scripts/measure_sweeping.py
    python scripts/measure_sweeping.py --planning 20 --runs 20

An update is one application of the learning rule to one cell. `dyna-q` makes
one for the real step and `planning_steps` more, every step, forever.
`prioritised-sweeping` makes one for every entry it takes off the queue, and
when the queue is empty it makes none.

`solved` counts the seeds whose greedy policy became exactly optimal inside the
episode cap, and `updates` is the median number of updates that took, over
those seeds. `idle` is the updates per real step over the last twenty episodes,
which is what a run costs once it has nothing left to learn.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.base import Agent
from rel.agents.dp import evaluate_policy, value_iteration
from rel.agents.dyna import DynaQ
from rel.agents.sweeping import PrioritisedSweeping
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

PLANNERS = ("dyna-q", "prioritised-sweeping")

#: How many episodes at the end of a run count as settled.
TAIL = 20


def greedy_policy(agent: Agent[int], states: int) -> list[int]:
    """The agent's greedy policy, without spending its source of chance.

    `greedy` breaks a tie by drawing, and an untouched row is all ties, so
    asking for the policy after every episode changes the run being measured.
    Measured on seed 5 of the maze: probing every episode moved the point
    where prioritised sweeping first held the optimal policy from episode 68 to
    episode 258, and the updates it took from 6590 to 10964. The draws are put
    back.
    """
    before = agent.rng.snapshot()
    policy = [agent.greedy(state) for state in range(states)]
    agent.rng = Rng.restore(*before)
    return policy


def updates_of(agent: Agent[int], planning: int) -> int:
    """How many times this agent has applied the learning rule to a cell."""
    if isinstance(agent, PrioritisedSweeping):
        return agent.replays
    # One for the real step, and the quota of replays after it.
    return agent.steps * (1 + planning)


def measure(
    planner: str, grid: str, episodes: int, runs: int, planning: int
) -> tuple[str, ...]:
    probe = ENVIRONMENTS.make(grid, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount
    best = value_iteration(probe, discount=discount).start_value

    solved: list[int] = []
    idles: list[float] = []

    for seed in range(1, runs + 1):
        rng = Rng(seed)
        env = ENVIRONMENTS.make(grid, rng.stream("env"))
        builder = DynaQ if planner == "dyna-q" else PrioritisedSweeping
        agent: Agent[int] = builder(
            rng.stream("agent"),
            env.action_space,
            planning_steps=planning,
            step_size=0.5,
            discount=discount,
            epsilon=0.1,
        )

        first: int | None = None
        tail_updates = tail_steps = 0
        for episode in range(episodes):
            before_updates = updates_of(agent, planning)
            before_steps = agent.steps
            train(env, agent, 1, discount=discount)

            if episode >= episodes - TAIL:
                tail_updates += updates_of(agent, planning) - before_updates
                tail_steps += agent.steps - before_steps

            if first is None:
                policy = greedy_policy(agent, env.observation_space.n)
                report = evaluate_policy(env, policy, discount=discount)
                if report.reaches_end and report.start_value >= best - 1e-9:
                    first = updates_of(agent, planning)

        if first is not None:
            solved.append(first)
        idles.append(tail_updates / tail_steps if tail_steps else 0.0)

    return (
        planner,
        f"{len(solved)} of {runs}",
        f"{statistics.median(solved):.0f}" if solved else "-",
        f"{min(solved)}" if solved else "-",
        f"{max(solved)}" if solved else "-",
        f"{sum(idles) / len(idles):.3f}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="maze")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--planning", type=int, default=5)
    args = parser.parse_args()

    probe = ENVIRONMENTS.make(args.grid, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount
    best = value_iteration(probe, discount=discount).start_value
    print(
        f"{args.grid}, {args.episodes} episodes, seeds 1 to {args.runs}, "
        f"{args.planning} planning steps.\n"
        f"The best possible policy is worth {best:.3f}."
    )

    rows = [
        measure(planner, args.grid, args.episodes, args.runs, args.planning)
        for planner in PLANNERS
    ]
    headings = ["planner", "solved", "updates", "fewest", "most", "idle"]
    print()
    for line in table(headings, rows, align=["left"] + ["right"] * 5):
        print(f"  {line}")

    print(
        "\n'updates' is the median number of applications of the learning rule\n"
        "before the greedy policy was exactly optimal, over the seeds where it\n"
        "was. 'idle' is the updates per real step in the last twenty episodes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
