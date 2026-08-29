#!/usr/bin/env python3
"""What four ways of exploring cost on a grid where exploring is the problem.

Epsilon-greedy explores by ignoring what the agent knows. The three rules
beside it use it, each in a different way, and the corridor is where the
difference shows: one route, forty seven steps, and nothing pays until the end
of it.

    python scripts/measure_exploration.py
    python scripts/measure_exploration.py --env maze --runs 10
    python scripts/measure_exploration.py --each-seed

The number that matters here is not the return. Reaching the goal pays 1
whether it took forty seven steps or nine hundred, so a table of returns says
only how often each rule arrived. What separates the four is when the goal was
first reached and how many steps went into reaching it.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.dp import DidNotSettleError, value_iteration
from rel.core import TabularEnv
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import Record, train
from rel.ui.table import table

#: The four arms. Each is a label, an `explore` setting and an `optimism`.
#:
#: Optimistic initialisation is not a rule, so it cannot be one of the three in
#: `rel/agents/explore.py`. It is a starting value: every action of a state
#: nobody has visited begins worth the best outcome there is, so a greedy agent
#: walks towards whatever it has not tried and stops as soon as the numbers
#: come down to the truth. It belongs in this table because it answers the same
#: question and answers it in the other place.
ARMS: tuple[tuple[str, str, float], ...] = (
    ("epsilon-greedy", "epsilon-greedy", 0.0),
    ("softmax", "softmax:0.02", 0.0),
    ("count-bonus", "count-bonus:0.5", 0.0),
    ("optimistic", "epsilon-greedy:0.0", 1.0),
)


def first_arrival(record: Record) -> tuple[int | None, int]:
    """The episode that first reached an ending, and the steps spent before it.

    An ending rather than a positive return. The first version of this asked
    whether the return was above zero, which is right on a grid where only the
    goal pays and says "never" on every run of the cliff walk, where every
    return is negative. An episode that ended inside the rules reached the
    goal on every grid here except the lake, where it may have fallen in.

    `None` for a run that never arrived, rather than the episode count, which
    would read as a run that arrived on the last episode.
    """
    spent = 0
    paired = zip(record.terminated, record.lengths, strict=True)
    for index, (ended, length) in enumerate(paired):
        if ended:
            return index + 1, spent
        spent += length
    return None, spent


def measure(
    grid: str,
    agent_name: str,
    episodes: int,
    runs: int,
    explore: str,
    optimism: float,
    settings: dict[str, object],
) -> tuple[list[int | None], list[int], list[float], list[float | None], float]:
    """The four numbers of every seed, and the seconds the whole row took."""
    arrivals: list[int | None] = []
    costs: list[int] = []
    settled: list[float] = []
    exact: list[float | None] = []
    started = time.perf_counter()

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(grid, root.stream("env"))
        discount = env.spec.suggested_discount
        agent = AGENTS.make(
            agent_name,
            root.stream("agent"),
            env,
            discount=discount,
            explore=explore,
            optimism=optimism,
            **settings,
        )
        record = train(env, agent, episodes, discount=discount)

        arrived, spent = first_arrival(record)
        arrivals.append(arrived)
        costs.append(spent)
        settled.append(statistics.mean(record.lengths[-50:]))
        exact.append(learned_value(env, agent, discount))

    return arrivals, costs, settled, exact, time.perf_counter() - started


def learned_value(env: object, agent: object, discount: float) -> float | None:
    """What the greedy policy is worth, from the model, or `None` if it loops.

    Read once at the end of the run rather than every episode. Reading it costs
    the agent a draw wherever the greedy action ties, and a probe that spends
    the agent's own source of chance changes the run it is measuring.
    """
    from rel.agents.base import Agent
    from rel.agents.dp import evaluate_policy

    if not isinstance(env, TabularEnv) or not isinstance(agent, Agent):
        return None

    policy = [agent.greedy(state) for state in range(env.observation_space.n)]
    try:
        learned = evaluate_policy(env, policy, discount=discount)
    except DidNotSettleError:
        return None
    return learned.start_value if learned.reaches_end else None


def _middle(numbers: list[int | None]) -> str:
    """The median of the runs that arrived, and how many did not."""
    arrived = [one for one in numbers if one is not None]
    if not arrived:
        return "never"
    text = f"{statistics.median(arrived):.0f}"
    missing = len(numbers) - len(arrived)
    return text if not missing else f"{text} ({missing} never)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="corridor")
    parser.add_argument("--agent", default="q-learning")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--set",
        action="append",
        metavar="NAME=VALUE",
        help="a setting held fixed across all four rows, repeatable",
    )
    parser.add_argument(
        "--each-seed",
        action="store_true",
        help="print the episode every seed first arrived on, under the table",
    )
    args = parser.parse_args()

    from rel.cli import parse_settings

    settings = parse_settings(args.set)

    probe = ENVIRONMENTS.make(args.env, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount
    shortest = "-"
    if isinstance(probe, TabularEnv):
        try:
            best = value_iteration(probe, discount=discount)
            shortest = f"{best.start_value:.4f}"
        except DidNotSettleError:
            shortest = "-"

    print(
        f"{args.agent} on {args.env}, {args.episodes} episodes, "
        f"seeds 1 to {args.runs}, discount {discount:g}.\n"
        f"The best possible return from the start is {shortest}."
    )
    if settings:
        print("Held fixed: " + ", ".join(f"{k}={v}" for k, v in settings.items()))

    rows = []
    every: list[tuple[str, list[int | None]]] = []
    for label, explore, optimism in ARMS:
        arrivals, costs, settled, exact, seconds = measure(
            args.env,
            args.agent,
            args.episodes,
            args.runs,
            explore,
            optimism,
            settings,
        )
        every.append((label, arrivals))
        reached = [one for one in exact if one is not None]
        rows.append(
            (
                label,
                _middle(arrivals),
                f"{statistics.median(costs):.0f}",
                f"{statistics.median(settled):.0f}",
                "-" if not reached else f"{statistics.median(reached):.4f}",
                str(len(exact) - len(reached)),
                f"{seconds:.0f}s",
            )
        )

    print()
    for line in table(
        ["rule", "first end", "steps to it", "settled", "policy", "stuck", "time"],
        rows,
        align=["left"] + ["right"] * 6,
    ):
        print(f"  {line}")

    if args.each_seed:
        print()
        for label, arrivals in every:
            numbers = " ".join(
                "  -" if one is None else f"{one:3d}" for one in arrivals
            )
            print(f"  {label:15s} {numbers}")

    print(
        "\n'first end' is the episode that first reached an ending and 'steps\n"
        "to it' is how many steps went into reaching it, both the median over\n"
        "the seeds. 'settled' is the mean episode length over the last fifty,\n"
        "which counts the exploring the agent is still doing. 'policy' is what\n"
        "the greedy policy is worth from the model, over the seeds whose\n"
        "policy reaches an ending, and 'stuck' counts the ones that do not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
