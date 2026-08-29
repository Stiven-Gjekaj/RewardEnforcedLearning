#!/usr/bin/env python3
"""What planning at decision time buys, and what it costs to get it.

`mcts` runs simulations from the state it is standing in and acts on what they
said. `dyna-q` and `prioritised-sweeping` spend the same kind of work in the
background, on a table, before the question is asked. This puts the three on
one axis.

    python scripts/measure_search.py
    python scripts/measure_search.py --env corridor --episodes 40
    python scripts/measure_search.py --runs 5

The axis is model steps. Every one of these agents works by asking a model what
would happen, and the number of times it asks is what each of them is spending.
A table of returns against wall clock would compare the speed of the code, and
a table of returns against episodes would let an agent that asks a thousand
times a step look free.

`mcts` here is given the environment's own model. `dyna-q` learns one. That is
not a fair fight and it is the interesting direction: the agent that was handed
the answer is the one to beat.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.dp import DidNotSettleError, evaluate_policy, value_iteration
from rel.core import Env, TabularEnv
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

#: Each row: a label, the agent, and the settings that make it that row.
ARMS: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("mcts 10", "mcts", {"simulations": 10}),
    ("mcts 25", "mcts", {"simulations": 25}),
    ("mcts 50", "mcts", {"simulations": 50}),
    ("mcts 50, no reuse", "mcts", {"simulations": 50, "reuse": False}),
    ("dyna-q 5", "dyna-q", {"planning_steps": 5}),
    ("dyna-q 20", "dyna-q", {"planning_steps": 20}),
    ("prioritised-sweeping", "prioritised-sweeping", {}),
)


def model_steps(agent: object) -> int:
    """How many times this agent asked a model what would happen.

    `mcts` counts simulated steps and the two planners count replays. They are
    the same thing spent in different places, which is the whole point of
    putting them in one column.

    Two names and no more. An agent that counts something else calls it
    something else, and a search over every integer attribute would sooner or
    later find one and report it as though it belonged here.
    """
    for name in ("simulated", "replays"):
        found = getattr(agent, name, None)
        if isinstance(found, int):
            return found
    return 0


def exact(env: Env[object], agent: object, discount: float) -> float | None:
    """What the greedy policy is worth from the model, or `None` if it loops."""
    from rel.agents.base import Agent

    if not isinstance(env, TabularEnv) or not isinstance(agent, Agent):
        return None
    policy = [agent.greedy(state) for state in range(env.observation_space.n)]
    try:
        learned = evaluate_policy(env, policy, discount=discount)
    except DidNotSettleError:
        return None
    return learned.start_value if learned.reaches_end else None


def measure(
    grid: str,
    agent_name: str,
    settings: dict[str, object],
    episodes: int,
    runs: int,
) -> tuple[list[float], list[int], list[float | None], float]:
    lengths: list[float] = []
    asked: list[int] = []
    values: list[float | None] = []
    started = time.perf_counter()

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(grid, root.stream("env"))
        discount = env.spec.suggested_discount
        agent = AGENTS.make(
            agent_name, root.stream("agent"), env, discount=discount, **settings
        )
        record = train(env, agent, episodes, discount=discount)

        lengths.append(statistics.mean(record.lengths[-10:]))
        asked.append(model_steps(agent))
        values.append(exact(env, agent, discount))

    return lengths, asked, values, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="maze")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--only",
        help=(
            "run only these rows, comma separated, by the label in the first "
            "column, for example 'mcts 50,dyna-q 5'"
        ),
    )
    args = parser.parse_args()

    arms = ARMS
    if args.only:
        wanted = [one.strip() for one in args.only.split(",")]
        unknown = [one for one in wanted if one not in {label for label, _, _ in ARMS}]
        if unknown:
            offered = ", ".join(repr(label) for label, _, _ in ARMS)
            raise SystemExit(f"There is no row {unknown[0]!r}. There is: {offered}.")
        arms = tuple(arm for arm in ARMS if arm[0] in wanted)

    probe = ENVIRONMENTS.make(args.env, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount
    best = "-"
    shortest = "-"
    if isinstance(probe, TabularEnv):
        try:
            solved = value_iteration(probe, discount=discount)
            best = f"{solved.start_value:.4f}"
            shortest = str(_steps_to_end(probe, list(solved.policy)))
        except DidNotSettleError:
            best = "-"

    print(
        f"{args.env}, {args.episodes} episodes, seeds 1 to {args.runs}, "
        f"discount {discount:g}.\nThe best possible return is {best}, "
        f"and the shortest route is {shortest} steps."
    )

    rows = []
    for label, name, settings in arms:
        lengths, asked, values, seconds = measure(
            args.env, name, settings, args.episodes, args.runs
        )
        reached = [one for one in values if one is not None]
        rows.append(
            (
                label,
                f"{statistics.median(lengths):.0f}",
                "-" if not reached else f"{statistics.median(reached):.4f}",
                str(len(values) - len(reached)),
                f"{statistics.median(asked):,.0f}",
                f"{seconds:.0f}s",
            )
        )

    print()
    for line in table(
        ["agent", "settled", "policy", "stuck", "model steps", "time"],
        rows,
        align=["left"] + ["right"] * 5,
    ):
        print(f"  {line}")

    print(
        "\n'settled' is the mean episode length over the last ten, the median\n"
        "over the seeds. 'policy' is what the greedy policy is worth from the\n"
        "model, and 'model steps' is how many times the agent asked a model\n"
        "what would happen, which is the work all three of these are spending."
    )
    return 0


def _steps_to_end(env: TabularEnv, policy: list[int]) -> int:
    """How many steps the best policy takes from the start, by walking it."""
    state = env.start_states()[0][1]
    for step in range(1, 10_000):
        branch = env.transitions(state, policy[state])[0]
        if branch.terminated:
            return step
        state = branch.observation
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
