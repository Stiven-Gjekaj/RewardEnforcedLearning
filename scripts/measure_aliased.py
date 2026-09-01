#!/usr/bin/env python3
"""What an agent that ranks its actions can reach when the ranking is not the
answer, and what it looks like it reached.

    python scripts/measure_aliased.py
    python scripts/measure_aliased.py --episodes 1200 --runs 20

`aliased` is Sutton and Barto's example 13.1. Three cells that give the same
observation, the middle one with its actions reversed, and every step pays -1.
Both fixed choices never finish: always right bounces between the first two
cells and always left walks into the wall.

So the answer is a probability rather than a choice, and it is exact. The best
policy of this shape goes right `2 - sqrt 2` of the time and reaches the goal
in `6 + 4 sqrt 2` steps, which is 11.66. An agent that ranks its two actions
and explores with probability `epsilon` takes its favourite `1 - epsilon / 2`
of the time, and the same arithmetic says what that costs. At an epsilon of 0.1
it is 44.2 steps.

## The four numbers each agent gets, and why one of them is a mirage

**Last 50 while learning** is what a run of this project normally reports.

**Its own policy** freezes the agent and runs the policy it acts with, which
for an epsilon-greedy agent is epsilon-greedy over the values it ended with and
for a policy gradient agent is a draw from its softmax.

**Frozen greedy** takes its favourite action every time, with no exploring at
all. This is the number `rel train` prints as the return of the greedy policy.

**Share of right** is the probability its policy puts on going right, against
the `2 - sqrt 2` that is best.

The gap between the first two columns is the whole point of the script. An
agent whose action values have not settled acts as a mixture, because the two
values keep crossing, and the mixture is not a policy anybody chose and is not
what the agent has learned. Freeze it and read the same agent again.

`docs/algorithms.md` has the table.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.envs import ENVIRONMENTS
from rel.envs.aliased import ANYWHERE, RIGHT, AliasedCorridor
from rel.rng import Rng
from rel.training import run_episode, train
from rel.ui.table import table

#: The agents. The first three rank their two actions and the last two hold a
#: probability, which is the difference the whole environment is about.
AGENT_NAMES: tuple[str, ...] = (
    "q-learning",
    "sarsa",
    "expected-sarsa",
    "reinforce",
    "actor-critic",
)

ENVIRONMENT = "aliased"
EPISODES = 600
RUNS = 10
EPSILON = 0.1

#: How many episodes each frozen policy is run for. A run of this corridor is
#: about a dozen steps at best and forty at worst, and the spread of a
#: geometric-looking length is wide, so this is not small.
WATCHED = 200

#: How many episodes the frozen greedy policy is run for. It either finishes
#: or it does not, and one episode says which, so three is generous.
WATCHED_GREEDY = 3

#: The shares the closed form is printed at, to show the shape of it.
SHARES: tuple[float, ...] = (0.05, 0.2, 0.4, 0.5858, 0.75, 0.95)


def exploring(name: str, epsilon: float) -> dict[str, float]:
    """The epsilon setting, for the agents that have one.

    A policy gradient agent has no epsilon, and that is not an oversight in
    the registry. It does not explore by ranking its actions and then taking a
    different one now and then. It holds a probability, and exploring is what
    the probability already is. So passing an epsilon to one is a mistake and
    the registry says so, and this asks rather than being told.
    """
    if "epsilon" in AGENTS[name].options(AGENTS.fixed):
        return {"epsilon": epsilon}
    return {}


def share_of_right(agent: object) -> float:
    """How much of its policy the agent puts on going right.

    A policy gradient agent holds that number and hands it over. An agent that
    ranks its actions does not have one: its policy is its favourite action
    plus exploring, so the share is one or nothing before the exploring is
    added, and this returns that.
    """
    holds = getattr(agent, "probabilities", None)
    if holds is not None:
        return float(holds(ANYWHERE)[RIGHT])

    row = list(agent.action_values(ANYWHERE))  # type: ignore[attr-defined]
    return 1.0 if row[RIGHT] > row[1 - RIGHT] else 0.0


def watched(env: object, agent: object, episodes: int, greedy: bool) -> float:
    """Mean episode length with the learning switched off."""
    return statistics.mean(
        run_episode(
            env,  # type: ignore[arg-type]
            agent,  # type: ignore[arg-type]
            learn=False,
            greedy=greedy,
            number=number,
            discount=1.0,
        ).length
        for number in range(episodes)
    )


def one_agent(
    name: str, runs: int, episodes: int, env_name: str, epsilon: float
) -> tuple[float, float, float, float]:
    """The four numbers, averaged over the seeds."""
    learning: list[float] = []
    its_own: list[float] = []
    frozen: list[float] = []
    shares: list[float] = []

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(env_name, root.stream("env"))
        agent = AGENTS.make(name, root.stream("agent"), env, **exploring(name, epsilon))
        record = train(env, agent, episodes, discount=1.0)

        learning.append(statistics.mean(record.returns[-50:]))
        its_own.append(watched(env, agent, WATCHED, greedy=False))
        frozen.append(watched(env, agent, WATCHED_GREEDY, greedy=True))
        shares.append(share_of_right(agent))

    return (
        statistics.mean(learning),
        statistics.mean(its_own),
        statistics.mean(frozen),
        statistics.mean(shares),
    )


def closed_form_section(shares: tuple[float, ...], epsilon: float) -> None:
    print(
        "What a policy of this shape costs, from the arithmetic alone.\n"
        "It runs off to infinity at both ends, which is the two fixed"
        " choices never finishing.\n"
    )

    rows = [
        [f"{share:g}", f"{AliasedCorridor.steps_from_start(share):.2f}"]
        for share in shares
    ]
    for line in table(
        ["share of right", "steps to the goal"], rows, align=["right", "right"]
    ):
        print(f"  {line}")

    best = AliasedCorridor.best_share()
    ranking = AliasedCorridor.best_ranking_share(epsilon)
    print(
        f"\n  best share       {best:.4f}, which is 2 - sqrt 2, at"
        f" {AliasedCorridor.best_steps():.2f} steps\n"
        f"  best a ranking   {ranking:.4f}, which is 1 - epsilon / 2, at"
        f" {AliasedCorridor.steps_from_start(ranking):.2f} steps"
    )


def agents_section(
    names: tuple[str, ...],
    env_name: str,
    runs: int,
    episodes: int,
    epsilon: float,
) -> None:
    print(
        f"\n\nWhat each agent reached. {env_name}, {runs} seeds, {episodes}"
        f" episodes, epsilon {epsilon:g}.\nThe last three columns have the"
        f" learning switched off.\n"
    )

    rows = []
    for name in names:
        learning, its_own, frozen, share = one_agent(
            name, runs, episodes, env_name, epsilon
        )
        rows.append(
            [
                name,
                f"{learning:.1f}",
                f"{its_own:.1f}",
                f"{frozen:.0f}",
                f"{share:.3f}",
            ]
        )

    for line in table(
        [
            "agent",
            "last 50 while learning",
            "its own policy, steps",
            "frozen greedy, steps",
            "share of right",
        ],
        rows,
        align=["left"] + ["right"] * 4,
    ):
        print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=ENVIRONMENT)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--runs", type=int, default=RUNS, help="seeds per agent")
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--agents", nargs="+", default=list(AGENT_NAMES))
    args = parser.parse_args()

    started = time.perf_counter()
    closed_form_section(SHARES, args.epsilon)
    agents_section(tuple(args.agents), args.env, args.runs, args.episodes, args.epsilon)
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
