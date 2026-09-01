#!/usr/bin/env python3
"""What drawing from the buffer by priority does, and what it costs to leave
the correction out.

    python scripts/measure_prioritised.py
    python scripts/measure_prioritised.py --runs 20 --episodes 300

A buffer that draws evenly spends most of a batch on steps the agent already
predicts. Drawing in proportion to the size of the last error spends the batch
where there is something to learn. That is the case for it, and it is a good
one.

It also breaks the estimate. Fitting to a batch is an average over the steps in
it, and an average only estimates what it is meant to estimate if the steps
arrived with the right probability. Drawing by priority changes those
probabilities on purpose, so the agent now settles somewhere else. It is not a
faster route to the same answer. `weighting` puts it back.

This script measures both halves of that.

## Where it settles, against an answer that is known

The first section is not a task. It is one state, one action and one number to
learn: a stream of rewards with no future, so the target of every step is the
reward itself and the network is fitting a constant to a fixed set of numbers.

That is worth doing because the answer is arithmetic. The constant that
minimises the mean squared error over a set of numbers is their mean. The
constant an uncorrected priority draw settles at instead is where the pull of
the numbers above it balances the pull of the numbers below it once each is
counted in proportion to how far away it is, which is the root of

    sum over the numbers of |c - y| * (c - y) = 0

and the script solves that by bisection rather than quoting it. The rewards are
skewed on purpose, because on a symmetric set the two answers are the same one
and the measurement would show nothing.

The uncorrected estimate does not reach that root exactly, and the reason is in
the buffer rather than in the argument. Every step put in is given the largest
priority the buffer has held, so each of them is pulled back up once per pass
whatever the agent now believes, and that pull is towards the even draw. The
measured point sits between the mean and the root, on the far side of anything
the even draw does.

## Whether it helps

The second section is the cart pole, which is the task the value network was
built against. Three settings, the same seeds, the same everything else, and
`rel.compare` for whether a difference is real. The seconds are in the table
because a priority draw adds up the weights of the whole buffer once per batch
and an even draw does not, so the rows are not the same amount of work.

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
from rel.agents.base import Transition
from rel.agents.features import one_hot
from rel.agents.value_network import DeepQ
from rel.compare import compare
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import evaluate, train
from rel.ui.table import table

#: The three settings, as (name, priority, weighting). The middle one is the
#: mistake: it draws by priority and corrects nothing.
SETTINGS: tuple[tuple[str, float, float], ...] = (
    ("even", 0.0, 0.0),
    ("priority", 0.6, 0.0),
    ("corrected", 0.6, 0.4),
)

#: The settings used for the settling measurement, where the point is the
#: fixed point rather than the speed, so the powers are the full ones.
EXACT_SETTINGS: tuple[tuple[str, float, float], ...] = (
    ("even", 0.0, 0.0),
    ("priority", 1.0, 0.0),
    ("corrected", 1.0, 1.0),
)

#: Fifteen rewards of nothing and five of one. Skewed, because a symmetric set
#: puts both answers in the same place and the measurement would show nothing.
REWARDS: tuple[float, ...] = (0.0,) * 15 + (1.0,) * 5

ENVIRONMENT = "cartpole"
EPISODES = 150
RUNS = 10
PASSES = 400
SETTLE_SEEDS = 5


# -- Where it settles --------------------------------------------------------


def uncorrected_answer(rewards: tuple[float, ...]) -> float:
    """The constant an uncorrected priority draw is pulled towards.

    The root of `sum |c - y| * (c - y) = 0`, found by bisection between the
    smallest reward and the largest. The sum rises with `c` everywhere in that
    range, so there is one root and bisection finds it.
    """
    low, high = min(rewards), max(rewards)
    for _ in range(200):
        middle = (low + high) / 2.0
        pull = sum(abs(middle - y) * (middle - y) for y in rewards)
        if pull < 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def where_it_settles(
    seed: int,
    priority: float,
    weighting: float,
    rewards: tuple[float, ...],
    passes: int,
) -> float:
    """The number a value network reaches on a stream of rewards with no future.

    One state and one action, so the network has one number to say. Every step
    is terminated and the discount is zero, so the target of a step is its
    reward and nothing else, and what the agent is doing is fitting a constant
    to a fixed set of numbers.

    The buffer holds exactly as many steps as there are rewards and the stream
    goes round in order, so after the first pass the buffer holds each reward
    once and holds it for the rest of the run.
    """
    agent: DeepQ[int] = DeepQ(
        Rng(seed),
        Discrete(1),
        one_hot(1),
        1,
        hidden=8,
        step_size=0.01,
        discount=0.0,
        epsilon=0.0,
        replay=len(rewards),
        batch=8,
        target_refresh=0,
        priority=priority,
        weighting=weighting,
    )
    for _ in range(passes):
        for reward in rewards:
            agent.observe(Transition(0, 0, reward, 0, True, False))
    return agent.action_values(0)[0]


def settling_section(rewards: tuple[float, ...], passes: int, seeds: int) -> None:
    mean = statistics.mean(rewards)
    root = uncorrected_answer(rewards)

    print(
        f"One state, one action and {len(rewards)} rewards with no future, so"
        f" the agent is\nfitting a constant to a fixed set of numbers."
        f" {seeds} seeds, {passes} passes each.\n"
    )
    print(f"  the mean of the rewards, which is what fitting them means  {mean:.4f}")
    print(f"  where an uncorrected priority draw is pulled instead       {root:.4f}\n")

    rows = []
    for name, priority, weighting in EXACT_SETTINGS:
        got = [
            where_it_settles(seed, priority, weighting, rewards, passes)
            for seed in range(1, seeds + 1)
        ]
        settled = statistics.mean(got)
        rows.append(
            [
                name,
                f"{priority:g}",
                f"{weighting:g}",
                f"{settled:.4f}",
                f"{settled - mean:+.4f}",
                f"{max(got) - min(got):.4f}",
            ]
        )

    for line in table(
        ["setting", "priority", "weighting", "settled at", "off the mean", "spread"],
        rows,
        align=["left"] + ["right"] * 5,
    ):
        print(f"  {line}")


# -- Whether it helps --------------------------------------------------------


def one_setting(
    priority: float, weighting: float, runs: int, episodes: int, env_name: str
) -> tuple[list[float], float]:
    """What every seed's greedy policy was worth, and the seconds the row took."""
    finals: list[float] = []
    started = time.perf_counter()

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(env_name, root.stream("env"))
        agent = AGENTS.make(
            "deep-q",
            root.stream("agent"),
            env,
            priority=priority,
            weighting=weighting,
        )
        train(env, agent, episodes, discount=env.spec.suggested_discount)
        finals.append(statistics.mean(evaluate(env, agent, 5).returns))

    return finals, time.perf_counter() - started


def helping_section(env_name: str, runs: int, episodes: int, rng: Rng) -> None:
    print(
        f"\n\nThe same three settings on a task. {env_name}, {runs} seeds,"
        f" {episodes} episodes,\nread as the return of the greedy policy at"
        f" the end.\n"
    )

    got: dict[str, list[float]] = {}
    seconds: dict[str, float] = {}
    for name, priority, weighting in SETTINGS:
        got[name], seconds[name] = one_setting(
            priority, weighting, runs, episodes, env_name
        )

    even = SETTINGS[0][0]
    rows = []
    for name, priority, weighting in SETTINGS:
        row = [
            name,
            f"{priority:g}",
            f"{weighting:g}",
            f"{statistics.mean(got[name]):.1f}",
            f"{seconds[name]:.0f}",
        ]
        if name == even:
            row += ["-", "-", "-"]
        else:
            answer = compare(got[name], got[even], rng)
            row += [
                f"{answer.difference:+.1f}",
                f"[{answer.low:+.1f}, {answer.high:+.1f}]",
                f"{answer.p_value:.4f}",
            ]
        rows.append(row)

    for line in table(
        [
            "setting",
            "priority",
            "weighting",
            "return",
            "seconds",
            f"minus {even}",
            "95 percent interval",
            "p",
        ],
        rows,
        align=["left"] + ["right"] * 7,
    ):
        print(f"  {line}")

    floor = 2.0 / 2.0**runs
    if floor > 0.05:
        print(
            f"\n  {runs} seeds cannot give a p below {floor:.4f} however large"
            f" the difference is."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=ENVIRONMENT)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--runs", type=int, default=RUNS, help="seeds per setting")
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--settle-seeds", type=int, default=SETTLE_SEEDS)
    parser.add_argument(
        "--skip-task",
        action="store_true",
        help="run only the section whose answer is known",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    settling_section(REWARDS, args.passes, args.settle_seeds)
    if not args.skip_task:
        helping_section(args.env, args.runs, args.episodes, Rng(9).stream("compare"))
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
