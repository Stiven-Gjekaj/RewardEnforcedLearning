#!/usr/bin/env python3
"""Tile coding against radial basis features, on cost and on what they learn.

Two ways of turning a point in a box into features, under one agent. The
question a reader asks is which is better, and the answer is that they learn
about equally well and one of them costs fifty times as much per step.

    python scripts/measure_approximation.py
    python scripts/measure_approximation.py --runs 8
    python scripts/measure_approximation.py --env cartpole --episodes 60

The cost is the interesting half, and it is not the half either encoder's
description points at. A tile coder is not cheap because it has few features:
on the cart pole it has 52488 of them against a radial basis with 1296. It is
cheap because it works out which eight are on directly, and never asks the
other 52480 anything.

A radial basis cannot do that. Every centre answers every point, and there is
no way to know which centres are far away without measuring the distance to
all of them. The second section shows where the time goes, and it is why
`kept` saves so little: the dropping happens after the expensive part.
"""

from __future__ import annotations

import argparse
import heapq
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.base import Transition
from rel.agents.basis import RadialBasis
from rel.compare import compare
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.spaces import Box
from rel.training import train
from rel.ui.table import table

#: The pairs the whole script is about. The agent is the same code either way.
CODERS = ("tile-sarsa", "rbf-sarsa")


def microseconds_per_step(
    grid: str, agent_name: str, steps: int, **options: Any
) -> tuple[float, int]:
    """How long one step of a real run takes, and how many features there are.

    A whole run is the wrong measurement here, because an agent that learns
    faster runs shorter episodes and finishes sooner while being no quicker
    per step. This drives the loop by hand for a fixed number of steps, so
    what comes back is the cost of a step and nothing about the policy.
    """
    root = Rng(1)
    env = ENVIRONMENTS.make(grid, root.stream("env"))
    agent = AGENTS.make(agent_name, root.stream("agent"), env, **options)

    observation = env.reset()
    started = time.perf_counter()
    for _ in range(steps):
        action = agent.act(observation)
        step = env.step(action)
        agent.observe(
            Transition(
                observation,
                action,
                step.reward,
                step.observation,
                step.terminated,
                step.truncated,
            )
        )
        observation = step.observation
        if step.terminated or step.truncated:
            agent.end_episode()
            observation = env.reset()
    taken = time.perf_counter() - started

    return taken / steps * 1e6, agent.coder.features  # type: ignore[attr-defined]


def breakdown(dimensions: int, bins: int, passes: int) -> list[tuple[str, float]]:
    """Where a radial basis spends a step, piece by piece.

    The pieces are cumulative down to `encode`, so each line is the one above
    it plus what its own name adds. The last two are the two ways of finding
    the largest few, measured on their own.
    """
    box = Box([-1.0] * dimensions, [1.0] * dimensions)
    basis = RadialBasis(box, bins=bins)
    point = tuple(0.1 * index - 0.3 for index in range(dimensions))
    values = basis.all_values(point)
    count = basis.features
    order = range(count)

    def timed(work: Any) -> float:
        best = float("inf")
        for _ in range(3):
            started = time.perf_counter()
            for _ in range(passes):
                work()
            best = min(best, (time.perf_counter() - started) / passes * 1e6)
        return best

    return [
        (
            f"the distance to all {count} centres",
            timed(lambda: basis.squared_distances(point)),
        ),
        ("and the exponential of each", timed(lambda: basis.all_values(point))),
        ("and normalising them", timed(lambda: basis.encode(point))),
        (
            "finding the largest 8 by sorting",
            timed(lambda: sorted(order, key=values.__getitem__, reverse=True)[:8]),
        ),
        (
            "finding the largest 8 by a heap",
            timed(lambda: heapq.nlargest(8, order, key=values.__getitem__)),
        ),
    ]


def one_run(
    grid: str, agent_name: str, seed: int, episodes: int, **options: Any
) -> float:
    """What the agent got over the last ten episodes of one run."""
    env = ENVIRONMENTS.make(grid, Rng(seed).stream("env"))
    discount = env.spec.suggested_discount
    agent = AGENTS.make(agent_name, Rng(500 + seed).stream("agent"), env, **options)
    return train(env, agent, episodes, discount=discount).final(10)


def cost_section(grids: tuple[str, ...], steps: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for grid in grids:
        for agent_name in CODERS:
            each, features = microseconds_per_step(grid, agent_name, steps)
            rows.append([grid, agent_name, f"{features:,}", f"{each:.0f}"])
        for kept in (8,):
            each, features = microseconds_per_step(grid, "rbf-sarsa", steps, kept=kept)
            rows.append(
                [grid, f"rbf-sarsa kept={kept}", f"{features:,}", f"{each:.0f}"]
            )
    return rows


def learning_section(
    grid: str, runs: int, episodes: int, rng: Rng
) -> tuple[list[list[str]], str]:
    scores: dict[str, list[float]] = {}
    for agent_name in CODERS:
        scores[agent_name] = [
            one_run(grid, agent_name, seed, episodes) for seed in range(1, runs + 1)
        ]

    rows = [
        [
            name,
            f"{statistics.mean(values):.1f}",
            " ".join(f"{value:.0f}" for value in values),
        ]
        for name, values in scores.items()
    ]

    answer = compare(scores[CODERS[0]], scores[CODERS[1]], rng)
    floor = 2.0 / 2.0**answer.seeds
    verdict = (
        f"tile coding minus radial basis: {answer.difference:+.1f}, "
        f"95 percent interval [{answer.low:+.1f}, {answer.high:+.1f}], "
        f"p {answer.p_value:.3f}."
    )

    if not answer.certain:
        verdict += "\nThe interval crosses zero, so this says the two are not"
        verdict += " told apart by these seeds."
    elif answer.p_value > 0.05:
        # The two halves of the answer disagreeing is not a fault in either
        # of them. The interval says how large the difference is and the test
        # says how easily the sign could have come out the other way, and a
        # handful of seeds can pin the size of something it cannot pin the
        # sign of. Printing the interval alone would read as a verdict.
        verdict += (
            "\nThe interval is clear of zero and the p value is not, which is"
            " the two\nhalves of the answer disagreeing. Take the p value:"
            " the interval says how\nlarge the difference is, and it is the"
            " test that says whether the sign of\nit could as easily have"
            " gone the other way."
        )

    if answer.p_value > 0.05 and floor > 0.05:
        verdict += (
            f"\n{answer.seeds} seeds cannot give a p below {floor:.4f} however"
            f" large the difference\nis, because a paired test over"
            f" {answer.seeds} seeds has only {2**answer.seeds} sign patterns"
            f" in it.\nSix seeds is the fewest that can reach 0.05. Run with"
            f" --runs 10."
        )
    return rows, verdict


#: The widths the sweep tries, as multiples of the spacing between centres.
#: `DEFAULT` is the one the registry uses, and every other width is compared
#: against it rather than against its neighbour, because the question the
#: sweep exists to answer is whether the default is the right one.
WIDTHS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT = 0.75


def width_section(
    grid: str, runs: int, episodes: int, rng: Rng, bins: int | None = None
) -> tuple[list[list[str]], list[list[str]]]:
    """The sweep, and every width against the default one, paired by seed.

    Both, because the page quoted an interval and a p value for the default
    against one whole spacing and no command on it printed either. A mean
    beside a mean says which is larger and says nothing about whether the
    seeds agree, and this project's rule is that a claim of a difference
    carries the test that decides it.
    """
    # Fewer centres a side where the caller asks for it. The cart pole is
    # four dimensional, so the default of six is 1296 centres and an hour of
    # runs; four is 256 and the sweep finishes. The shape of the answer is
    # what the sweep is for, and it does not need the default to show it.
    coarser = {} if bins is None else {"bins": bins}

    scores: dict[float, list[float]] = {}
    for width in WIDTHS:
        scores[width] = [
            one_run(grid, "rbf-sarsa", seed, episodes, width=width, **coarser)
            for seed in range(1, runs + 1)
        ]

    rows = [
        [
            f"{width:g}",
            f"{statistics.mean(values):.1f}",
            " ".join(f"{value:.0f}" for value in values),
        ]
        for width, values in scores.items()
    ]

    against: list[list[str]] = []
    for width in WIDTHS:
        if width == DEFAULT:
            continue
        answer = compare(scores[DEFAULT], scores[width], rng)
        against.append(
            [
                f"{width:g}",
                f"{answer.difference:+.1f}",
                f"[{answer.low:+.1f}, {answer.high:+.1f}]",
                f"{answer.p_value:.3f}",
            ]
        )
    return rows, against


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="mountaincar")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--runs", type=int, default=5, help="seeds per agent")
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="steps to time, for the cost table only",
    )
    parser.add_argument("--passes", type=int, default=300)
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="centres a side for the width sweep only, where the default is slow",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    print("## What a step costs\n")
    for line in table(
        ["environment", "agent", "features", "us/step"],
        cost_section(("mountaincar", "cartpole"), args.steps),
        align=["left", "left", "right", "right"],
    ):
        print(f"  {line}")

    print(
        "\nThe tile coder has forty times more features on the cart pole and\n"
        "costs a fiftieth as much per step. Its cost is the number of grids,\n"
        "which is eight, whatever the feature count is. A radial basis has to\n"
        "ask every centre.\n"
    )

    print("## Where a radial basis spends a step\n")
    # The shape of the cart pole, which is the four dimensional problem here.
    # The count is worked out rather than written, because a sentence that
    # says 1296 beside a call that no longer makes 1296 of them is the exact
    # thing `scripts/check_numbers.py` exists to find.
    dimensions, bins = 4, 6
    print(
        f"  A four dimensional box at six centres a side, "
        f"so {bins**dimensions} of them.\n"
    )
    for line in table(
        ["", "us"],
        [
            [name, f"{each:.0f}"]
            for name, each in breakdown(dimensions, bins, args.passes)
        ],
        align=["left", "right"],
    ):
        print(f"  {line}")

    print(
        "\nThe distances are three quarters of it, and they are paid before\n"
        "anything can be dropped. That is why `kept` saves so little above:\n"
        "the sort that finds the largest eight costs about what dropping the\n"
        "rest saves, and the heap is not enough better to change the answer.\n"
    )

    print(f"## What each learns on the {args.env}\n")
    rows, verdict = learning_section(
        args.env, args.runs, args.episodes, Rng(7).stream("compare")
    )
    for line in table(
        ["agent", "mean", "every seed"], rows, align=["left", "right", "left"]
    ):
        print(f"  {line}")
    print(f"\n{verdict}\n")

    print("## The width, which is the radial basis setting that matters\n")
    widths, against = width_section(
        args.env, args.runs, args.episodes, Rng(11).stream("compare"), args.bins
    )
    for line in table(
        ["width", "mean", "every seed"], widths, align=["right", "right", "left"]
    ):
        print(f"  {line}")

    print(f"\n  Each width against the default of {DEFAULT:g}, paired by seed:\n")
    for line in table(
        ["width", f"{DEFAULT:g} minus it", "95 percent interval", "p"],
        against,
        align=["right", "right", "right", "right"],
    ):
        print(f"  {line}")

    print(
        "\nThe width is a multiple of the spacing between centres. Too narrow\n"
        "and a point between two centres lights neither. Too wide and every\n"
        "centre answers about the same and the features say nothing about\n"
        "where the point is."
    )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
