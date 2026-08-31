#!/usr/bin/env python3
"""What the deadly triad costs, on the problem built to show it.

Function approximation, bootstrapping and off-policy learning are safe in any
two together. All three at once have no convergence guarantee, and Baird's
counterexample is where the missing guarantee is a fact rather than a gap in
what anybody has proved.

    python scripts/measure_triad.py
    python scripts/measure_triad.py --runs 10
    python scripts/measure_triad.py --sizes 5 6 8 10

Four things are measured.

**The run away.** `linear-td` and `gradient-td` on the same problem, from the
same weights, with the same features and the same two policies. At the
discount the literature uses, one of them reaches a value error past what a
float can hold and the other stays near two. At half that discount both reach
the answer, and nothing else about the run is different.

**It is not the step size.** Every step size runs away, and a smaller one only
takes longer. A reader who has seen a divergence blamed on a step size once
will want that ruled out, so it is ruled out here rather than asserted.

**The crossing.** Divergence needs a discount above about 0.88, and below it
the same three ingredients settle. The number is found by running the update
the model says to make, and `rel.envs.baird` derives the same number in closed
form. The two agreeing is what makes either one worth printing.

**The size of the problem.** More upper states makes the crossing lower, so a
larger version of the counterexample diverges at discounts a smaller one is
safe at. Four upper states or fewer never diverge at all.

`docs/algorithms.md` has the tables.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.linear_prediction import (
    GradientTD,
    LinearPredictor,
    SemiGradientTD,
    fixed,
)
from rel.agents.lookup import Lookup
from rel.envs.baird import Baird
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

#: The discount the literature runs this at, which is above the crossing, and
#: one below it. The second pair of rows in the first table is what makes the
#: point that the discount is the whole of the difference.
DISCOUNT = 0.99
DISCOUNTS: tuple[float, ...] = (0.99, 0.5)
STEP_SIZE = 0.05
#: An episode of the counterexample is a thousand steps, because nothing in it
#: ever ends and the environment's step limit is what stops an episode.
EPISODES = 20
STEPS_EACH = 1000

#: Step sizes for the sweep that rules the step size out.
SIZES: tuple[float, ...] = (0.005, 0.02, 0.05, 0.2)

#: How many upper states the crossing is measured at.
UPPER: tuple[int, ...] = (4, 5, 6, 8, 10, 20)

#: Settings of the expected update, which is deterministic and has no seeds.
#: The step size is five times the sampled one, because the sampled one is
#: divided by the features of a state dotted with themselves and every row of
#: this table has a squared length of five.
EXPECTED_STEP = 0.25
EXPECTED_STEPS = 4000

#: Steps of the growth rate that the bisection reads, and the rate above which
#: a run counts as running away.
#:
#: The rate is never far below zero when the update settles, because what it
#: settles on is a direction that no update can reach and that therefore
#: neither grows nor shrinks. So the question is whether the rate is above
#: zero, and a thousandth is the margin that keeps the last bits of a float
#: from answering it.
RATE_STEPS = 3000
RUNNING_AWAY = 1e-3


def starting_weights(env: Baird) -> list[float]:
    """Ones everywhere, with a ten on the lower state's own weight.

    `rel.envs.baird.STARTING_WEIGHTS` is this at six upper states, and this is
    what it means at any other number, so a run at another size starts from
    the same shape rather than from the first few numbers of a vector written
    for six.

    The length comes from the table rather than from the number of states,
    because the table is what the weights are for.
    """
    weights = [1.0] * len(env.feature_rows[0])
    weights[env.lower] = 10.0
    return weights


def expected_change(
    env: Baird,
    coder: Lookup,
    spread: list[float],
    weights: list[float],
    discount: float,
) -> list[float]:
    """The change the model says to make, averaged over where the agent is.

    No seeds and no sampling. Every state contributes what it is worth times
    how often the behaviour policy is there, so this is the average update
    rather than one draw from it, and what it does is a fact about the problem
    rather than about a run.
    """
    start = env.action_space.start
    change = [0.0] * coder.features

    def worth(state: int) -> float:
        indices, values = coder.encode(state)
        return sum(
            weights[index] * value for index, value in zip(indices, values, strict=True)
        )

    for state in range(env.observation_space.n):
        indices, values = coder.encode(state)
        share = spread[state] / coder.squared_length(values)
        here = worth(state)
        for action in env.action_space:
            taken = env.behaviour_shares[action - start]
            wanted = env.target_shares[action - start]
            if taken <= 0.0 or wanted <= 0.0:
                # The ratio is zero, and every term of the update carries it.
                continue
            for outcome in env.transitions(state, action):
                weight = share * wanted * outcome.probability
                error = outcome.reward + discount * worth(outcome.observation) - here
                for index, value in zip(indices, values, strict=True):
                    change[index] += weight * error * value

    return change


def a_predictor(
    which: str, env: Baird, seed: int, discount: float, step_size: float
) -> LinearPredictor[int]:
    """One of the two agents, started from the weights the figure starts from."""
    kind = {"linear-td": SemiGradientTD, "gradient-td": GradientTD}[which]
    agent: LinearPredictor[int] = kind(
        Rng(seed).stream("agent"),
        env.action_space,
        Lookup(env.feature_rows),
        fixed(env.behaviour_shares),
        fixed(env.target_shares),
        step_size=step_size,
        discount=discount,
    )
    # `starting_weights` rather than the constant, which is the right length
    # at six upper states and at no other. Assigning a slice would resize the
    # list rather than complain, so a run at another size would have carried
    # eight weights over a coder that wanted seven.
    agent.weights[:] = starting_weights(env)
    return agent


def one_run(
    which: str,
    seed: int,
    discount: float = DISCOUNT,
    step_size: float = STEP_SIZE,
    episodes: int = EPISODES,
) -> float:
    """The root mean square value error after a run.

    The true value is zero at every state, under either policy, because
    nothing in this environment pays anything. So this number is the size of
    what the agent believes, and every bit of it is error.
    """
    env = Baird(Rng(seed).stream("env"))
    agent = a_predictor(which, env, seed, discount, step_size)
    train(env, agent, episodes, discount=discount)
    return agent.error_against(dict.fromkeys(range(env.observation_space.n), 0.0))


def visits(env: Baird) -> list[float]:
    """How often the behaviour policy is in each state, by power iteration.

    On this environment the answer is even over all of them, and it is worked
    out rather than written down so that the expected update below is the
    update for the environment rather than for one the reader has to trust is
    the same.
    """
    states = env.observation_space.n
    spread = [1.0 / states] * states
    for _ in range(500):
        ahead = [0.0] * states
        for state, share in enumerate(spread):
            for action in env.action_space:
                taken = share * env.behaviour_shares[action - env.action_space.start]
                for outcome in env.transitions(state, action):
                    ahead[outcome.observation] += taken * outcome.probability
        spread = ahead
    return spread


def expected_size(env: Baird, discount: float, step: float, steps: int) -> float:
    """The largest weight after running the expected update, for reading."""
    coder = Lookup(env.feature_rows)
    weights = starting_weights(env)
    spread = visits(env)

    for _ in range(steps):
        change = expected_change(env, coder, spread, weights, discount)
        weights = [
            weight + step * moved for weight, moved in zip(weights, change, strict=True)
        ]
        largest = max(abs(weight) for weight in weights)
        if largest > 1e40:
            return largest
    return max(abs(weight) for weight in weights)


def growth_rate(env: Baird, discount: float, step: float, steps: int) -> float:
    """How fast the expected update grows, for each unit of step size.

    The weights are scaled back to a largest of one after every step and the
    growth that was taken out is added up, which is how a power iteration
    reads an eigenvalue. Only the second half is counted, so what is measured
    is the rate the run settles into rather than the transient it starts with.

    Reading the rate rather than waiting for a weight to pass a bar is what
    makes the crossing findable. Just above it the weights grow, and they grow
    so slowly that a run of any affordable length never passes any bar worth
    setting.
    """
    coder = Lookup(env.feature_rows)
    weights = starting_weights(env)
    spread = visits(env)

    # At least one step is counted however few were asked for, because a run
    # of one step halves to none and the rate would be divided by nothing.
    counted = max(1, steps // 2)
    first = steps - counted

    grown = 0.0
    for number in range(steps):
        change = expected_change(env, coder, spread, weights, discount)
        weights = [
            weight + step * moved for weight, moved in zip(weights, change, strict=True)
        ]
        largest = max(abs(weight) for weight in weights)
        if largest == 0.0:
            # Every weight is exactly zero, which is the opposite of running
            # away and is as far from it as a run can get. Nothing can be
            # scaled back to a largest of one from here.
            return -math.inf
        if number >= first:
            grown += math.log(largest)
        weights = [weight / largest for weight in weights]

    return grown / (counted * step)


def runs_away(env: Baird, discount: float, step: float, steps: int) -> bool:
    return growth_rate(env, discount, step, steps) > RUNNING_AWAY


def crossing(upper: int, step: float, steps: int, rounds: int = 22) -> float:
    """The discount at which the expected update starts to run away.

    By bisection, the same way `rel.envs.continuing` finds the discount where
    the two loops swap places. The closed form is in `rel.envs.baird` and this
    does not read it.

    Not a number, when no discount below one runs away. That is four upper
    states or fewer, and it is an answer rather than a failure.
    """
    env = Baird(Rng(1).stream("env"), upper=upper)
    if not runs_away(env, 0.999, step, steps):
        return float("nan")

    low, high = 0.0, 0.999
    for _ in range(rounds):
        middle = (low + high) / 2.0
        if runs_away(env, middle, step, steps):
            high = middle
        else:
            low = middle
    return (low + high) / 2.0


def run_away_section(runs: int, episodes: int, discounts: tuple[float, ...]) -> None:
    print(
        f"Baird's counterexample. Every reward is zero, so every state is "
        f"worth zero,\nand eight weights of zero say so exactly. "
        f"{runs} seeds, {episodes * STEPS_EACH} steps each,\n"
        f"step size {STEP_SIZE:g}.\n"
    )

    rows: list[list[str]] = []
    for discount in discounts:
        for which in ("linear-td", "gradient-td"):
            errors = [
                one_run(which, seed, discount=discount, episodes=episodes)
                for seed in range(1, runs + 1)
            ]
            rows.append(
                [
                    which,
                    f"{discount:g}",
                    f"{statistics.median(errors):.4g}",
                    " ".join(f"{error:.3g}" for error in errors),
                ]
            )

    for line in table(
        ["agent", "discount", "median value error", "every seed"],
        rows,
        align=["left", "right", "right", "left"],
    ):
        print(f"  {line}")


def step_size_section(runs: int, episodes: int, sizes: tuple[float, ...]) -> None:
    print(
        "\nIt is not the step size. A smaller one takes longer to run away and\n"
        "runs away.\n"
    )
    rows: list[list[str]] = []
    for size in sizes:
        errors = [
            one_run("linear-td", seed, step_size=size, episodes=episodes)
            for seed in range(1, runs + 1)
        ]
        rows.append([f"{size:g}", f"{statistics.median(errors):.4g}"])

    for line in table(
        ["step size", "median value error of linear-td"],
        rows,
        align=["right", "right"],
    ):
        print(f"  {line}")


def crossing_section(step: float, steps: int, rate_steps: int) -> None:
    env = Baird(Rng(1).stream("env"))
    measured = crossing(env.upper, step, rate_steps)
    print(
        f"\nThe crossing. Running the update the model says to make, and "
        f"bisecting on\nwhether it runs away: {measured:.4f}. "
        f"The closed form is {env.runs_away_above():.4f}.\n"
    )

    rows = [
        [f"{discount:g}", f"{expected_size(env, discount, step, steps):.4g}"]
        for discount in (0.5, 0.8, 0.85, 0.88, 0.9, 0.95, 0.99)
    ]
    for line in table(
        ["discount", "largest weight after the expected update"],
        rows,
        align=["right", "right"],
    ):
        print(f"  {line}")


def size_section(sizes: tuple[int, ...], step: float, rate_steps: int) -> None:
    print(
        "\nThe size of the problem. More upper states lowers the crossing, so a\n"
        "larger counterexample runs away at discounts a smaller one settles at.\n"
    )
    rows: list[list[str]] = []
    for upper in sizes:
        env = Baird(Rng(1).stream("env"), upper=upper)
        closed = env.runs_away_above()
        measured = crossing(upper, step, rate_steps)
        # The two columns are worked out without reading each other, so a row
        # where they agree is two answers rather than one printed twice.
        rows.append(
            [
                str(upper),
                "never" if closed >= 1.0 else f"{closed:.4f}",
                "never" if measured != measured else f"{measured:.4f}",
            ]
        )

    for line in table(
        ["upper states", "closed form", "measured"],
        rows,
        align=["right", "right", "right"],
    ):
        print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="seeds per agent")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(UPPER))
    parser.add_argument("--discounts", type=float, nargs="+", default=list(DISCOUNTS))
    parser.add_argument("--step-sizes", type=float, nargs="+", default=list(SIZES))
    parser.add_argument("--expected-step", type=float, default=EXPECTED_STEP)
    parser.add_argument("--expected-steps", type=int, default=EXPECTED_STEPS)
    parser.add_argument("--rate-steps", type=int, default=RATE_STEPS)
    args = parser.parse_args()

    started = time.perf_counter()
    run_away_section(args.runs, args.episodes, tuple(args.discounts))
    step_size_section(args.runs, args.episodes, tuple(args.step_sizes))
    crossing_section(args.expected_step, args.expected_steps, args.rate_steps)
    size_section(tuple(args.sizes), args.expected_step, args.rate_steps)

    print(
        "\nWhat is left at 0.99 is one direction, in which every state's estimate "
        "is\nthe same number. A constant estimate creates an error of one minus "
        "the\ndiscount times that constant, so at 0.99 the last direction goes a "
        "hundred\ntimes more slowly than at 0.5. --episodes 200 leaves it at 1.85."
    )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
