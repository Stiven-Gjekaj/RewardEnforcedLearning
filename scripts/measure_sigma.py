#!/usr/bin/env python3
"""Does part of a sample and part of an expectation beat either on its own?

`q-sigma` has tree backup at sigma of nothing and n step SARSA with a control
variate at sigma of one. Sutton and Barto raise the middle as a question and
leave it open, and this is the measurement of it.

    python scripts/measure_sigma.py
    python scripts/measure_sigma.py --env maze --runs 20
    python scripts/measure_sigma.py --sigmas 0 0.5 1 --episodes 800

**Every sigma is read at its own best step size.** Sigma changes how large the
target is as well as what it is made of, so a comparison at one step size would
find which sigma suits that step size and report it as which sigma is better.

**Every sigma is compared against tree backup, paired by seed.** Tree backup is
the end this project already had, so it is the row the others are read against.
A mean beside a mean says which is larger and says nothing about whether the
seeds agree.

**The schedule the book suggests is a row of its own.** Sigma falling from one
to nothing over the run samples early and averages late, which is the shape of
the suggestion, and it is measured rather than assumed.

**The score is the exact value of what was learned, and it is coarse.** A cliff
walk policy is worth -13 along the edge, -15 one row up, -17 two rows up, and
little else, so a mean over ten seeds moves in steps of a fifth. That is what
makes the paired interval worth reading rather than the mean alone.

`docs/algorithms.md` has the table.
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
from rel.agents.dp import evaluate_policy, value_iteration
from rel.compare import compare
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.schedules import Schedule, linear
from rel.training import train
from rel.ui.table import table

#: The sigmas to run. Nothing is tree backup and one is the sample.
SIGMAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: The step sizes each sigma is swept over. All of them see all of these.
STEP_SIZES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4)

ENVIRONMENT = "cliff"
EPISODES = 400
RUNS = 10
STEPS = 3
EPSILON = 0.1


@cache
def best_possible(grid: str) -> tuple[float, float]:
    """The best return this grid allows, and the discount it is run at."""
    env = ENVIRONMENTS.make(grid, Rng(1).stream("env"))
    discount = env.spec.suggested_discount
    return value_iteration(env, discount=discount).start_value, discount


def falling(episodes: int) -> Schedule:
    """Sigma from one to nothing over the run, which is the book's suggestion.

    Read on steps rather than episodes, because that is what `current_sigma`
    is given, so the length is an estimate of how many steps the run takes
    rather than how many episodes it has.
    """
    return linear(1.0, 0.0, episodes * 20)


def one_run(
    grid: str,
    sigma: float | Schedule,
    step_size: float,
    episodes: int,
    seed: int,
) -> float:
    """The exact value of what one seed learned, or the worst there is.

    Exact rather than the return it collected, because the grid has a model
    and a policy that never reaches the goal has no value at all rather than a
    small one.
    """
    _, discount = best_possible(grid)
    env = ENVIRONMENTS.make(grid, Rng(seed).stream("env"))
    agent = AGENTS.make(
        "q-sigma",
        Rng(seed).stream("agent"),
        env,
        n=STEPS,
        sigma=sigma,
        step_size=step_size,
        discount=discount,
        epsilon=EPSILON,
    )
    train(env, agent, episodes, discount=discount)

    policy = [agent.greedy(state) for state in range(env.observation_space.n)]
    report = evaluate_policy(env, policy, discount=discount)
    if not report.reaches_end:
        # A policy that circles for ever is worth the cap rather than minus
        # infinity, so one stuck seed does not swallow a whole row.
        return -float(env.spec.max_episode_steps or 1000)
    return report.start_value


def best_of(
    grid: str,
    sigma: float | Schedule,
    steps: tuple[float, ...],
    episodes: int,
    runs: int,
) -> tuple[list[float], float]:
    """Every seed at the best of the swept step sizes, and the step that won."""
    got = {
        step: [
            one_run(grid, sigma, step, episodes, seed) for seed in range(1, runs + 1)
        ]
        for step in steps
    }
    best = max(got, key=lambda step: statistics.mean(got[step]))
    return got[best], best


def sigma_section(
    grid: str,
    sigmas: tuple[float, ...],
    steps: tuple[float, ...],
    episodes: int,
    runs: int,
    rng: Rng,
) -> None:
    reachable, discount = best_possible(grid)
    print(
        f"{grid}, {episodes} episodes, {runs} seeds, n of {STEPS}, epsilon "
        f"{EPSILON:g},\ndiscount {discount:g}. The best possible return is "
        f"{reachable:.3f}.\nEvery sigma is swept over {len(steps)} step sizes "
        f"and read at its own best.\n"
    )

    against, at_against = best_of(grid, 0.0, steps, episodes, runs)
    rows = []
    named: list[tuple[str, float | Schedule]] = [
        (f"{sigma:g}", sigma) for sigma in sigmas
    ]
    named.append(("1 falling to 0", falling(episodes)))

    for label, sigma in named:
        got, at_step = (
            (against, at_against)
            if sigma == 0.0
            else best_of(grid, sigma, steps, episodes, runs)
        )
        answer = compare(got, against, rng)
        rows.append(
            [
                label,
                f"{at_step:g}",
                f"{statistics.mean(got):.3f}",
                "-" if sigma == 0.0 else f"{answer.difference:+.3f}",
                "-" if sigma == 0.0 else f"[{answer.low:+.3f}, {answer.high:+.3f}]",
                "-" if sigma == 0.0 else f"{answer.p_value:.3f}",
            ]
        )

    for line in table(
        [
            "sigma",
            "at step",
            "greedy, exactly",
            "over tree backup",
            "95 percent interval",
            "p",
        ],
        rows,
        align=["right"] * 6,
    ):
        print(f"  {line}")

    floor = 2.0 / 2.0**runs
    if floor > 0.05:
        print(
            f"\n  {runs} seeds cannot give a p below {floor:.4f} however large"
            f" the difference is.\n  Six seeds is the fewest that can reach "
            f"0.05."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=ENVIRONMENT)
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    parser.add_argument("--step-sizes", type=float, nargs="+", default=list(STEP_SIZES))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    # Tree backup is the row every other row is read against, so it goes in
    # whether or not it was asked for.
    sigmas = tuple(sorted({*args.sigmas, 0.0}))

    started = time.perf_counter()
    sigma_section(
        args.env,
        sigmas,
        tuple(args.step_sizes),
        args.episodes,
        args.runs,
        Rng(13).stream("compare"),
    )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
