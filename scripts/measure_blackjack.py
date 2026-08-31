#!/usr/bin/env python3
"""How close a Monte Carlo agent gets to an answer that is known exactly.

Blackjack is where the book introduces Monte Carlo, because the dealer playing
out his whole hand is awkward to write as a one step model and easy to sample.
`rel.envs.blackjack` writes it out anyway, so the optimal policy and the value
of the deal are known before any hand is played, and an agent that learns from
episodes can be scored against the answer rather than against another agent.

    python scripts/measure_blackjack.py
    python scripts/measure_blackjack.py --budgets 50000 200000
    python scripts/measure_blackjack.py --steps none 0.05

Two things are reported for every setting.

**What the learned policy is worth**, worked out exactly with
`rel.agents.dp.evaluate_policy` over the same model. Not the return the agent
happened to collect, which mixes what it learned with the cards it was dealt.

**How many of the two hundred squares it plays differently from the optimum**,
and how much those squares are worth. A count on its own says nothing about
size: the squares an agent gets wrong are the ones it rarely reaches, and a
policy twenty squares from the optimum can be worth almost the optimum.

Blackjack decides nearly every square, which is worth saying because the
gambler's problem next to it decides none. The smallest gap between the two
actions anywhere on this board is 0.0025 and the middle one is 0.32, so a
square played differently from the optimum really is a mistake here.

The step size is the question. `MonteCarloControl` takes a running average with
no step size given, which is the textbook rule, and a fixed weight on the
newest return with one. Its docstring says the fixed weight is what control
needs even in a fixed environment, because the policy keeps changing and a
return from long ago was collected by a policy that no longer exists. That is a
claim, and this is the measurement of it.

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
from rel.agents.dp import Solution, evaluate_policy, value_iteration
from rel.envs.blackjack import Blackjack
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

#: The budgets to run. Blackjack hands are two or three steps, so a hundred
#: thousand of them is a few seconds.
BUDGETS: tuple[int, ...] = (50_000, 200_000, 500_000)

#: The step sizes to run. `None` is the running average, which is the textbook
#: rule for prediction and the one this measurement is about.
STEPS: tuple[float | None, ...] = (None, 0.01, 0.05, 0.1)

#: How much better one action has to be for a square to count as decided. Every
#: square of this board clears it but two, which is the point of printing it.
MATTERS = 0.01

RUNS = 5
EPSILON = 0.1


@cache
def answer() -> tuple[Blackjack, Solution]:
    """The exact optimum, swept once and shared by every row."""
    env = Blackjack(Rng(1).stream("env"))
    return env, value_iteration(env, discount=1.0)


@cache
def margins() -> tuple[float, ...]:
    """How much better the best action is than the other, at each square.

    Read off the optimal values. The sum of these over the squares an agent
    plays differently is what its mistakes are worth, which is a different
    number from how many of them there are.
    """
    env, best = answer()
    gaps = []
    for state in range(env.over):
        worths = [
            sum(
                branch.probability
                * (
                    branch.reward
                    + (0.0 if branch.terminated else best.values[branch.observation])
                )
                for branch in env.transitions(state, action)
            )
            for action in env.action_space
        ]
        gaps.append(abs(worths[0] - worths[1]))
    return tuple(gaps)


def one_run(step: float | None, episodes: int, seed: int) -> tuple[float, int, float]:
    """What one agent learned: its exact value, and the squares it plays wrongly.

    Three numbers: what the policy is worth, how many squares it plays
    differently from the optimum, and what those squares are worth added up.
    """
    env, best = answer()
    playing = Blackjack(Rng(seed).stream("env"))
    agent = AGENTS.make(
        "monte-carlo",
        Rng(seed).stream("agent"),
        playing,
        step_size=step,
        discount=1.0,
        epsilon=EPSILON,
    )
    train(playing, agent, episodes, discount=1.0)

    policy = [agent.greedy(state) for state in range(env.observation_space.n)]
    got = evaluate_policy(env, policy, discount=1.0)

    wrong = [state for state in range(env.over) if policy[state] != best.policy[state]]
    return got.start_value, len(wrong), sum(margins()[state] for state in wrong)


def step_section(
    budgets: tuple[int, ...],
    steps: tuple[float | None, ...],
    runs: int,
    matters: float,
) -> None:
    env, best = answer()
    settled = sum(1 for gap in margins() if gap >= matters)
    print(
        f"Monte Carlo control on blackjack, {runs} seeds, epsilon "
        f"{EPSILON:g}.\nThe deal is worth {best.start_value:.5f} played "
        f"perfectly, and {settled} of the {env.over} squares\nhave one action "
        f"worth {matters:g} more than the other.\n"
    )

    rows = []
    for episodes in budgets:
        for step in steps:
            got = [one_run(step, episodes, seed) for seed in range(1, runs + 1)]
            rows.append(
                [
                    f"{episodes:,}",
                    "running average" if step is None else f"{step:g}",
                    f"{statistics.mean(value for value, _, _ in got):+.5f}",
                    f"{statistics.mean(wrong for _, wrong, _ in got):.1f}",
                    f"{statistics.mean(lost for _, _, lost in got):.2f}",
                ]
            )

    for line in table(
        [
            "episodes",
            "step size",
            "what it learned is worth",
            "squares apart",
            "what those squares are worth",
        ],
        rows,
        align=["right", "right", "right", "right", "right"],
    ):
        print(f"  {line}")

    print(
        "\n  'what it learned is worth' is the exact value of the greedy "
        "policy over\n  the same model, not the return the agent collected. "
        "The last column adds up\n  what the squares it plays differently are "
        "worth, which is a different\n  number from how many of them there "
        "are: the squares an agent gets wrong\n  are the ones it rarely "
        "reaches."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["none" if step is None else f"{step:g}" for step in STEPS],
        help="step sizes, with 'none' for the running average",
    )
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--matters", type=float, default=MATTERS)
    args = parser.parse_args()

    steps = tuple(
        None if given.lower() == "none" else float(given) for given in args.steps
    )

    started = time.perf_counter()
    step_section(tuple(args.budgets), steps, args.runs, args.matters)
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
