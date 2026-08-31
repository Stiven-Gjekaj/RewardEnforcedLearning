#!/usr/bin/env python3
"""How much of the gambler's famous policy picture is the problem, and how much
is the solver.

Sutton and Barto, example 4.3, is usually drawn twice: the value of each
capital, and the stake to make at it. The second picture is a jagged staircase,
and it is the one people remember.

    python scripts/measure_gambler.py
    python scripts/measure_gambler.py --coins 0.25 0.4 0.5
    python scripts/measure_gambler.py --goal 50

This asks how much of that staircase survives a change that cannot change the
answer. Two things are varied and neither is part of the problem: how tightly
the sweep is run, and which of the two solvers runs it. Both give the same
values. Where they give different stakes, the stakes were never determined by
the problem.

**The fair coin is the control.** A fair game stopped at either end is worth
the same however it is played, so the value of a capital is that capital over
the goal and every stake is exactly as good as every other. Whatever a solver
draws there is its own arithmetic and nothing about gambling.

`docs/algorithms.md` has the tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.dp import policy_iteration, value_iteration
from rel.envs.gambler import Gambler
from rel.rng import Rng
from rel.ui.table import table

#: The coins to run. A fair one is the control and has to be in the list, so
#: `main` puts it back if a caller leaves it out.
COINS: tuple[float, ...] = (0.25, 0.4, 0.5, 0.55)

GOAL = 100
LOOSE = 1e-6
TIGHT = 1e-9


def built(goal: int, heads: float) -> Gambler:
    return Gambler(Rng(1).stream("env"), goal=goal, heads=heads)


def stakes(env: Gambler, policy: tuple[int, ...]) -> list[int]:
    """What this policy really stakes at each capital that can be held."""
    return [env.stake(capital, policy[capital]) for capital in playable(env)]


def playable(env: Gambler) -> range:
    """The capitals a gambler can hold, which leaves out the two endings."""
    return range(1, env.goal)


def apart(first: list[int], second: list[int]) -> int:
    """How many capitals the two policies stake differently at."""
    return sum(one != other for one, other in zip(first, second, strict=True))


def worths_at(
    env: Gambler, values: tuple[float, ...], capital: int
) -> dict[int, float]:
    """What each stake at this capital is worth, as the sweep sees it.

    Keyed by the stake rather than by the action. Several actions clip to the
    same stake at a small capital, and counting those as different choices
    would say a capital had more to choose between than it has.
    """
    worths: dict[int, float] = {}
    for action in range(env.action_space.n):
        staked = env.stake(capital, action)
        up, down = capital + staked, capital - staked
        won = 1.0 if up == env.goal else values[up]
        worths[staked] = env.heads * won + (1.0 - env.heads) * values[down]
    return worths


def widest(env: Gambler, values: tuple[float, ...]) -> float:
    """The largest gap between the best stake and the worst, over all capitals.

    Zero would mean every stake is exactly as good everywhere, which is what a
    fair coin makes true in arithmetic. What a sweep reports instead is the
    size of its own rounding, and printing it lets a reader see which of the
    two they are looking at.
    """
    return max(
        max(worths_at(env, values, capital).values())
        - min(worths_at(env, values, capital).values())
        for capital in playable(env)
    )


def decided(env: Gambler, values: tuple[float, ...], margin: float) -> int:
    """How many capitals have one best stake by more than this margin.

    A capital whose best two stakes are worth the same to within the sweep's
    own tolerance has no best stake that the problem chose. Which one a solver
    reports there is its arithmetic, and this counts the capitals where that
    is not so.
    """
    count = 0
    for capital in playable(env):
        ranked = sorted(worths_at(env, values, capital).values(), reverse=True)
        if len(ranked) > 1 and ranked[0] - ranked[1] > margin:
            count += 1
    return count


def closed_form_section(goal: int) -> None:
    print(
        "A fair game stopped at either end is worth the same however it is "
        "played,\nso the value of a capital is that capital over the goal. "
        "That is arithmetic\nrather than another sweep, and it is what says "
        "the sweep is right.\n"
    )
    env = built(goal, 0.5)
    solved = value_iteration(env, discount=1.0, tolerance=TIGHT)
    truth = env.true_values()

    shown = [capital for capital in (1, goal // 4, goal // 2, goal - 1)]
    rows = [
        [
            str(capital),
            f"{truth[capital]:.6f}",
            f"{solved.values[capital]:.6f}",
        ]
        for capital in shown
    ]
    for line in table(
        ["capital", "closed form", "swept"], rows, align=["right", "right", "right"]
    ):
        print(f"  {line}")

    worst = max(
        abs(solved.values[capital] - truth[capital]) for capital in playable(env)
    )
    print(f"\n  Worst gap over all {goal - 1} capitals: {worst:.3g}")


def picture_section(coins: tuple[float, ...], goal: int) -> None:
    print(
        f"\n\nHow much of the stake picture survives a change that cannot "
        f"change the\nanswer. Tolerances {LOOSE:g} and {TIGHT:g}, and the two "
        f"solvers at {TIGHT:g}.\nA goal of {goal}, so there are {goal - 1} "
        f"capitals to stake at.\n"
    )

    rows = []
    for heads in coins:
        env = built(goal, heads)
        loose = value_iteration(env, discount=1.0, tolerance=LOOSE)
        tight = value_iteration(env, discount=1.0, tolerance=TIGHT)
        other = policy_iteration(env, discount=1.0, tolerance=TIGHT)

        rows.append(
            [
                f"{heads:g}",
                f"{tight.start_value:.6f}",
                f"{widest(env, tight.values):.3g}",
                f"{decided(env, tight.values, TIGHT)}",
                str(apart(stakes(env, tight.policy), stakes(env, loose.policy))),
                str(apart(stakes(env, tight.policy), stakes(env, other.policy))),
            ]
        )

    for line in table(
        [
            "heads",
            "value at the start",
            "widest gap between stakes",
            "capitals with one best stake",
            "moved by the tolerance",
            "moved by the solver",
        ],
        rows,
        align=["right"] * 6,
    ):
        print(f"  {line}")

    print(
        "\n  The values agree to the tolerance in every row. Only the stakes "
        "move,\n  and a stake that moves is one the problem never decided."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", type=float, nargs="+", default=list(COINS))
    parser.add_argument("--goal", type=int, default=GOAL)
    args = parser.parse_args()

    # The fair coin is the control, so it goes in whether or not it was asked
    # for. A run without it has no row that is known to be undetermined, and
    # every other row would then be read against nothing.
    coins = tuple(sorted({*args.coins, 0.5}))

    started = time.perf_counter()
    closed_form_section(args.goal)
    picture_section(coins, args.goal)
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
