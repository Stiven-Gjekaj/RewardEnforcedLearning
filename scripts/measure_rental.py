#!/usr/bin/env python3
"""What the van is worth, and the price at which it stops being worth using.

Sutton and Barto, example 4.2, is usually run once and drawn once: here is the
policy, here is the value. This asks the two questions the example sets up and
does not answer. A van that moves cars overnight costs 2 a car. What does it
buy, and how much would it have to cost before it stopped being used?

    python scripts/measure_rental.py
    python scripts/measure_rental.py --costs 0 2 10
    python scripts/measure_rental.py --capacity 10 --van 3

Both answers are arithmetic rather than runs. The environment gives its own
model, so every number here is a sweep of that model and no seed appears
anywhere.

**The control is a van that holds nothing.** Solving with `van=0` leaves one
action, which moves nothing, so the value it reaches is what the two locations
are worth with no moving at all. Everything else is read against it.

**The value is discounted, and the day is not.** A discount of 0.9 turns a
gain of `g` every day into a value of `g` over one minus the discount, so the
per day column is the gap times one tenth. That is the number a reader can
compare with the 2 a car the van charges.

`docs/algorithms.md` has the tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.dp import TOLERANCE, Solution, value_iteration
from rel.envs.rental import CarRental
from rel.rng import Rng
from rel.ui.table import table

#: The prices to try. Two is the book's, nothing is the van for free, and the
#: rest run up to a price no policy would pay.
COSTS: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0)

DISCOUNT = 0.9


def solve(capacity: int, van: int, cost: float) -> tuple[CarRental, Solution]:
    env = CarRental(Rng(1).stream("env"), capacity=capacity, van=van, move_cost=cost)
    return env, value_iteration(env, discount=DISCOUNT)


def settled(gap: float) -> str:
    """A gap, or nothing at all where the sweep cannot tell it from nothing.

    A van priced past what it is worth is never used, so the value it reaches
    is the value with no van. The two sweeps land about 1e-10 apart, which is
    the sweep's own arithmetic and not a loss the van causes, and printing it
    as a small negative number would say the van cost something.
    """
    return "0.000" if abs(gap) < TOLERANCE else f"{gap:+.3f}"


def moving_states(env: CarRental, solved: Solution) -> int:
    """How many states the best policy moves a car in.

    Read as moves rather than as action numbers. Several actions clip onto
    moving nothing at the edges of the board, so a count of actions that are
    not the middle one would count states where nothing happens.
    """
    return sum(
        1
        for state in range(env.observation_space.n)
        if env.moved(state, solved.policy[state]) != 0
    )


def largest_move(env: CarRental, solved: Solution) -> int:
    """The most cars the best policy moves in any one night."""
    return max(
        abs(env.moved(state, solved.policy[state]))
        for state in range(env.observation_space.n)
    )


def price_section(capacity: int, van: int, costs: tuple[float, ...]) -> None:
    print(
        f"Two locations holding {capacity} cars each, a van that moves up to "
        f"{van},\nand a discount of {DISCOUNT:g}. Every row is a sweep of the "
        f"model rather than a run.\n"
    )

    _, without = solve(capacity, 0, 0.0)
    floor = without.start_value

    rows = [
        [
            "no van",
            f"{floor:.3f}",
            "-",
            "-",
            "0",
            "0",
        ]
    ]
    for cost in costs:
        env, solved = solve(capacity, van, cost)
        gain = solved.start_value - floor
        rows.append(
            [
                f"{cost:g}",
                f"{solved.start_value:.3f}",
                settled(gain),
                settled(gain * (1.0 - DISCOUNT)),
                str(moving_states(env, solved)),
                str(largest_move(env, solved)),
            ]
        )

    for line in table(
        [
            "a car costs",
            "value at the start",
            "over no van",
            "a day",
            "states that move",
            "largest move",
        ],
        rows,
        align=["right"] * 6,
    ):
        print(f"  {line}")

    print(
        "\n  'states that move' counts the states whose best move is not "
        "nothing, out\n  of "
        f"{(capacity + 1) ** 2}"
        ". It falls with the price and reaches nothing when the van is\n"
        "  worth less than it charges. A gap the sweep cannot tell from "
        f"nothing at\n  its tolerance of {TOLERANCE:g} is printed as nothing."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=float, nargs="+", default=list(COSTS))
    parser.add_argument("--capacity", type=int, default=20)
    parser.add_argument("--van", type=int, default=5)
    args = parser.parse_args()

    started = time.perf_counter()
    price_section(args.capacity, args.van, tuple(args.costs))
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
