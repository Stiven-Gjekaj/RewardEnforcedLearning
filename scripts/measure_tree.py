#!/usr/bin/env python3
"""What a buffer that draws in the log of its size costs, and whether it draws
the same places a buffer that scans does.

    python scripts/measure_tree.py
    python scripts/measure_tree.py --sizes 256 1024 4096
    python scripts/measure_tree.py --skip-agent

`rel/agents/replay.py` draws by priority in two ways. The scan adds the whole
buffer up once for each batch and binary-searches the running totals. The tree
keeps those totals between batches in `rel/agents/sums.py` and mends the ones
a changed weight touches, so a draw walks down from the root and a change
walks up from a leaf.

The scan costs `n` per batch and nothing per change. The tree costs `log2(n)`
per draw and `log2(n)` per change. Which is cheaper is a question about `n`,
and the second section runs the ladder rather than arguing about it.

## Why the same distribution can still be a different draw

Both structures add the same weights, and neither adds them in the same order.
A scan accumulates left to right, so the running total before place `k` is one
chain of `k` roundings. A tree adds in pairs, so the same total is assembled
from about `log2(k)` subtotals, each rounded on its own.

Where the two totals differ, a target between them belongs to place `k` by one
structure and to place `k - 1` by the other. The draw is then a different draw
from the same random number, which would move every digest this project has
recorded.

## Counting that exactly rather than sampling for it

The first section does not draw at all. For every boundary it finds the
smallest double the tree sends to the place above it, which is a search over
the bit patterns of a float and lands on the exact boundary. The distance from
there to the running total the scan would compare against is the width of the
band where the two disagree.

Adding those widths and dividing by the total gives the chance that one
uniform draw disagrees. That is a probability of about one in a million
million, which no amount of sampling would ever have shown, and it is exact.

`docs/algorithms.md` has the tables.
"""

from __future__ import annotations

import argparse
import itertools
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.base import Transition
from rel.agents.replay import Replay
from rel.agents.sums import Sums
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import digest_of, train
from rel.ui.table import table

#: The buffer sizes the exact count runs over. Each boundary costs a search
#: over the bit patterns of a float, so this ladder stops where it stops for
#: time rather than because the answer does anything new past it.
SIZES: tuple[int, ...] = (256, 1024, 4096, 8192)

#: The buffer sizes the cost ladder runs over. 2000 is the default buffer of
#: `deep-q`, so the crossover matters either side of it.
COSTS: tuple[int, ...] = (128, 512, 2000, 8192, 32768, 131072)

#: The range the weights are drawn from. The floor is `FLOOR` in the buffer,
#: which is the smallest priority a step can be left with, and the ceiling is
#: about the largest error a network makes early on.
SMALLEST = 1e-6
LARGEST = 5.0

PRIORITY = 0.6
WEIGHTING = 0.4
BATCH = 8

#: How much work each cost reading averages over, as batches times buffer
#: size. A scan of a hundred thousand is four milliseconds, so a fixed number
#: of batches would make the top of the ladder the whole running time.
WORK = 800_000
LEAST_BATCHES = 200

AGENT_ENV = "cartpole"
AGENT_EPISODES = 600
AGENT_SEED = 1

#: The buffers the agent section runs at. The first is the default of `deep-q`
#: and the second is where the cost ladder says the tree is clearly ahead, so
#: the pair says how much of the saving an agent actually collects.
AGENT_BUFFERS: tuple[int, ...] = (2000, 32768)


def as_bits(value: float) -> int:
    """The bit pattern of a double, read as a whole number.

    Doubles at or above zero compare in the same order as their patterns do,
    which is what makes a search over the patterns a search over the values.
    """
    return int(struct.unpack("<q", struct.pack("<d", value))[0])


def as_float(pattern: int) -> float:
    return float(struct.unpack("<d", struct.pack("<q", pattern))[0])


def weights_of(size: int, seed: int) -> list[float]:
    rng = Rng(seed)
    return [rng.uniform(SMALLEST, LARGEST) for _ in range(size)]


def tree_of(weights: list[float]) -> Sums:
    tree = Sums(len(weights))
    for place, weight in enumerate(weights):
        tree[place] = weight
    return tree


def first_target_above(tree: Sums, place: int, total: float) -> float:
    """The smallest target the tree sends to `place` or past it.

    `Sums.find` never goes backwards as the target rises, so the boundary can
    be found by halving the range of bit patterns between nothing and the
    total. Sixty-four halvings land on one pattern, which is the boundary
    itself and not an estimate of it.
    """
    low, high = as_bits(0.0), as_bits(total)
    while low < high:
        middle = (low + high) // 2
        if tree.find(as_float(middle)) >= place:
            high = middle
        else:
            low = middle + 1
    return as_float(low)


def disagreement(size: int, seed: int) -> tuple[float, int, float, float]:
    """How wide the disagreement is, in four numbers.

    The share is the chance that one uniform draw lands where the two
    structures differ. The widest gap comes twice: in the last places of a
    double, which says how many roundings piled up, and as a number, which is
    what a place has to be narrower than for two boundaries to cross one
    target. The fourth number is the narrowest place there is.
    """
    weights = weights_of(size, seed)
    running = list(itertools.accumulate(weights))
    total = running[-1]
    tree = tree_of(weights)

    band = 0.0
    widest = 0.0
    places = 0
    for place in range(1, size):
        scan_says = running[place - 1]
        tree_says = first_target_above(tree, place, total)
        band += abs(tree_says - scan_says)
        widest = max(widest, abs(tree_says - scan_says))
        places = max(places, abs(as_bits(tree_says) - as_bits(scan_says)))

    return band / total, places, widest, min(weights)


def agreement_section(sizes: tuple[int, ...], seed: int) -> None:
    print(
        "\nWhere a tree draw and a scan draw part company.\n"
        f"Weights drawn evenly from {SMALLEST:g} to {LARGEST:g}, seed {seed}."
        "\nThe share is the chance that one uniform draw lands on a target"
        " the two\nread differently. Nothing here is sampled.\n"
    )

    rows = []
    for size in sizes:
        share, places, widest, narrowest = disagreement(size, seed)
        rows.append(
            [
                f"{size}",
                f"{share:.2e}",
                f"1 in {1.0 / share:.1e}" if share > 0.0 else "never",
                f"{places}",
                f"{widest:.2e}",
                f"{narrowest / widest:.1e}",
            ]
        )

    for line in table(
        [
            "buffer",
            "share that disagree",
            "one draw in",
            "widest gap, last places",
            "widest gap",
            "narrowest place over it",
        ],
        rows,
        align=["right"] * 6,
    ):
        print(f"  {line}")

    print(
        "\n  The last column is why a disagreement can only be about"
        " neighbouring\n  places. Two boundaries cross one target only where a"
        " place is narrower\n  than the gap, and the narrowest place is"
        " millions of gaps wide."
    )


def step_at(number: int) -> Transition[int, int]:
    return Transition(number, 0, float(number), number + 1, False, False)


def one_cost(size: int, tree: bool, batches: int, batch: int) -> float:
    """Microseconds for one update: a batch drawn, then its errors put back."""
    buffer: Replay[int] = Replay(
        Rng(1), size, priority=PRIORITY, weighting=WEIGHTING, tree=tree
    )
    for number in range(size):
        buffer.add(step_at(number))

    errors = Rng(2)
    measured = [errors.uniform(0.0, LARGEST) for _ in range(batch)]

    started = time.perf_counter()
    for _ in range(batches):
        drawn = buffer.sample(batch)
        buffer.reprioritise(drawn.places, measured)
    return (time.perf_counter() - started) / batches * 1e6


def cost_section(sizes: tuple[int, ...], batch: int, work: int) -> None:
    print(
        f"\n\nWhat one update costs, as a batch of {batch} drawn and put back."
        f"\nEach reading averages over at least {LEAST_BATCHES} updates, and"
        " over more\nwhen the buffer is small enough for that to be quick.\n"
    )

    rows = []
    for size in sizes:
        batches = max(LEAST_BATCHES, work // size)
        scan = one_cost(size, False, batches, batch)
        tree = one_cost(size, True, batches, batch)
        rows.append(
            [
                f"{size}",
                f"{batches}",
                f"{scan:.1f}",
                f"{tree:.1f}",
                f"{scan / tree:.2f}",
            ]
        )

    for line in table(
        ["buffer", "updates timed", "scan, us", "tree, us", "scan over tree"],
        rows,
        align=["right"] * 5,
    ):
        print(f"  {line}")


def agent_section(
    episodes: int, seed: int, env_name: str, buffers: tuple[int, ...]
) -> None:
    """Whether the choice changes what an agent learns, and what it saves.

    A draw is a small part of an update: the batch it draws is then run
    forward and backward through a network. So the saving here is smaller than
    the saving on the draw alone, and that is the number worth having, because
    it is the one an agent actually collects.

    The buffers are the default and one where the ladder above says the tree
    is clearly ahead, because those two answer different questions. The first
    says whether the default should change and the second says what the tree
    is for.
    """
    print(
        f"\n\nThe same choice inside an agent. deep-q on {env_name},"
        f" {episodes} episodes, seed {seed},\npriority {PRIORITY:g}."
        " The digest is over everything the agent did.\n"
    )

    rows = []
    for size in buffers:
        seen: list[tuple[str, str | None]] = []
        took: list[float] = []
        for tree in (False, True):
            root = Rng(seed)
            env = ENVIRONMENTS.make(env_name, root.stream("env"))
            agent = AGENTS.make(
                "deep-q",
                root.stream("agent"),
                env,
                priority=PRIORITY,
                tree=tree,
                replay=size,
            )
            started = time.perf_counter()
            record = train(env, agent, episodes)
            took.append(time.perf_counter() - started)
            seen.append((record.digest.hexdigest(), digest_of(agent)))

        rows.append(
            [
                f"{size}",
                f"{took[0]:.1f}",
                f"{took[1]:.1f}",
                f"{took[0] / took[1]:.2f}",
                "yes" if seen[0] == seen[1] else "NO",
                seen[0][0][:16],
            ]
        )

    for line in table(
        [
            "buffer",
            "scan, seconds",
            "tree, seconds",
            "scan over tree",
            "same digests",
            "digest, the path",
        ],
        rows,
        align=["right"] * 4 + ["right", "right"],
    ):
        print(f"  {line}")

    print(
        "\n  Where the digests agree digit for digit the tree is a cost and"
        " not a\n  behaviour, which is the only reason it can be turned on"
        " without\n  rerunning everything this project has written down."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--costs", type=int, nargs="+", default=list(COSTS))
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--work", type=int, default=WORK)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--env", default=AGENT_ENV)
    parser.add_argument("--episodes", type=int, default=AGENT_EPISODES)
    parser.add_argument(
        "--agent-buffers", type=int, nargs="+", default=list(AGENT_BUFFERS)
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="run only the sections about the buffer itself",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    agreement_section(tuple(args.sizes), args.seed)
    cost_section(tuple(args.costs), args.batch, args.work)
    if not args.skip_agent:
        agent_section(args.episodes, AGENT_SEED, args.env, tuple(args.agent_buffers))

    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
