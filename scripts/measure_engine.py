#!/usr/bin/env python3
"""How fast the gradient engine is, and whether it still does the same thing.

Both halves matter and the second one is the harder to get right. An
optimisation that reassociates a sum is faster and gives different numbers, and
on a learning agent different numbers look exactly like the same agent on
another seed. The digest is what tells them apart.

    python scripts/measure_engine.py
    python scripts/measure_engine.py --episodes 60
    python scripts/measure_engine.py --agent deep-q --env cliff

The digest lines are the point. Run this before a change and after it, and if
the digests match then whatever moved was the speed and nothing else. If they
do not match, the change altered the arithmetic, and no timing below is worth
reading until that is understood.
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
from rel.nn.autograd import Tensor, linear
from rel.rng import Rng
from rel.training import digest_of, train
from rel.ui.table import table


def one_run(
    grid: str, agent_name: str, episodes: int, seed: int
) -> tuple[float, str, str]:
    """The seconds a run took, and the two digests it produced."""
    root = Rng(seed)
    env = ENVIRONMENTS.make(grid, root.stream("env"))
    discount = env.spec.suggested_discount
    agent = AGENTS.make(agent_name, root.stream("agent"), env)

    started = time.perf_counter()
    record = train(env, agent, episodes, discount=discount)
    seconds = time.perf_counter() - started

    learned = digest_of(agent)
    return seconds, record.digest.hexdigest(), "-" if learned is None else learned


def one_layer(inputs: int, outputs: int, passes: int) -> tuple[float, float]:
    """The seconds a bare layer takes, forward and backward.

    A whole agent measures the engine mixed with everything around it. This is
    the engine on its own, at the shape the agents really use, so a change that
    helps one layer and hurts the loop around it shows up as two numbers rather
    than as one that did not move.

    The backward pass is driven by calling the node's own closure rather than
    by `Tensor.backward`, which would walk the graph as well and is timed by
    the whole run above. Reaching for a private name is the price of measuring
    one operation instead of two.
    """
    rng = Rng(1).stream("weights")
    weight = Tensor(
        [rng.normal(0.0, 0.1) for _ in range(outputs * inputs)], (outputs, inputs)
    )
    bias = Tensor([0.0] * outputs, (outputs,))
    x = Tensor([rng.normal() for _ in range(inputs)])

    started = time.perf_counter()
    made = [linear(x, weight, bias) for _ in range(passes)]
    forward = time.perf_counter() - started

    started = time.perf_counter()
    for result in made:
        result.grad = [1.0] * len(result)
        assert result._backward is not None
        result._backward()
    backward = time.perf_counter() - started

    return forward, backward


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="cartpole")
    parser.add_argument("--agent", default="reinforce")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--passes", type=int, default=20_000)
    args = parser.parse_args()

    times: list[float] = []
    path = learned = ""
    for _ in range(args.repeats):
        seconds, path, learned = one_run(args.env, args.agent, args.episodes, 1)
        times.append(seconds)

    forward, backward = one_layer(4, 16, args.passes)
    wide_forward, wide_backward = one_layer(48, 16, args.passes)

    print(
        f"{args.agent} on {args.env}, {args.episodes} episodes, best of {args.repeats}."
    )
    print()

    rows = [
        ("a whole run, best", f"{min(times):.2f}s"),
        ("a whole run, median", f"{statistics.median(times):.2f}s"),
        (f"4 to 16 forward, {args.passes:,} passes", f"{forward:.2f}s"),
        (f"4 to 16 backward, {args.passes:,} passes", f"{backward:.2f}s"),
        (f"48 to 16 forward, {args.passes:,} passes", f"{wide_forward:.2f}s"),
        (f"48 to 16 backward, {args.passes:,} passes", f"{wide_backward:.2f}s"),
    ]
    for line in table(["", "time"], rows, align=["left", "right"]):
        print(f"  {line}")

    print()
    for line in table(
        ["digest", "value"],
        [("the path", path), ("what it learned", learned)],
        align=["left", "left"],
    ):
        print(f"  {line}")

    print(
        "\nRun this before a change and after it. Matching digests mean the\n"
        "change moved the speed and nothing else. Digests that differ mean the\n"
        "arithmetic moved, and no timing above is worth reading until that is\n"
        "understood: on a learning agent, different numbers look exactly like\n"
        "the same agent on another seed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
