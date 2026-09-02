#!/usr/bin/env python3
"""Is reusing an episode worth as much as collecting another one?

    python scripts/measure_clipped.py
    python scripts/measure_clipped.py --runs 20
    python scripts/measure_clipped.py --skip-pendulum

`reinforce` takes one gradient step from an episode and throws it away. The
episode cost real steps of a real environment and one step of gradient is very
little to get for them. `clipped-policy` walks over the same episode `passes`
times, with a clip on how far any one pass may move a step whose probability
has already moved, and `rel/agents/policy.py` says what the clip is and why
there is no clip operation in the engine.

## The comparison that means something

Four passes over a hundred episodes is four hundred gradient steps on a
hundred episodes of environment. One pass over four hundred episodes is four
hundred gradient steps on four hundred episodes of environment. Those two are
the comparison: same learning, four times the interaction, and the question is
whether the cheap one keeps up.

The row at one pass over a hundred episodes is under both of them, because a
method that reuses an episode has to beat collecting nothing as well as beat
collecting more.

**One pass is `reinforce` with a baseline exactly.** Every ratio is one before
the policy has moved, and the gradient of `weight * exp(logp - logp)` is the
gradient of `weight * logp`. `tests/test_policy_gradient.py` holds the two
runs against each other digit for digit, so the row is a control rather than a
second agent.

## The share the clip binds on

At zero the clip is doing nothing and the agent is plain repeated gradient. Near
one it is doing everything and the passes past the first are wasted. Neither
end is a setting anybody wants and the sweep says where the middle is.

## The pendulum is here to correct a sentence

`docs/algorithms.md` says the two policy gradient agents lose to a random
policy on the pendulum, and gives their shared bootstrapped target as the
reason. `reinforce` is not bootstrapped and this runs it there.

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
from rel.agents.policy import ClippedPolicy
from rel.compare import compare
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import evaluate, train
from rel.ui.table import table

ENVIRONMENT = "cartpole"
RUNS = 10
WATCHED = 10

#: The four rows of the first section, as (label, agent, episodes, settings).
#: The first is the one the others are compared against, and it is the budget
#: the reusing row is trying to reach on a quarter of the environment.
BUDGETS: tuple[tuple[str, str, int, dict[str, object]], ...] = (
    ("reinforce, 400", "reinforce", 400, {}),
    ("clipped 1 x 400", "clipped-policy", 400, {"passes": 1}),
    ("reinforce, 100", "reinforce", 100, {}),
    ("clipped 4 x 100", "clipped-policy", 100, {"passes": 4}),
    ("clipped 4 x 400", "clipped-policy", 400, {"passes": 4}),
)

#: The clip ranges swept, at four passes. 1.0 is wide enough that a ratio has
#: to double before it binds, which is the setting that says what the clip is
#: worth by leaving it off.
RANGES: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 1.0)

#: The pass counts swept, at a clip range of 0.2.
PASSES: tuple[int, ...] = (1, 2, 4, 8)

SWEEP_EPISODES = 100
SWEEP_RUNS = 6

PENDULUM = "pendulum-levels"
PENDULUM_EPISODES = 300
PENDULUM_RUNS = 5
PENDULUM_AGENTS = ("random", "reinforce", "actor-critic", "clipped-policy")


def one_setting(
    name: str,
    episodes: int,
    runs: int,
    env_name: str,
    settings: dict[str, object],
) -> tuple[list[float], float]:
    """The greedy return of each seed, and the share the clip bound on.

    The share is nothing for an agent that has no clip, which is what makes it
    printable in the same column as one that does.
    """
    got: list[float] = []
    clipped = 0
    considered = 0

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(env_name, root.stream("env"))
        agent = AGENTS.make(name, root.stream("agent"), env, **settings)
        train(env, agent, episodes)
        got.append(statistics.mean(evaluate(env, agent, WATCHED).returns))
        if isinstance(agent, ClippedPolicy):
            clipped += agent.clipped
            considered += agent.considered

    return got, (clipped / considered if considered else 0.0)


def budget_section(
    rows: tuple[tuple[str, str, int, dict[str, object]], ...],
    runs: int,
    env_name: str,
    rng: Rng,
) -> None:
    print(
        f"Reusing an episode against collecting another one. {env_name},"
        f" {runs} seeds.\n'steps' is gradient steps and 'episodes' is what the"
        f" environment was asked for.\nThe best possible return is 500.\n"
    )

    got: dict[str, list[float]] = {}
    shares: dict[str, float] = {}
    seconds: dict[str, float] = {}
    for label, name, episodes, settings in rows:
        started = time.perf_counter()
        got[label], shares[label] = one_setting(
            name, episodes, runs, env_name, settings
        )
        seconds[label] = time.perf_counter() - started

    against = rows[0][0]
    printed = []
    for label, _, episodes, settings in rows:
        passes = int(settings.get("passes", 1))  # type: ignore[arg-type]
        row = [
            label,
            f"{passes * episodes}",
            f"{episodes}",
            f"{statistics.median(got[label]):.1f}",
            f"{statistics.mean(got[label]):.1f}",
            f"{shares[label]:.4f}",
            f"{seconds[label]:.0f}",
        ]
        if label == against:
            row += ["-", "-", "-"]
        else:
            answer = compare(got[label], got[against], rng)
            row += [
                f"{answer.difference:+.1f}",
                f"[{answer.low:+.1f}, {answer.high:+.1f}]",
                f"{answer.p_value:.4f}",
            ]
        printed.append(row)

    for line in table(
        [
            "run",
            "steps",
            "episodes",
            "median",
            "mean",
            "share clipped",
            "seconds",
            f"mean minus {against}",
            "95 percent interval",
            "p",
        ],
        printed,
        align=["left"] + ["right"] * 9,
    ):
        print(f"  {line}")

    floor = 2.0 / 2.0**runs
    if floor > 0.05:
        print(
            f"\n  {runs} seeds cannot give a p below {floor:.4f} however large"
            f" the difference is."
        )


def sweep_section(
    ranges: tuple[float, ...],
    passes: tuple[int, ...],
    runs: int,
    episodes: int,
    env_name: str,
) -> None:
    print(
        f"\n\nWhere the clip binds. {env_name}, {runs} seeds, {episodes}"
        f" episodes.\nThe range is swept at four passes and the passes at a"
        f" range of 0.2.\n"
    )

    rows = []
    for width in ranges:
        got, share = one_setting(
            "clipped-policy",
            episodes,
            runs,
            env_name,
            {"passes": 4, "clip_range": width},
        )
        rows.append(
            [
                f"{width:g}",
                "4",
                f"{share:.4f}",
                f"{statistics.median(got):.1f}",
                f"{statistics.mean(got):.1f}",
            ]
        )
    for count in passes:
        got, share = one_setting(
            "clipped-policy",
            episodes,
            runs,
            env_name,
            {"passes": count, "clip_range": 0.2},
        )
        rows.append(
            [
                "0.2",
                f"{count}",
                f"{share:.4f}",
                f"{statistics.median(got):.1f}",
                f"{statistics.mean(got):.1f}",
            ]
        )

    for line in table(
        ["clip range", "passes", "share clipped", "median", "mean"],
        rows,
        align=["right"] * 5,
    ):
        print(f"  {line}")


def pendulum_section(
    names: tuple[str, ...], runs: int, episodes: int, env_name: str
) -> None:
    """Whether the shared cause the page names is the shared cause.

    The page says the two policy gradient agents lose to a random policy here
    because both bootstrap their target from a value network. `reinforce` does
    not bootstrap, and neither does `clipped-policy`, and this runs them.
    """
    print(
        f"\n\nThe problem no policy gradient agent here learns. {env_name},"
        f" {runs} seeds, {episodes} episodes.\n"
    )

    rows = []
    for name in names:
        got, _ = one_setting(name, episodes, runs, env_name, {})
        rows.append(
            [
                name,
                f"{statistics.median(got):.1f}",
                f"{statistics.mean(got):.1f}",
            ]
        )

    for line in table(
        ["agent", "median", "mean"], rows, align=["left", "right", "right"]
    ):
        print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=ENVIRONMENT)
    parser.add_argument("--runs", type=int, default=RUNS, help="seeds per row")
    parser.add_argument("--sweep-runs", type=int, default=SWEEP_RUNS)
    parser.add_argument("--sweep-episodes", type=int, default=SWEEP_EPISODES)
    parser.add_argument("--ranges", type=float, nargs="+", default=list(RANGES))
    parser.add_argument("--passes", type=int, nargs="+", default=list(PASSES))
    parser.add_argument("--pendulum-runs", type=int, default=PENDULUM_RUNS)
    parser.add_argument("--pendulum-episodes", type=int, default=PENDULUM_EPISODES)
    parser.add_argument(
        "--skip-pendulum",
        action="store_true",
        dest="skip_pendulum",
        help="run only the sections about the reuse itself",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    rng = Rng(11).stream("compare")
    budget_section(BUDGETS, args.runs, args.env, rng)
    sweep_section(
        tuple(args.ranges),
        tuple(args.passes),
        args.sweep_runs,
        args.sweep_episodes,
        args.env,
    )
    if not args.skip_pendulum:
        pendulum_section(
            PENDULUM_AGENTS, args.pendulum_runs, args.pendulum_episodes, PENDULUM
        )

    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
