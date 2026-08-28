#!/usr/bin/env python3
"""What importance sampling costs, and which estimator pays it.

Off-policy Monte Carlo corrects a return collected under one policy into an
estimate about another, by multiplying it by the ratio of the two at every
step. The two estimators divide that product by different things, and the
textbook difference between them is a sentence: one is unbiased and the other
is not. The difference a run notices is variance.

    python scripts/measure_importance.py
    python scripts/measure_importance.py --episodes 2000 --runs 12

Three numbers are reported for each estimator.

`value` is the exact value of the greedy policy from the model, averaged over
the seeds whose policy reaches an ending.

`estimate` is what the agent itself believes the best action at the start is
worth, and `spread` is how far the worst seed's belief sits from the mean of
them. That pair is what the two estimators really differ on. The variance
importance sampling is famous for is variance in the estimate, so measuring the
spread of the finished policy instead would be measuring a downstream
consequence of it and calling it the thing.

`stuck` counts the seeds whose greedy policy never reaches an ending at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.dp import evaluate_policy, value_iteration
from rel.agents.off_policy import ESTIMATORS, OffPolicyMonteCarlo
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.training import train
from rel.ui.table import table

#: The grids this can learn at all. The cliff walk is not one of them, and the
#: reason is in `docs/algorithms.md`: its episodes run to the step limit far
#: more often than they finish, and an episode teaches only the tail after the
#: last step the behaviour policy explored.
GRIDS = ("lake", "maze")


def measure(
    grid: str, estimator: str, episodes: int, runs: int, epsilon: float
) -> tuple[str, ...]:
    probe = ENVIRONMENTS.make(grid, Rng(1).stream("env"))
    discount = probe.spec.suggested_discount

    values: list[float] = []
    beliefs: list[float] = []
    stuck = 0

    for seed in range(1, runs + 1):
        rng = Rng(seed)
        env = ENVIRONMENTS.make(grid, rng.stream("env"))
        agent = OffPolicyMonteCarlo(
            rng.stream("agent"),
            env.action_space,
            epsilon=epsilon,
            discount=discount,
            estimator=estimator,  # type: ignore[arg-type]
        )
        train(env, agent, episodes, discount=discount)

        # What the agent believes the start is worth. This is the estimate
        # importance sampling corrects, and the one whose spread is the lesson.
        start = next(iter(env.start_states()))
        beliefs.append(max(agent.peek(start)))

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=discount)
        if report.reaches_end:
            values.append(report.start_value)
        else:
            stuck += 1

    centre = sum(beliefs) / len(beliefs)
    spread = max(abs(belief - centre) for belief in beliefs)
    value = f"{sum(values) / len(values):.3f}" if values else "-"
    return (
        estimator,
        f"{centre:.3f}",
        f"{spread:.3f}",
        value,
        str(stuck) if stuck else "",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--grids", nargs="+", default=list(GRIDS))
    args = parser.parse_args()

    print(
        f"{args.episodes} episodes, seeds 1 to {args.runs}, "
        f"a behaviour policy that explores {args.epsilon:.0%} of the time."
    )

    for grid in args.grids:
        probe = ENVIRONMENTS.make(grid, Rng(1).stream("env"))
        discount = probe.spec.suggested_discount
        best = value_iteration(probe, discount=discount).start_value
        print(f"\n{grid}, the best possible is {best:.3f}")

        rows = [
            measure(grid, estimator, args.episodes, args.runs, args.epsilon)
            for estimator in ESTIMATORS
        ]
        headings = ["estimator", "estimate", "spread", "policy value", "stuck"]
        for line in table(headings, rows, align=["left"] + ["right"] * 4):
            print(f"  {line}")

    print(
        "\n'estimate' is what the agent believes the start is worth and 'spread'\n"
        "is how far the worst seed's belief sits from the mean. That pair is the\n"
        "lesson. 'policy value' is the exact value of what it ended up doing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
