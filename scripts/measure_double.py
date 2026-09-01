#!/usr/bin/env python3
"""How much the maximum in Q-learning overstates, and what splitting the
choosing from the valuing takes off.

    python scripts/measure_double.py
    python scripts/measure_double.py --episodes 2000 --runs 30
    python scripts/measure_double.py --skip-task

Q-learning backs up `max Q(s')`. That is the largest of several estimates, each
of which carries error in both directions, and the largest of several noisy
numbers is above the largest true number. The error does not average out with
more data: it comes from the choosing, not from any one estimate.

Double estimation splits the two. One estimator names the best action and the
other says what it is worth, so neither grades its own choice. `double-q` does
it with a second table. `deep-q --set double=true` does it with the target
network it already keeps.

## The problem, and the number to read

`bias` is Sutton and Barto's example 6.7. Going right ends the episode with
nothing. Going left leads to a gamble whose mean is -0.1 and whose spread is 1.
Value iteration says right, so an agent that goes left is wrong by a tenth and
the answer is not in doubt.

The number read here is the share of episodes that went left, which the
environment reports as an audit. The return would say far less: an episode that
went left is worth -1.1 or +0.9 and an episode that went right is worth
nothing, so a mean return mixes the mistake with the noise that caused it.

**An agent that has learned the answer still goes left sometimes,** because
epsilon-greedy explores. That floor is `epsilon / actions` and the script works
it out rather than quoting it. Anything above the floor is the bias.

## Why the late column matters as much as the early one

The step size is a constant here, as it is everywhere in this project. A
constant step size tracks rather than converges: each estimate of the gamble
settles into a band around -0.1 whose width is proportional to the step size
and stays there. So the question of whether the bias is a start-up cost that a
long run pays off is a real one, and the third section answers it by running
the ladder rather than arguing about it.

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
from rel.compare import compare
from rel.envs import ENVIRONMENTS
from rel.envs.bias import MaximisationBias
from rel.rng import Rng
from rel.training import evaluate, train
from rel.ui.table import table

#: The two tabular agents, as (name, extra settings). The first is the one
#: everything else is compared against.
TABLES: tuple[tuple[str, dict[str, object]], ...] = (
    ("q-learning", {}),
    ("double-q", {}),
)

#: The same split on a network, as (label, settings). One agent, one setting.
NETWORKS: tuple[tuple[str, dict[str, object]], ...] = (
    ("deep-q", {"double": False}),
    ("deep-q double", {"double": True}),
)

ENVIRONMENT = "bias"
EPISODES = 1000
RUNS = 30
EPSILON = 0.1
STEP_SIZE = 0.1
NETWORK_STEP = 0.01

#: How many episodes at each end of a run the two columns average over by
#: default. The early one is short because the effect is largest before
#: anything is known.
EARLY = 50
LATE = 200

TASK = "cartpole"
TASK_EPISODES = 150
TASK_RUNS = 10

#: The budgets the settling section runs, and how many seeds each gets. The
#: ladder doubles so that a bias fading like one over the episodes and a bias
#: flattening out look different rather than similar.
LADDER: tuple[int, ...] = (500, 1000, 2000, 4000, 8000)
LADDER_RUNS = 20


def floor_share(env: MaximisationBias, epsilon: float) -> float:
    """How often an agent that has learned the answer still goes left.

    Epsilon-greedy explores with probability `epsilon` and then draws evenly,
    so one action in `actions` of those draws is the wrong turn. Worked out
    from the environment rather than quoted, because the environment's action
    count is a setting.
    """
    return epsilon / env.action_space.n


def one_agent(
    name: str,
    settings: dict[str, object],
    runs: int,
    episodes: int,
    env_name: str,
    epsilon: float,
    ends: tuple[int, int],
) -> tuple[list[float], list[float]]:
    """The left share of each seed, early in the run and late in it."""
    first, last = ends
    early: list[float] = []
    late: list[float] = []

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = ENVIRONMENTS.make(env_name, root.stream("env"))
        agent = AGENTS.make(
            name.split()[0], root.stream("agent"), env, epsilon=epsilon, **settings
        )
        record = train(env, agent, episodes, discount=1.0)
        went = record.audits["went_left"]
        early.append(statistics.mean(went[:first]))
        late.append(statistics.mean(went[-last:]))

    return early, late


def bias_section(
    heading: str,
    agents: tuple[tuple[str, dict[str, object]], ...],
    env_name: str,
    runs: int,
    episodes: int,
    epsilon: float,
    rng: Rng,
    ends: tuple[int, int] = (EARLY, LATE),
) -> None:
    first, last = ends
    if episodes < first + last:
        # Two windows that overlap are two views of the same episodes, and a
        # table that put them side by side would say the bias faded when the
        # two columns were the same number twice.
        raise SystemExit(
            f"{episodes} episodes cannot hold a first {first} and a last "
            f"{last} without overlapping them."
        )

    env = ENVIRONMENTS.make(env_name, Rng(1).stream("env"))
    assert isinstance(env, MaximisationBias)
    floor = floor_share(env, epsilon)

    print(
        f"\n{heading}\n{env_name}, {runs} seeds, {episodes} episodes, epsilon"
        f" {epsilon:g}.\nThe share of episodes that went the wrong way, over"
        f" the first {first} and the last {last}.\nAn agent that has learned"
        f" the answer still goes left {floor:.3f} of the time.\n"
    )

    got: dict[str, tuple[list[float], list[float]]] = {}
    for name, settings in agents:
        got[name] = one_agent(name, settings, runs, episodes, env_name, epsilon, ends)

    against = agents[0][0]
    rows = []
    for name, _ in agents:
        early, late = got[name]
        # The median is beside the mean because on the late window the two say
        # different things: a seed whose estimates never recover goes left
        # about a fifth of the time for ever, and a few of those move a mean
        # further than the whole difference between the two agents.
        row = [
            name,
            f"{statistics.mean(early):.3f}",
            f"{statistics.median(late):.3f}",
            f"{statistics.mean(late):.3f}",
            f"{statistics.mean(late) / floor:.1f}",
        ]
        if name == against:
            row += ["-", "-", "-"]
        else:
            answer = compare(late, got[against][1], rng)
            row += [
                f"{answer.difference:+.3f}",
                f"[{answer.low:+.3f}, {answer.high:+.3f}]",
                f"{answer.p_value:.4f}",
            ]
        rows.append(row)

    for line in table(
        [
            "agent",
            f"left, first {first}",
            f"last {last}, median",
            "mean",
            "mean over the floor",
            f"late minus {against}",
            "95 percent interval",
            "p",
        ],
        rows,
        align=["left"] + ["right"] * 7,
    ):
        print(f"  {line}")

    floor_p = 2.0 / 2.0**runs
    if floor_p > 0.05:
        print(
            f"\n  {runs} seeds cannot give a p below {floor_p:.4f} however"
            f" large the difference is."
        )


def settling_section(
    agents: tuple[tuple[str, dict[str, object]], ...],
    env_name: str,
    runs: int,
    ladder: tuple[int, ...],
    epsilon: float,
    window: int,
) -> None:
    """Whether a longer run pays the bias off, or whether it flattens out.

    The step size is a constant, so each estimate of the gamble settles into a
    band around its mean rather than onto it. Whether the maximum over ten of
    those bands stays above zero for ever is not something to argue about, so
    this runs the budgets and prints what happened.

    The median is here beside the mean because at the long budgets they say
    different things. A seed whose estimates never recover goes left about a
    fifth of the time for ever, and one such seed in twenty moves the mean by
    more than the whole difference between the two agents. The mean is then a
    failure count wearing the clothes of a performance number.
    """
    env = ENVIRONMENTS.make(env_name, Rng(1).stream("env"))
    assert isinstance(env, MaximisationBias)
    floor = floor_share(env, epsilon)

    print(
        f"\n\nWhether a longer run pays it off. {env_name}, {runs} seeds"
        f" each,\nthe share over the last {window} episodes of the budget."
        f" The floor is {floor:.3f}.\n"
    )

    rows = []
    for episodes in ladder:
        if episodes < window:
            continue
        row = [f"{episodes}"]
        for name, settings in agents:
            _, late = one_agent(
                name, settings, runs, episodes, env_name, epsilon, (1, window)
            )
            row += [
                f"{statistics.median(late):.4f}",
                f"{statistics.mean(late):.4f}",
            ]
        rows.append(row)

    headings = ["episodes"]
    for name, _ in agents:
        headings += [f"{name}, median", "mean"]

    for line in table(headings, rows, align=["right"] * len(headings)):
        print(f"  {line}")


def task_section(runs: int, episodes: int, rng: Rng) -> None:
    """Whether the split helps where the bias is not the whole problem.

    The bias is there on any task with a maximum in it. Whether removing it is
    worth anything where a dozen other things are also going wrong is a
    different question, and this is the one section that asks it.
    """
    print(
        f"\n\nThe same setting where the bias is not the point. {TASK},"
        f" {runs} seeds,\n{episodes} episodes, read as the return of the"
        f" greedy policy at the end.\n"
    )

    got: dict[str, list[float]] = {}
    for label, settings in NETWORKS:
        finals: list[float] = []
        for seed in range(1, runs + 1):
            root = Rng(seed)
            env = ENVIRONMENTS.make(TASK, root.stream("env"))
            agent = AGENTS.make("deep-q", root.stream("agent"), env, **settings)
            train(env, agent, episodes, discount=env.spec.suggested_discount)
            finals.append(statistics.mean(evaluate(env, agent, 5).returns))
        got[label] = finals

    against = NETWORKS[0][0]
    rows = []
    for label, _ in NETWORKS:
        row = [label, f"{statistics.mean(got[label]):.1f}"]
        if label == against:
            row += ["-", "-", "-"]
        else:
            answer = compare(got[label], got[against], rng)
            row += [
                f"{answer.difference:+.1f}",
                f"[{answer.low:+.1f}, {answer.high:+.1f}]",
                f"{answer.p_value:.4f}",
            ]
        rows.append(row)

    for line in table(
        ["agent", "return", f"minus {against}", "95 percent interval", "p"],
        rows,
        align=["left"] + ["right"] * 4,
    ):
        print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=ENVIRONMENT)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--runs", type=int, default=RUNS, help="seeds per agent")
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--early", type=int, default=EARLY)
    parser.add_argument("--late", type=int, default=LATE)
    parser.add_argument("--ladder", type=int, nargs="+", default=list(LADDER))
    parser.add_argument("--ladder-runs", type=int, default=LADDER_RUNS)
    parser.add_argument("--task-runs", type=int, default=TASK_RUNS)
    parser.add_argument("--task-episodes", type=int, default=TASK_EPISODES)
    parser.add_argument(
        "--skip-task",
        action="store_true",
        help="run only the sections on the problem the bias is about",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    rng = Rng(9).stream("compare")

    tables = tuple(
        (name, {"step_size": STEP_SIZE, **settings}) for name, settings in TABLES
    )
    ends = (args.early, args.late)
    bias_section(
        "Two tables against one.",
        tables,
        args.env,
        args.runs,
        args.episodes,
        args.epsilon,
        rng,
        ends,
    )

    networks = tuple(
        (label, {"step_size": NETWORK_STEP, **settings}) for label, settings in NETWORKS
    )
    bias_section(
        "\nThe same split on a network, using the target copy it already keeps.",
        networks,
        args.env,
        args.runs,
        args.episodes,
        args.epsilon,
        rng,
        ends,
    )

    settling_section(
        tables, args.env, args.ladder_runs, tuple(args.ladder), args.epsilon, args.late
    )

    if not args.skip_task:
        task_section(args.task_runs, args.task_episodes, rng)

    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
