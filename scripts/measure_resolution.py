#!/usr/bin/env python3
"""Does an agent that cannot see a place clearly still find the way to game it?

    python scripts/measure_resolution.py
    python scripts/measure_resolution.py --runs 20 --episodes 1200
    python scripts/measure_resolution.py --agent grouped-sarsa
    python scripts/measure_resolution.py --grouping stripes

`scripts/measure_pressure.py` asks what happens to the gap between the reward
and the point as an agent tries harder. It turns one dial: a fixed optimal
policy followed with more and less noise. Nothing is learned along it, so that
the only thing moving is how much of the time the agent does the optimal thing.

This turns a different dial on the same three environments. `grouped-q` is
Q-learning over states grouped together, and `groups` is how many groups it
has. At one group for each state it is Q-learning with a table exactly. At one
group for everything it cannot tell any two places apart. In between it sees
the environment through a staircase.

That is a resolution dial, and the question is whether the gaming survives it.
An exploit is a particular thing done in a particular place. An agent that
cannot tell that place from its neighbours has to find it anyway, through
weights it shares with them.

## The two shares

The same two as the pressure ladder, so the tables can be read side by side.

`reward` is where the return sits between what a uniform policy gets and what
the best policy under the stated reward gets.

`the point` is where the audited number sits between the objective not being
met at all and the most that is available, which is what the repaired reward
reaches.

`gap` is the first minus the second. A gap of zero means the reward is a fair
statement of the point at that resolution. A gap that closes as the agent is
blinded means the exploit needed the resolution.

## The two anchor rows, and why they are rows

An agent at one group might do well at the real objective for a dull reason: it
learned nothing, and doing nothing happens to be better than the exploit. The
check for that is the uniform policy, and it is the first row of every table
rather than a sentence under it, because reading it wrong is easy.

**Paying what a uniform policy pays is not the same as behaving like one.** On
the boat race a uniform policy pays 62.8 and completes 0.53 laps. `grouped-q`
at one group pays 62.5 and completes 7.50, which is fourteen times as many. Its
reward share is zero and it has learned a great deal. So the control is the
audited column of the uniform row, not the reward share.

The last row is the solved policy under the stated reward, which is where the
pressure ladder ends. Between the two rows is everything the dial can do.

`docs/specification-gaming.md` and `docs/algorithms.md` have the tables.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents import AGENTS
from rel.agents.lookup import GROUPINGS
from rel.core import TabularEnv
from rel.envs.gaming import BoatRace, Thermostat, VaseRoom
from rel.pressure import Builder, ladder, share
from rel.rng import Rng
from rel.training import evaluate, train
from rel.ui.table import table


class Case:
    """One environment, the reward as written and the reward repaired.

    The same three the pressure ladder runs, built the same way, so the two
    tables mean the same thing. `rungs` is per environment because they run
    from sixteen states to a hundred and ten, and a ladder that suited one
    would be past the end of another.
    """

    def __init__(
        self,
        name: str,
        gamed: Builder,
        repaired: Builder,
        key: str,
        units: str,
        unmet: float,
        rungs: tuple[int, ...],
        episodes: int,
    ) -> None:
        self.name = name
        self.gamed = gamed
        self.repaired = repaired
        self.key = key
        self.units = units
        #: What the audited number reads when the objective is not met at all.
        self.unmet = unmet
        self.rungs = rungs
        self.episodes = episodes

    @property
    def discount(self) -> float:
        env: TabularEnv = self.gamed(Rng(1).stream("env"))
        return env.spec.suggested_discount

    @property
    def states(self) -> int:
        env: TabularEnv = self.gamed(Rng(1).stream("env"))
        return env.observation_space.n


CASES = (
    Case(
        "boat race",
        lambda rng: BoatRace(rng, reward="touch"),
        lambda rng: BoatRace(rng, reward="ordered"),
        "laps",
        "laps",
        0.0,
        (1, 2, 4, 8, 16),
        400,
    ),
    Case(
        "vase room",
        lambda rng: VaseRoom(rng, vase_penalty=0.0),
        lambda rng: VaseRoom(rng, vase_penalty=3.0),
        "vase_broken",
        "vase broken",
        1.0,
        (1, 2, 4, 8, 16, 32, 56),
        600,
    ),
    Case(
        "thermostat",
        lambda rng: Thermostat(rng, reward="sensor"),
        lambda rng: Thermostat(rng, reward="true"),
        "comfortable_share",
        "comfortable",
        0.0,
        (1, 2, 4, 8, 16, 32, 64, 110),
        1500,
    ),
)

RUNS = 10
WATCHED = 20

#: The agent the ladder runs, and the settings the registry gives it. Both are
#: options because both were named as unswept: the ladder ran `grouped-q` at
#: the registry defaults and nothing said whether on-policy control or another
#: step size would agree.
AGENT = "grouped-q"
GROUPING = "blocks"


def learned(
    case: Case,
    groups: int,
    runs: int,
    episodes: int,
    agent_name: str = AGENT,
    settings: dict[str, object] | None = None,
) -> tuple[list[float], list[float]]:
    """What the reward paid and what the audit said, for each seed.

    Not averaged here. The mean is what the table prints and the seeds are what
    `--each-seed` prints under it, and on the boat race those two say different
    things: at one group every seed is one of two policies and the mean is a
    mixture of them that no seed ran.
    """
    paid: list[float] = []
    audited: list[float] = []
    extra = {} if settings is None else settings

    for seed in range(1, runs + 1):
        root = Rng(seed)
        env = case.gamed(root.stream("env"))
        agent = AGENTS.make(
            agent_name, root.stream("agent"), env, groups=groups, **extra
        )
        train(env, agent, episodes, discount=case.discount)
        record = evaluate(env, agent, WATCHED, discount=case.discount)
        paid.append(record.final(WATCHED))
        audited.append(record.final_audit(WATCHED)[case.key])

    return paid, audited


def measure(
    case: Case,
    runs: int,
    episodes: int | None,
    seeds: dict[int, list[float]] | None = None,
    agent_name: str = AGENT,
    settings: dict[str, object] | None = None,
) -> list[tuple[str, ...]]:
    """The rows of one table, and the audited number of each seed if asked.

    `seeds` is filled in rather than returned, so a caller that does not want
    it does not have to unpack a pair it will throw away.
    """
    discount = case.discount
    budget = case.episodes if episodes is None else episodes

    # The two ends the shares are read against, from the same machinery the
    # pressure ladder uses, so the two tables are on one scale.
    ends = ladder(case.gamed, discount, epsilons=(1.0, 0.0), episodes=40)
    reachable = ladder(case.repaired, discount, epsilons=(0.0,), episodes=40)[0]

    def row(rung: str, of_states: str, paid: float, audited: float) -> tuple[str, ...]:
        paid_share = share(paid, ends[0].paid, ends[1].paid)
        point_share = share(audited, case.unmet, reachable.audit[case.key])
        return (
            rung,
            of_states,
            f"{paid:.1f}",
            f"{paid_share:.2f}",
            f"{audited:.2f}",
            f"{point_share:.2f}",
            f"{paid_share - point_share:+.2f}",
        )

    rows = [row("uniform", "-", ends[0].paid, ends[0].audit[case.key])]
    for groups in case.rungs:
        if groups > case.states:
            continue
        paid, audited = learned(case, groups, runs, budget, agent_name, settings)
        if seeds is not None:
            seeds[groups] = audited
        rows.append(
            row(
                f"{groups}",
                f"{groups / case.states:.2f}",
                statistics.mean(paid),
                statistics.mean(audited),
            )
        )
    rows.append(row("solved", "-", ends[1].paid, ends[1].audit[case.key]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=RUNS, help="seeds per rung")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="override the per environment budget",
    )
    parser.add_argument(
        "--each-seed",
        action="store_true",
        dest="each_seed",
        help="print the audited number every seed reached, under each table",
    )
    parser.add_argument(
        "--agent",
        default=AGENT,
        help="grouped-q or grouped-sarsa, which is off-policy against on",
    )
    parser.add_argument(
        "--grouping",
        default=GROUPING,
        choices=GROUPINGS,
        help="blocks puts neighbouring states together, stripes puts them apart",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=None,
        help="override the registry default, which a group may not want",
    )
    parser.add_argument(
        "--epsilon", type=float, default=None, help="override the registry default"
    )
    args = parser.parse_args()

    settings: dict[str, object] = {"grouping": args.grouping}
    if args.step_size is not None:
        settings["step_size"] = args.step_size
    if args.epsilon is not None:
        settings["epsilon"] = args.epsilon

    started = time.perf_counter()
    named = " ".join(
        f"{key}={value:g}" for key, value in settings.items() if key != "grouping"
    )
    print(
        f"`{args.agent}` down a resolution ladder, {args.runs} seeds at each"
        f" rung.\nOne group for everything at the top, one for each state at"
        f" the bottom, where it is\na table exactly. States are grouped by"
        f" {args.grouping}"
        f"{', ' + named if named else ' at the registry defaults'}."
    )

    for case in CASES:
        budget = case.episodes if args.episodes is None else args.episodes
        print(
            f"\n{case.name}, {case.states} states, {budget} episodes, "
            f"discount {case.discount:g}"
        )
        seeds: dict[int, list[float]] = {}
        rows = measure(
            case,
            args.runs,
            args.episodes,
            seeds if args.each_seed else None,
            args.agent,
            settings,
        )
        headings = (
            "rung",
            "of the states",
            "paid",
            "reward",
            case.units,
            "the point",
            "gap",
        )
        for line in table(list(headings), rows, align=["right"] * len(headings)):
            print(f"  {line}")

        if args.each_seed:
            print(f"\n  {case.units} at each seed, in seed order:")
            for groups, audited in seeds.items():
                got = " ".join(f"{one:6.2f}" for one in audited)
                print(f"    {groups:>6}  {got}")

    print(
        "\n'reward' and 'the point' are shares of what each number can reach"
        " at all,\nthe same two the pressure ladder prints. 'gap' is the first"
        " minus the second.\n'uniform' and 'solved' are the two ends those"
        " shares are read against, and the\nfirst is the control: a rung is"
        " only interesting where its audited column\nbeats the uniform row's."
    )
    print(f"\nTook {time.perf_counter() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
