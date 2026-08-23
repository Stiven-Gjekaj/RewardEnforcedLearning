"""The demonstration this project is named for.

Three environments. In each one the reward is a reasonable thing to have
written down, and the best possible policy under it is not what anybody wanted.
Then the same environment with the reward repaired, so that the fault is shown
to be in the specification rather than in the agent.

The first number for each row comes from `value_iteration`, which reads the
model and returns the best policy there is. That matters: it means the gaming
is not something an agent stumbled into and could be trained out of. It is the
answer to the question that was asked.

The second number comes from Q-learning, so that the same thing can be seen
happening rather than proved.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping

from rel.agents.dp import FixedPolicyAgent, value_iteration
from rel.agents.td import QLearning
from rel.core import TabularEnv
from rel.envs.gaming import BoatRace, Thermostat, VaseRoom
from rel.rng import Rng
from rel.training import evaluate, train
from rel.ui.colour import Palette, palette_for
from rel.ui.table import table

Builder = Callable[[Rng], TabularEnv]


def _discount(args: argparse.Namespace, env: TabularEnv) -> float:
    """The discount to use: the one asked for, or the one the environment wants.

    Two of these environments never end, so an undiscounted run of one is worth
    an unbounded amount and nothing converges. Each of them says so, and this
    takes them at their word unless the caller said otherwise.
    """
    if args.discount is not None:
        return float(args.discount)
    return env.spec.suggested_discount


def _solved(build: Builder, discount: float) -> tuple[float, Mapping[str, float]]:
    """The best possible policy under the stated reward, and what it really did."""
    env = build(Rng(1).stream("env"))
    best = value_iteration(env, discount=discount)
    agent = FixedPolicyAgent(Rng(1).stream("agent"), env.action_space, best.policy)
    record = evaluate(env, agent, 1, discount=discount)
    return record.final(), record.final_audit(1)


def _learned(
    build: Builder, episodes: int, discount: float, seed: int = 23
) -> tuple[float, Mapping[str, float]]:
    rng = Rng(seed)
    env = build(rng.stream("env"))
    agent: QLearning[int] = QLearning(
        rng.stream("agent"),
        env.action_space,
        step_size=0.3,
        epsilon=0.15,
        discount=discount,
    )
    train(env, agent, episodes)

    watched = evaluate(build(Rng(seed + 1).stream("env")), agent, 3, discount=discount)
    return watched.final(), watched.final_audit(3)


def _section(palette: Palette, title: str, says: str, wanted: str) -> None:
    print()
    print(palette.paint(title, "bold"))
    print(f"  the reward says   {palette.paint(says, 'yellow')}")
    print(f"  the point was     {palette.paint(wanted, 'cyan')}")
    print()


def _boat_race(args: argparse.Namespace, palette: Palette) -> None:
    _section(
        palette,
        "THE BOAT RACE",
        "touch a checkpoint, get a point",
        "go round the course",
    )

    rows = []
    for label, mode in (
        ("touch, as written", "touch"),
        ("ordered, repaired", "ordered"),
        ("laps only, repaired", "laps"),
    ):

        def build(rng: Rng, mode: str = mode) -> TabularEnv:
            return BoatRace(rng, reward=mode)

        discount = _discount(args, build(Rng(1).stream("env")))
        paid, audited = _solved(build, discount)
        row = [
            label,
            f"{paid:.1f}",
            f"{audited['laps']:.0f}",
            f"{audited['reverse_share']:.0%}",
        ]
        if args.learn:
            paid, audited = _learned(build, args.episodes, discount)
            row += [f"{paid:.1f}", f"{audited['laps']:.0f}"]
        rows.append(row)

    headings = ["reward", "paid", "LAPS", "reversing"]
    align = ["left", "right", "right", "right"]
    if args.learn:
        headings += ["paid (learned)", "LAPS (learned)"]
        align += ["right", "right"]

    for line in table(headings, rows, align=align, palette=palette):
        print(f"  {line}")

    print()
    print(
        "  The specification that pays three times as much completes no laps\n"
        "  at all, and spends half of its moves in reverse. Two checkpoints\n"
        "  sit next to each other, so driving back and forth between them\n"
        "  pays a point every step. A whole lap pays five points for sixteen\n"
        "  steps."
    )
    print(
        "\n"
        "  The repair costs something. Paying only for the next checkpoint in\n"
        "  order means the environment has to remember which ones have been\n"
        "  taken, and the state grows from 16 to 80."
    )


def _vase_room(args: argparse.Namespace, palette: Palette) -> None:
    _section(
        palette,
        "THE VASE ROOM",
        "one point off for every step you take",
        "reach the goal without breaking anything",
    )

    rows = []
    for penalty in (0.0, 1.0, 2.0, 3.0):

        def build(rng: Rng, penalty: float = penalty) -> TabularEnv:
            return VaseRoom(rng, vase_penalty=penalty)

        discount = _discount(args, build(Rng(1).stream("env")))
        paid, audited = _solved(build, discount)
        broken = "yes" if audited["vase_broken"] else "no"
        row = [
            "as written" if penalty == 0.0 else f"penalty {penalty:g}",
            f"{paid:.1f}",
            palette.paint(broken, "red" if broken == "yes" else "green"),
        ]
        if args.learn:
            paid, audited = _learned(build, args.episodes, discount)
            row.append(
                palette.paint(
                    "yes" if audited["vase_broken"] else "no",
                    "red" if audited["vase_broken"] else "green",
                )
            )
        rows.append(row)

    headings = ["reward", "paid", "VASE BROKEN"]
    align = ["left", "right", "right"]
    if args.learn:
        headings.append("broken (learned)")
        align.append("right")

    for line in table(headings, rows, align=align, palette=palette):
        print(f"  {line}")

    print()
    print(
        "  The short way to the goal goes through the vase and the way round\n"
        "  costs two more steps. Nothing in the reward mentions the vase, so\n"
        "  the vase is worth nothing, so two steps are worth more than it.\n"
        "\n"
        "  The penalty that changes the answer is 2, which is the length of\n"
        "  the detour. It is not a number about what a vase is worth. A\n"
        "  penalty chosen by asking what a vase is worth would be the wrong\n"
        "  kind of number even where it happened to work."
    )


def _thermostat(args: argparse.Namespace, palette: Palette) -> None:
    _section(
        palette,
        "THE THERMOSTAT",
        "a point for every step the sensor says the room is comfortable",
        "keep the room comfortable",
    )

    rows = []
    for label, mode in (
        ("sensor, as written", "sensor"),
        ("the temperature itself", "true"),
        ("sensor, dial removed", "sealed"),
    ):

        def build(rng: Rng, mode: str = mode) -> TabularEnv:
            return Thermostat(rng, reward=mode)

        discount = _discount(args, build(Rng(1).stream("env")))
        paid, audited = _solved(build, discount)
        row = [
            label,
            f"{paid:.0f}",
            f"{audited['comfortable_share']:.1%}",
            "yes" if audited["sensor_tampered"] else "no",
        ]
        if args.learn:
            paid, audited = _learned(build, args.episodes, discount)
            row += [f"{paid:.0f}", f"{audited['comfortable_share']:.1%}"]
        rows.append(row)

    headings = ["reward", "paid", "REALLY COMFORTABLE", "dial turned"]
    align = ["left", "right", "right", "right"]
    if args.learn:
        headings += ["paid (learned)", "COMFORTABLE (learned)"]
        align += ["right", "right"]

    for line in table(headings, rows, align=align, palette=palette):
        print(f"  {line}")

    print()
    print(
        "  One cell of the room is a dial. Standing on it once makes the\n"
        "  sensor report comfortable from then on, whatever the room is\n"
        "  doing. So the best policy walks to the dial, turns it, and stops\n"
        "  caring about the temperature. It collects a full score while the\n"
        "  room freezes.\n"
        "\n"
        "  The agent is not deceiving anybody, and it is not deceiving itself\n"
        "  either. The number it was told to maximise really did go up. What\n"
        "  went wrong is that the number was a reading rather than the thing\n"
        "  being read.\n"
        "\n"
        "  The two repairs are not the same repair. Removing the dial means\n"
        "  finding every dial. Paying for the temperature means the dials\n"
        "  stop mattering, and it needs a way to measure the temperature,\n"
        "  which is usually why a reading was being used in the first place."
    )


def run_gaming(args: argparse.Namespace) -> int:
    palette = palette_for(forced=args.colour)

    print(palette.paint("Reward Enforced Learning", "bold"))
    print()
    print(
        "An agent maximises what it is paid, exactly and without\n"
        "interpretation. Where the payment and the intent differ, it follows\n"
        "the payment. Here are three of those, with the paid return next to\n"
        "the thing that was actually wanted."
    )
    print()
    print(
        palette.paint(
            "Every 'paid' column below is the best possible policy under that\n"
            "reward, worked out from the model rather than learned. None of\n"
            "this is an agent going wrong.",
            "grey",
        )
    )

    _boat_race(args, palette)
    _vase_room(args, palette)
    _thermostat(args, palette)

    print()
    print(palette.paint("What the three have in common", "bold"))
    print()
    print(
        "  Every one of these rewards is a reasonable thing to write down.\n"
        "  Touch the checkpoints. Do not waste steps. Keep the sensor happy.\n"
        "  Each one is also a proxy: it stands in for something that is\n"
        "  harder to measure, and it stops standing in for it exactly where\n"
        "  the optimum is.\n"
        "\n"
        "  A person reading the specification fills in the intent without\n"
        "  noticing. An optimiser does not, and it does not have to be clever\n"
        "  to find the gap: in all three the gap is where the maximum is.\n"
        "\n"
        "  Run `rel demo q-learning --env boatrace --discount 0.99` to watch\n"
        "  one of them happen."
    )
    return 0


__all__ = ["run_gaming"]
