"""The command line.

Every run this project can do is one of these, and every number in the
documentation comes from one of them with the seed written next to it.

    rel list
    rel train q-learning --env cliff --seed 7
    rel compare sarsa q-learning --env cliff
    rel solve --env cliff
    rel demo q-learning --env cliff
    rel gaming

## What a report says, and why it says all of it

A learning run has more than one answer and printing one of them is how a
result stops being checkable.

- The return while learning includes the cost of exploring. It is what the
  chapter plots and it is not what the agent knows.
- The return of the greedy policy leaves the exploring out. It is higher, and
  it can be catastrophically lower when the greedy policy walks into a corner
  and stops, which happens and is reported rather than smoothed over.
- The exact value of the greedy policy, where the environment has a model.
  This has no sampling noise in it at all, and it says outright when the policy
  never finishes.
- The best possible value, from the same model. Without it a return is a number
  with nothing to compare it to.
- The audit: what the episode really did, whether or not the reward paid for
  it. On three of the environments this is the whole point.
- The digest, so that two runs can be compared without comparing prose.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from typing import Any

from rel import __version__
from rel.agents import AGENTS
from rel.agents.base import Agent
from rel.agents.dp import DidNotSettleError, evaluate_policy, value_iteration
from rel.core import Env, TabularEnv
from rel.envs import ENVIRONMENTS
from rel.envs.gridfile import read as read_grid
from rel.envs.gridworld import GridWorld
from rel.metrics import summarise
from rel.recording import Recorder, read_run, save_run
from rel.rng import Rng
from rel.training import Episode, Record, evaluate, train
from rel.ui.chart import line_chart, smooth, sparkline
from rel.ui.colour import Palette, palette_for
from rel.ui.grid import best_values, difference_map, policy_map, value_map
from rel.ui.live import Live
from rel.ui.table import pairs, table

DEFAULT_EPISODES = 500
DEFAULT_SEED = 1


# -- Reading what the caller asked for ---------------------------------------


def parse_value(text: str) -> Any:
    """Turn a setting written on the command line into a value.

    `--set epsilon=0.1` has to give a float and `--set n=4` an integer, because
    an agent that receives the string "4" fails somewhere far away from here.
    """
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null"):
        return None

    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_settings(pairs_given: Sequence[str] | None) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for item in pairs_given or ():
        if "=" not in item:
            raise SystemExit(
                f"A setting looks like name=value. {item!r} has no equals sign."
            )
        name, _, value = item.partition("=")
        settings[name.strip()] = parse_value(value.strip())
    return settings


def build(
    environment: str | None,
    agent_name: str | None,
    seed: int,
    env_settings: dict[str, Any],
    agent_settings: dict[str, Any],
    env_file: str | None = None,
) -> tuple[Env[Any], Agent[Any] | None]:
    """Build the environment and the agent from one seed.

    They take different streams of the same seed. That is what lets two agents
    meet the same environment: the number of times one of them draws does not
    move what the other one faces.

    `env_file` is a grid written in a text file rather than named in the
    registry. It takes the same settings, so a reader can start from somebody
    else's grid and change one number without editing the file.
    """
    root = Rng(seed)
    if env_file is not None:
        env: Env[Any] = read_grid(env_file).build(root.stream("env"), **env_settings)
    elif environment is not None:
        env = ENVIRONMENTS.make(environment, root.stream("env"), **env_settings)
    else:
        raise ValueError("An environment is named with --env or --env-file.")

    agent = None
    if agent_name is not None:
        agent = AGENTS.make(agent_name, root.stream("agent"), env, **agent_settings)
    return env, agent


# -- Reporting ---------------------------------------------------------------


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _number(value: float, places: int = 2) -> str:
    if not _finite(value):
        return "never finishes"
    return f"{value:.{places}f}"


def exact_report(
    env: Env[Any], agent: Agent[Any], discount: float
) -> dict[str, float] | None:
    """The value of the greedy policy and of the best one, worked out exactly.

    `None` when the environment keeps no model. This is the part of a report
    with no sampling noise in it, and the part that says outright when a policy
    never finishes rather than reporting the step limit as a return.
    """
    if not isinstance(env, TabularEnv):
        return None

    policy = [agent.greedy(state) for state in range(env.observation_space.n)]
    try:
        best = value_iteration(env, discount=discount)
        learned = evaluate_policy(env, policy, discount=discount)
    except DidNotSettleError:
        return None

    return {
        "best": best.start_value,
        "learned": learned.start_value,
        "reaches_end": 1.0 if learned.reaches_end else 0.0,
    }


def report_lines(
    env: Env[Any],
    agent: Agent[Any],
    learning: Record,
    watched: Record,
    exact: dict[str, float] | None,
    palette: Palette,
    seconds: float,
    *,
    over: int = 100,
) -> list[str]:
    lines: list[str] = []

    learned_tail = summarise(learning.returns[-over:])
    watched_summary = summarise(watched.returns)

    # An environment with no ending truncates every episode by definition, so
    # counting them would describe the environment rather than the policy.
    cut_off = 0
    if env.spec.ends:
        cut_off = sum(1 for finished in watched.terminated if not finished)

    rows = [
        ("episodes", f"{len(learning)} learning, {len(watched)} watched"),
        ("steps", f"{learning.steps:,} learning, {watched.steps:,} watched"),
        ("time", f"{seconds:.2f}s"),
        (
            f"return, last {min(over, len(learning))} while learning",
            f"{learned_tail.mean:.2f} +/- {learned_tail.error:.2f}",
        ),
        (
            "return of the greedy policy",
            f"{watched_summary.mean:.2f} +/- {watched_summary.error:.2f}"
            + (
                palette.paint(
                    f"   ({cut_off} of {len(watched)} ran out of steps)", "yellow"
                )
                if cut_off
                else ""
            ),
        ),
    ]

    if exact is not None:
        share = ""
        if exact["reaches_end"] and _finite(exact["learned"]) and exact["best"] != 0:
            share = f"   ({exact['learned'] / exact['best']:.1%} of the best)"
        rows.append(
            (
                "exact value of the greedy policy",
                (
                    _number(exact["learned"])
                    if exact["reaches_end"]
                    else palette.paint("never reaches an ending", "red")
                )
                + share,
            )
        )
        rows.append(("best possible", _number(exact["best"])))

    lines.extend(pairs(rows, palette))

    audited = learning.final_audit(over)
    if audited:
        lines.append("")
        lines.append(palette.paint("What the reward did not pay for", "bold"))
        lines.extend(
            pairs(
                [(name, f"{value:.3f}") for name, value in sorted(audited.items())],
                palette,
            )
        )

    lines.append("")
    lines.extend(
        pairs([("digest", palette.paint(learning.digest.hexdigest(), "grey"))], palette)
    )
    return lines


def grid_pictures(
    env: Env[Any], agent: Agent[Any], palette: Palette, discount: float
) -> list[str]:
    """The policy, the value map and where the policy differs from the best."""
    if not isinstance(env, GridWorld):
        return []

    lines = ["", palette.paint("The policy it learned", "bold")]
    lines.extend(policy_map(env, agent, palette).splitlines())

    try:
        values = best_values(env, agent)
    except TypeError:
        # An agent that keeps no value for an action, or one whose numbers are
        # shares rather than returns. Both draw a policy and neither draws a
        # value map.
        values = []
    if values:
        lines.append("")
        lines.append(palette.paint("What it thinks each cell is worth", "bold"))
        lines.extend(value_map(env, values, palette).splitlines())

    try:
        best = value_iteration(env, discount=discount)
    except DidNotSettleError:
        return lines

    lines.append("")
    lines.append(palette.paint("Where it differs from the best policy", "bold"))
    lines.extend(difference_map(env, agent, best.policy, palette).splitlines())
    return lines


# -- The commands ------------------------------------------------------------


def command_list(args: argparse.Namespace) -> int:
    palette = palette_for(forced=args.colour)

    print(palette.paint("Environments", "bold"))
    print()
    places = [
        [place.name, ", ".join(place.tags), place.summary] for place in ENVIRONMENTS
    ]
    for line in table(["name", "tags", "what it is"], places, palette=palette):
        print(f"  {line}")

    print()
    print(palette.paint("Agents", "bold"))
    print()
    learners = [
        [learner.name, ", ".join(learner.tags), learner.summary] for learner in AGENTS
    ]
    for line in table(["name", "tags", "what it does"], learners, palette=palette):
        print(f"  {line}")

    if args.settings:
        print()
        print(palette.paint("Settings", "bold"))
        print()
        accepted = [
            ["--env " + place.name, _settings_text(place.options(ENVIRONMENTS.fixed))]
            for place in ENVIRONMENTS
        ]
        accepted += [
            [learner.name, _settings_text(learner.options(AGENTS.fixed))]
            for learner in AGENTS
        ]
        for line in table(["for", "--set accepts"], accepted, palette=palette):
            print(f"  {line}")
    else:
        print()
        print("Run with --settings to see what each one accepts.")

    return 0


def resolve_discount(args: argparse.Namespace, env: Env[Any]) -> float:
    """The discount to use: the one asked for, or the one the environment wants."""
    if args.discount is not None:
        return float(args.discount)
    return env.spec.suggested_discount


def _window(args: argparse.Namespace, episodes: int) -> int:
    """How many episodes one point of the chart averages over.

    A learning curve on the cliff walk moves by fifty from one episode to the
    next, because one fall costs a hundred. Drawn raw it is a band rather than
    a curve, and the difference between two agents disappears into it.
    """
    chosen = int(args.smooth)
    if chosen > 0:
        return chosen
    return max(1, episodes // 25)


def _settings_text(options: dict[str, Any]) -> str:
    if not options:
        return "nothing"
    return ", ".join(f"{name}={value!r}" for name, value in options.items())


def command_solve(args: argparse.Namespace) -> int:
    palette = palette_for(forced=args.colour)
    env, _ = build(
        args.env, None, args.seed, parse_settings(args.env_set), {}, args.env_file
    )

    if not isinstance(env, TabularEnv):
        print(
            f"{args.env} keeps no model, so there is nothing to solve exactly. "
            f"Environments tagged 'tabular' have one.",
            file=sys.stderr,
        )
        return 2

    discount = resolve_discount(args, env)
    started = time.perf_counter()
    try:
        best = value_iteration(env, discount=discount)
    except DidNotSettleError as failure:
        print(f"{failure}", file=sys.stderr)
        if discount >= 1.0:
            print(
                "\nThis environment may never end. Try a discount below one, "
                "for example --discount 0.99.",
                file=sys.stderr,
            )
        return 2
    seconds = time.perf_counter() - started

    print(
        "\n".join(
            pairs(
                [
                    ("environment", f"{env.spec.name}   {env.spec.summary}"),
                    ("states", str(env.observation_space.n)),
                    ("actions", str(env.action_space.n)),
                    ("discount", f"{discount:g}"),
                    ("sweeps", str(best.sweeps)),
                    ("time", f"{seconds:.3f}s"),
                    ("best possible return", f"{best.start_value:.4f}"),
                ],
                palette,
            )
        )
    )

    if isinstance(env, GridWorld):
        from rel.agents.dp import FixedPolicyAgent

        agent = FixedPolicyAgent(Rng(args.seed), env.action_space, best.policy)
        print()
        print(palette.paint("The best policy", "bold"))
        print(policy_map(env, agent, palette))
        print()
        print(palette.paint("What each cell is worth under it", "bold"))
        print(value_map(env, best.values, palette))

    return 0


def command_replay(args: argparse.Namespace) -> int:
    """Read a run out of a file and draw what it did.

    The digest is checked on the way in, so a file that has been changed since
    it was written is refused rather than drawn.
    """
    palette = palette_for(forced=args.colour)
    found = read_run(args.path)
    record = found.record()

    named = [
        (name, palette.paint(found.header[name], "grey"))
        for name in ("env", "agent", "seed", "episodes", "discount", "steps")
        if name in found.header
    ]
    named.append(("digest", palette.paint(found.header["digest"], "grey")))
    for line in pairs(named, palette):
        print(line)

    if not record.returns:
        print("\nThe file holds steps and no episode that ended.")
        return 0

    window = _window(args, len(record))
    label = found.header.get("agent", "the run")
    print()
    for line in line_chart(
        {label: smooth(record.returns, window)},
        width=args.width,
        height=args.height,
        palette=palette,
        clip=args.clip,
        title=palette.paint(
            f"{label} on {found.header.get('env', 'an environment')}, "
            f"return per episode, smoothed over {window}",
            "bold",
        ),
    ):
        print(line)

    summary = record.summary()
    ended = sum(record.terminated)
    print()
    for line in pairs(
        [
            ("episodes", f"{len(record):,}"),
            ("return, mean", f"{summary.mean:.2f} +/- {summary.error:.2f}"),
            ("return, last 100", f"{record.final(100):.2f}"),
            ("episodes that ended", f"{ended:,} of {len(record):,}"),
        ],
        palette,
    ):
        print(line)
    return 0


def command_train(args: argparse.Namespace) -> int:
    palette = palette_for(forced=args.colour)
    env, agent = build(
        args.env,
        args.agent,
        args.seed,
        parse_settings(args.env_set),
        parse_settings(args.set),
        args.env_file,
    )
    assert agent is not None
    discount = resolve_discount(args, env)

    # A run that is being recorded hands the loop a recorder in place of its
    # digest. Nothing else about the run changes, so a recorded run and an
    # unrecorded one on the same seed have the same digest.
    recorder = Recorder() if args.out else None

    live = Live(enabled=(not args.quiet) and args.colour is not False)
    started = time.perf_counter()

    def draw(_episode: Episode, record: Record) -> None:
        if args.quiet:
            return
        window = _window(args, len(record))
        live.update(
            [
                f"{args.agent} on {args.env}, seed {args.seed}",
                *line_chart(
                    {args.agent: smooth(record.returns, window)},
                    width=args.width,
                    height=args.height,
                    palette=palette,
                    clip=args.clip,
                ),
                f"episode {len(record)} of {args.episodes}"
                f"   last {min(50, len(record))}: {record.final(50):.2f}",
            ]
        )

    with live:
        learning = train(
            env,
            agent,
            args.episodes,
            discount=discount,
            on_episode=draw,
            digest=recorder,
        )
    seconds = time.perf_counter() - started

    if recorder is not None:
        written = save_run(
            args.out,
            recorder,
            env=env.spec.name,
            agent=args.agent,
            seed=args.seed,
            episodes=args.episodes,
            discount=f"{discount:g}",
        )
        if not args.quiet:
            print(f"wrote {recorder.steps} steps to {written}")

    watch_env, _ = build(
        args.env, None, args.seed + 1, parse_settings(args.env_set), {}, args.env_file
    )
    watched = evaluate(watch_env, agent, args.watch, discount=discount)
    exact = exact_report(env, agent, discount)

    if args.print == "digest":
        print(learning.digest.hexdigest())
        return 0
    if args.print == "final":
        print(f"{learning.final(100):.4f}")
        return 0

    if not args.quiet:
        window = _window(args, args.episodes)
        marks = {}
        if exact is not None and _finite(exact["best"]):
            marks["best possible"] = exact["best"]
        print()
        for line in line_chart(
            {args.agent: smooth(learning.returns, window)},
            width=args.width,
            height=args.height,
            palette=palette,
            marks=marks,
            clip=args.clip,
            title=palette.paint(
                f"{args.agent} on {args.env}, return per episode, "
                f"smoothed over {window}",
                "bold",
            ),
        ):
            print(line)
        print()

    for line in report_lines(env, agent, learning, watched, exact, palette, seconds):
        print(line)

    if args.maps:
        for line in grid_pictures(env, agent, palette, discount):
            print(line)

    return 0


def parse_over(given: Sequence[str] | None) -> list[tuple[str, list[Any]]]:
    """`--over name=a,b,c` as a setting and the values to try for it.

    The values go through the same reader as `--set`, so `0.1` is a float,
    `4` is an integer and `off` is false. A sweep that handed an agent the
    string "0.1" would fail somewhere far away from here.
    """
    swept: list[tuple[str, list[Any]]] = []
    for item in given or ():
        name, _, values = item.partition("=")
        if not values.strip():
            raise SystemExit(
                f"A sweep looks like name=a,b,c. {item!r} names no values."
            )
        swept.append(
            (name.strip(), [parse_value(part.strip()) for part in values.split(",")])
        )
    return swept


def sweep_settings(
    swept: Sequence[tuple[str, list[Any]]],
) -> list[dict[str, Any]]:
    """Every combination of the values, in the order the settings were given.

    Two settings give every pair. That is the point of sweeping two: the ones
    in this project trade against each other, and a sweep that varied them one
    at a time would miss exactly that.
    """
    combinations: list[dict[str, Any]] = [{}]
    for name, values in swept:
        combinations = [
            {**chosen, name: value} for chosen in combinations for value in values
        ]
    return combinations


def command_sweep(args: argparse.Namespace) -> int:
    palette = palette_for(forced=args.colour)
    swept = parse_over(args.over)
    if not swept:
        raise SystemExit("A sweep needs at least one --over name=a,b,c.")

    fixed = parse_settings(args.set)
    env_settings = parse_settings(args.env_set)
    overlap = sorted(set(fixed) & {name for name, _ in swept})
    if overlap:
        raise SystemExit(
            f"{', '.join(overlap)} is both fixed with --set and swept with "
            f"--over. One of the two has to go."
        )

    rows = []
    for chosen in sweep_settings(swept):
        finals: list[float] = []
        exact_values: list[float] = []
        stuck = 0

        for run in range(args.runs):
            env, agent = build(
                args.env,
                args.agent,
                args.seed + run,
                env_settings,
                {**fixed, **chosen},
                args.env_file,
            )
            assert agent is not None

            discount = resolve_discount(args, env)
            learning = train(env, agent, args.episodes, discount=discount)
            finals.append(learning.final(100))

            exact = exact_report(env, agent, discount)
            if exact is not None:
                if exact["reaches_end"]:
                    exact_values.append(exact["learned"])
                else:
                    stuck += 1

        learned = summarise(finals)
        exact_text = "-"
        if exact_values:
            exact_text = f"{sum(exact_values) / len(exact_values):.2f}"

        row = [
            f"{chosen[name]:g}"
            if isinstance(chosen[name], float)
            else str(chosen[name])
            for name, _ in swept
        ]
        row.extend(
            [
                f"{learned.mean:.2f}",
                f"+/- {learned.error:.2f}",
                exact_text,
                str(stuck) if stuck else "",
            ]
        )
        if args.each_seed:
            row.append(" ".join(f"{value:.1f}" for value in finals))
        rows.append(row)

    headings = [name for name, _ in swept]
    headings.extend(["last 100", "error", "exact value", "stuck"])
    aligns = ["right"] * len(swept) + ["right", "left", "right", "right"]
    if args.each_seed:
        headings.append("every seed")
        aligns.append("left")

    print(
        f"{args.agent} on {args.env or args.env_file}, {args.episodes} episodes, "
        f"seeds {args.seed} to {args.seed + args.runs - 1}"
    )
    print()
    for line in table(headings, rows, align=aligns, palette=palette):
        print(line)

    print(
        "\n'last 100' is the mean return of the last hundred episodes, averaged\n"
        "over the seeds, with the standard error of that mean. 'exact value' is\n"
        "the value of the greedy policy from the model, over the seeds whose\n"
        "policy reaches an ending."
    )
    return 0


def command_compare(args: argparse.Namespace) -> int:
    palette = palette_for(forced=args.colour)
    settings = parse_settings(args.set)
    env_settings = parse_settings(args.env_set)

    curves: dict[str, list[float]] = {}
    rows = []
    marks: dict[str, float] = {}

    for name in args.agents:
        totals: list[list[float]] = []
        finals: list[float] = []
        exact_values: list[float] = []
        stuck = 0

        for run in range(args.runs):
            seed = args.seed + run
            env, agent = build(
                args.env, name, seed, env_settings, settings, args.env_file
            )
            assert agent is not None

            discount = resolve_discount(args, env)
            learning = train(env, agent, args.episodes, discount=discount)
            totals.append(learning.returns)
            finals.append(learning.final(100))

            exact = exact_report(env, agent, discount)
            if exact is not None:
                if exact["reaches_end"]:
                    exact_values.append(exact["learned"])
                else:
                    stuck += 1
                marks.setdefault("best possible", exact["best"])

        curves[name] = [
            sum(values[index] for values in totals) / len(totals)
            for index in range(args.episodes)
        ]

        learned = summarise(finals)
        exact_text = "-"
        if exact_values:
            exact_text = f"{sum(exact_values) / len(exact_values):.2f}"
        if stuck:
            exact_text += palette.paint(f"  ({stuck} stuck)", "yellow")

        rows.append(
            [
                name,
                f"{learned.mean:.2f}",
                f"+/- {learned.error:.2f}",
                exact_text,
                sparkline(curves[name], 24),
            ]
        )

    window = _window(args, args.episodes)
    print()
    for line in line_chart(
        {name: smooth(values, window) for name, values in curves.items()},
        width=args.width,
        height=args.height,
        palette=palette,
        marks=marks,
        clip=args.clip,
        title=palette.paint(
            f"{args.env}, mean of {args.runs} seeds, smoothed over {window}", "bold"
        ),
    ):
        print(line)
    print()

    for line in table(
        ["agent", "last 100", "error", "exact value", "curve"],
        rows,
        align=["left", "right", "left", "right", "left"],
        palette=palette,
    ):
        print(line)

    print()
    print(
        f"Seeds {args.seed} to {args.seed + args.runs - 1}, "
        f"{args.episodes} episodes each. "
        f"'last 100' is the return while learning, which includes the cost of "
        f"exploring."
    )
    return 0


def command_demo(args: argparse.Namespace) -> int:
    palette = palette_for(forced=args.colour)
    env, agent = build(
        args.env,
        args.agent,
        args.seed,
        parse_settings(args.env_set),
        parse_settings(args.set),
        args.env_file,
    )
    assert agent is not None

    if args.episodes:
        train(env, agent, args.episodes, discount=resolve_discount(args, env))

    watch_env, _ = build(
        args.env, None, args.seed + 1, parse_settings(args.env_set), {}, args.env_file
    )
    observation = watch_env.reset()
    agent.start_episode()

    live = Live(enabled=args.colour is not False, every=args.delay)
    total = 0.0
    frame: list[str] = []

    with live:
        for step in range(args.steps):
            action = agent.greedy(observation)
            outcome = watch_env.step(action)
            total += outcome.reward
            observation = outcome.observation

            frame = [
                palette.paint(
                    f"{args.agent} on {args.env} after {args.episodes} episodes",
                    "bold",
                ),
                *watch_env.render().splitlines(),
                f"step {step + 1}   action {action}   "
                f"reward {outcome.reward:+.2f}   total {total:+.2f}",
            ]
            live.update(frame, force=True)
            time.sleep(args.delay)

            if outcome.done:
                break

    # Nothing was drawn if this is not a terminal, because moving a cursor in a
    # pipe writes the escape codes into the pipe. Leave the last frame behind,
    # so that a run in a log says something rather than nothing at all.
    if not live.enabled and frame:
        for line in frame:
            print(line)

    audited = watch_env.audit()
    if audited:
        print()
        print(palette.paint("What the reward did not pay for", "bold"))
        print(
            "\n".join(
                pairs(
                    [(name, f"{value:.3f}") for name, value in sorted(audited.items())],
                    palette,
                )
            )
        )
    return 0


def command_gaming(args: argparse.Namespace) -> int:
    """The demonstration the project is named for."""
    from rel.cli_gaming import run_gaming

    return run_gaming(args)


# -- Wiring ------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, *, one_env: bool = True) -> None:
    # One environment, named or written down. A grid in a file takes the same
    # `--env-set` overrides as a built in one, so a reader can start from
    # somebody else's file and change one number without editing it.
    #
    # `rel gaming` runs all three gaming environments and names none, which is
    # why this is a choice rather than always added.
    if one_env:
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--env", help="the name of a built in environment")
        source.add_argument(
            "--env-file",
            metavar="PATH",
            help="a grid written in a text file, in place of --env",
        )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--env-set",
        action="append",
        metavar="NAME=VALUE",
        help="a setting for the environment, repeatable",
    )
    parser.add_argument(
        "--colour",
        dest="colour",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force colour on or off. The default reads the terminal.",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=None,
        help=(
            "the discount to use. The default is the one the environment "
            "suggests, which is below one for an environment that never ends."
        ),
    )


def _add_chart(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help=(
            "how many episodes to average over before drawing. "
            "0 picks one from the length of the run."
        ),
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.05,
        help=(
            "share of the lowest values to leave off the bottom of the chart, "
            "so that the first few episodes do not squash the rest. "
            "0 draws the whole range."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rel",
        description=(
            "A reinforcement learning workbench that imports nothing. "
            "Every run is reproducible from its seed."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rel {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="every environment and agent")
    listing.add_argument(
        "--settings", action="store_true", help="also show what each one accepts"
    )
    listing.add_argument(
        "--colour", action=argparse.BooleanOptionalAction, default=None
    )
    listing.set_defaults(run=command_list)

    solving = commands.add_parser(
        "solve", help="the best possible policy, worked out from the model"
    )
    _add_common(solving)
    solving.set_defaults(run=command_solve)

    training = commands.add_parser("train", help="train one agent and report")
    training.add_argument("agent")
    training.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    training.add_argument(
        "--out",
        metavar="PATH",
        help="write the run to a file. A name ending in .gz is compressed.",
    )
    training.add_argument(
        "--watch",
        type=int,
        default=20,
        help="episodes of the greedy policy to run afterwards",
    )
    training.add_argument(
        "--set", action="append", metavar="NAME=VALUE", help="an agent setting"
    )
    training.add_argument("--quiet", action="store_true")
    training.add_argument(
        "--maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw the policy and the value map, for a grid",
    )
    training.add_argument(
        "--print",
        choices=("report", "digest", "final"),
        default="report",
        help="print one number instead of the report, for scripting",
    )
    _add_common(training)
    _add_chart(training)
    training.set_defaults(run=command_train)

    comparing = commands.add_parser(
        "compare", help="run several agents on one environment"
    )
    comparing.add_argument("agents", nargs="+")
    comparing.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    comparing.add_argument("--runs", type=int, default=5, help="how many seeds")
    comparing.add_argument("--set", action="append", metavar="NAME=VALUE")
    _add_common(comparing)
    _add_chart(comparing)
    comparing.set_defaults(run=command_compare)

    watching = commands.add_parser("demo", help="watch an agent play, step by step")
    watching.add_argument("agent")
    watching.add_argument("--episodes", type=int, default=300)
    watching.add_argument("--steps", type=int, default=200)
    watching.add_argument("--delay", type=float, default=0.12)
    watching.add_argument("--set", action="append", metavar="NAME=VALUE")
    _add_common(watching)
    watching.set_defaults(run=command_demo)

    sweeping = commands.add_parser(
        "sweep", help="vary one or two settings and print the table"
    )
    sweeping.add_argument("agent")
    sweeping.add_argument(
        "--over",
        action="append",
        metavar="NAME=A,B,C",
        help="a setting to vary and the values to try, repeatable",
    )
    sweeping.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    sweeping.add_argument("--runs", type=int, default=5, help="how many seeds")
    sweeping.add_argument(
        "--set", action="append", metavar="NAME=VALUE", help="a setting held fixed"
    )
    sweeping.add_argument(
        "--each-seed",
        action="store_true",
        help="show the number from every seed behind each row",
    )
    _add_common(sweeping)
    sweeping.set_defaults(run=command_sweep)

    replaying = commands.add_parser(
        "replay", help="read a recorded run out of a file and draw it"
    )
    replaying.add_argument("path", help="a file written by rel train --out")
    replaying.add_argument(
        "--colour",
        dest="colour",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force colour on or off. The default reads the terminal.",
    )
    _add_chart(replaying)
    replaying.set_defaults(run=command_replay)

    gaming = commands.add_parser(
        "gaming", help="the demonstration this project is named for"
    )
    gaming.add_argument("--episodes", type=int, default=500)
    gaming.add_argument(
        "--learn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also train an agent, rather than only solving the model",
    )
    gaming.add_argument(
        "--pressure",
        action="store_true",
        help="also walk each reward from a uniform policy to its optimum",
    )
    gaming.add_argument(
        "--pressure-episodes",
        type=int,
        default=40,
        dest="pressure_episodes",
        help="episodes at each rung of that walk",
    )
    _add_common(gaming, one_env=False)
    gaming.set_defaults(run=command_gaming)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result: int = args.run(args)
    except KeyError as missing:
        print(missing.args[0], file=sys.stderr)
        return 2
    except (TypeError, ValueError) as wrong:
        print(str(wrong), file=sys.stderr)
        return 2
    except DidNotSettleError as failure:
        print(f"{failure}", file=sys.stderr)
        print(
            "\nAn environment that never ends needs a discount below one. "
            "Try --discount 0.99.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    return result


if __name__ == "__main__":
    sys.exit(main())
