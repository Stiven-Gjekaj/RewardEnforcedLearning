#!/usr/bin/env python3
"""Draw the README banner as an SVG of a terminal.

The banner is a picture of real output. Running this again after a change to
the numbers redraws it, so it cannot say something the project no longer does.

    python scripts/make_banner.py > assets/banner.svg

The text is measured rather than typed: the script runs the same solver the
`rel gaming` command runs and puts the answers in the picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.agents.dp import FixedPolicyAgent, value_iteration
from rel.envs.gaming import BoatRace, Thermostat, VaseRoom
from rel.rng import Rng
from rel.training import evaluate

WIDTH = 900
LINE = 22
LEFT = 26
TOP = 40
CHARACTER = 9.02

BACKGROUND = "#12141a"
FRAME = "#2a2f3a"
DIM = "#6b7280"
PLAIN = "#c9d1d9"
GOOD = "#4ade80"
BAD = "#f87171"
MARK = "#fbbf24"


def solved(env: object, discount: float) -> tuple[float, dict[str, float]]:
    best = value_iteration(env, discount=discount)  # type: ignore[arg-type]
    agent = FixedPolicyAgent(
        Rng(1),
        env.action_space,
        best.policy,  # type: ignore[attr-defined]
    )
    record = evaluate(env, agent, 1, discount=discount)  # type: ignore[arg-type]
    return record.final(), dict(record.final_audit(1))


def escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def row(index: int, pieces: list[tuple[str, str]]) -> str:
    """One line of the terminal, as coloured runs of text."""
    y = TOP + index * LINE
    parts = []
    column = 0
    for text, colour in pieces:
        x = LEFT + column * CHARACTER
        parts.append(f'<text x="{x:.1f}" y="{y}" fill="{colour}">{escape(text)}</text>')
        column += len(text)
    return "".join(parts)


def main() -> int:
    race_paid, race = solved(BoatRace(Rng(1), reward="touch"), 0.99)
    fixed_paid, fixed = solved(BoatRace(Rng(1), reward="ordered"), 0.99)
    vase_paid, vase = solved(VaseRoom(Rng(1)), 1.0)
    heat_paid, heat = solved(Thermostat(Rng(1), reward="sensor"), 0.99)

    lines = [
        [("$ ", DIM), ("rel gaming", PLAIN)],
        [],
        [("  the reward pays", DIM), ("            what it was for", DIM)],
        [("  ", PLAIN), ("-" * 18, FRAME), ("  ", PLAIN), ("-" * 30, FRAME)],
        [
            ("  boat race      ", PLAIN),
            (f"{race_paid:6.0f}", MARK),
            ("     laps completed  ", PLAIN),
            (f"{race['laps']:.0f}", BAD),
            (f"   (of {fixed['laps']:.0f})", DIM),
        ],
        [
            ("  vase room      ", PLAIN),
            (f"{vase_paid:6.0f}", MARK),
            ("     vase          ", PLAIN),
            ("broken" if vase["vase_broken"] else "whole", BAD),
        ],
        [
            ("  thermostat     ", PLAIN),
            (f"{heat_paid:6.0f}", MARK),
            ("     room comfortable  ", PLAIN),
            (f"{heat['comfortable_share']:.0%}", BAD),
        ],
        [],
        [
            ("  Every one of those is the best possible policy under the", DIM),
            ("", DIM),
        ],
        [("  reward it was given, worked out from the model.", DIM)],
        [],
        [
            ("  The agent is not going wrong. ", PLAIN),
            ("The specification is.", GOOD),
        ],
    ]

    height = TOP + len(lines) * LINE + 24
    body = "\n  ".join(row(index, pieces) for index, pieces in enumerate(lines))
    unused = fixed_paid  # kept so the fixed reward is measured, not assumed
    assert unused > 0

    print(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height}" viewBox="0 0 {WIDTH} {height}" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
        f'font-size="15">\n'
        f'  <rect width="{WIDTH}" height="{height}" rx="10" fill="{BACKGROUND}" '
        f'stroke="{FRAME}"/>\n'
        f"  {body}\n"
        f"</svg>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
