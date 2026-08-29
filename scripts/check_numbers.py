#!/usr/bin/env python3
"""Whether the numbers in the documentation are still what the code produces.

Nineteen tracks of work have gone past the tables in `docs/algorithms.md`, and
several of them changed a default. A default that changes makes every table
produced with the old one wrong, silently, and nothing about reading the page
says which. This is the thing that says which.

    python scripts/check_numbers.py --list
    python scripts/check_numbers.py --only tiling
    python scripts/check_numbers.py --all

## How it decides

For each command in a ```console block, it takes every table that follows the
block, runs the command, and asks of each number in each table: does this
number appear anywhere in what the command printed?

That is a weaker question than "is this the right number in the right cell",
and it is the one that can be asked without the document declaring which table
came from which column of which run. A number that has moved disappears from
the output, so it is caught. A number that has swapped places with another
number in the same table is not.

## What a low score means

A table where nothing matches is more likely to be a table this attached to
the wrong command than a table that is wholly wrong. The rule is that a table
belongs to the nearest command above it, and a page that discusses one command
under several headings breaks that rule. Both look the same from here, so
both are reported and neither is called stale.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rel.ui.table import table

ROOT = Path(__file__).resolve().parent.parent

#: Every number, including a sign and a decimal point. Thousands separators are
#: stripped before this runs, so that the 52,488 a document writes and the
#: 52,488 a table prints are the same string by the time they are compared.
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: Tables whose cells are commands rather than results, named by their heading
#: row. The closing table of `algorithms.md` lists what produces every other
#: table on the page, so every number in it belongs to a command line and not
#: to a measurement, and checking them would ask whether `--runs 10` is still
#: printed.
NOT_RESULTS = frozenset({"table | command"})


def heading_of(row: str) -> str:
    """A table row as a key, with the pipes and the case taken off."""
    return row.strip("|").strip().lower()


@dataclass
class Claim:
    """One table in the documentation, and the numbers it states."""

    #: The line the table starts on, for a report a reader can navigate by.
    line: int
    rows: list[str] = field(default_factory=list)

    @property
    def numbers(self) -> list[str]:
        found: list[str] = []
        for row in self.rows:
            if set(row) <= set("|-: "):
                # The rule under a heading row. It carries no claim.
                continue
            found.extend(NUMBER.findall(row.replace(",", "").replace("*", "")))
        return found


@dataclass
class Block:
    """One console block in the documentation, and the tables that follow it.

    All of the commands rather than one of them. A block that shows two
    commands and then one table is showing a table built from both, so a
    number is looked for in what any of them printed.
    """

    commands: list[str]
    line: int
    claims: list[Claim] = field(default_factory=list)

    @property
    def numbers(self) -> int:
        return sum(len(claim.numbers) for claim in self.claims)

    @property
    def label(self) -> str:
        rest = len(self.commands) - 1
        return self.commands[0] if not rest else f"{self.commands[0]}  (+{rest})"


def commands_in(fence: list[str]) -> list[str]:
    """The commands inside one console block, with continuations joined.

    A command in this documentation is split across lines with a trailing
    backslash when it is long. Reading those as separate commands would run
    the first half of one and report that the rest of it produced nothing.
    """
    found: list[str] = []
    building = ""
    for line in fence:
        text = line.strip()
        if not building and not text.startswith("$ "):
            continue
        text = text[2:] if text.startswith("$ ") else text
        if text.endswith("\\"):
            building += text[:-1].strip() + " "
            continue
        found.append((building + text).strip())
        building = ""
    if building:
        found.append(building.strip())
    return found


def read(path: Path) -> list[Block]:
    """Every command in a markdown file, with the tables that follow it.

    A table belongs to the nearest command above it. Headings do not break
    that, because this page states a command once and then discusses what it
    printed under several headings.
    """
    lines = path.read_text().splitlines()
    blocks: list[Block] = []
    claim: Claim | None = None
    # Whether the line before this one was part of a table. A table whose
    # heading says it holds commands is skipped whole, and without this the
    # rule under that heading would open a new table of its own.
    inside = False
    index = 0

    while index < len(lines):
        text = lines[index].strip()

        if text.startswith("```"):
            fence: list[str] = []
            language = text.strip("`")
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                fence.append(lines[index])
                index += 1
            index += 1
            claim = None
            inside = False
            if language == "console":
                found = commands_in(fence)
                if found:
                    blocks.append(Block(found, index))
            continue

        if text.startswith("|") and blocks:
            if not inside:
                inside = True
                claim = None if heading_of(text) in NOT_RESULTS else Claim(index + 1)
                if claim is not None:
                    blocks[-1].claims.append(claim)
            if claim is not None:
                claim.rows.append(text)
        else:
            inside = False
            claim = None

        index += 1

    return blocks


def run(command: str, seconds: float) -> tuple[str, float, str]:
    """What a documented command printed, how long it took, and what went wrong.

    The command is run from the root of the repository, because that is where
    the documentation says to run it from, and with the same interpreter this
    script is running under rather than whatever `python` happens to mean.
    """
    parts = shlex.split(command)
    if parts and parts[0] == "python":
        parts[0] = sys.executable
    elif parts and parts[0] == "rel":
        # The documentation writes the installed entry point. Nothing here
        # needs the package installed, and running the module instead is the
        # same code under a name that always works.
        parts[:1] = [sys.executable, "-m", "rel"]

    started = time.perf_counter()
    try:
        done = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", time.perf_counter() - started, f"took longer than {seconds:.0f}s"
    except FileNotFoundError as missing:
        return "", time.perf_counter() - started, f"could not run it: {missing}"

    taken = time.perf_counter() - started
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()
        return done.stdout, taken, f"exit {done.returncode}: {tail[-1] if tail else ''}"
    return done.stdout, taken, ""


def missing(claim: Claim, printed: str) -> list[str]:
    """The numbers a table states that the output does not contain.

    Counted with multiplicity. A table that states -13.00 four times and an
    output that prints it twice is two numbers short, and reporting it as
    nothing short would hide exactly the kind of change this is for.
    """
    available = list(NUMBER.findall(printed.replace(",", "")))
    absent: list[str] = []
    for number in claim.numbers:
        if number in available:
            available.remove(number)
        else:
            absent.append(number)
    return absent


@dataclass
class Result:
    """What running one block's commands said about the numbers under it."""

    found: int
    claimed: int
    absent: list[str]
    trouble: str
    seconds: float


def check(block: Block, seconds: float) -> Result:
    """How many of a block's claimed numbers its commands still print."""
    printed = ""
    taken = 0.0
    for command in block.commands:
        output, spent, trouble = run(command, seconds)
        taken += spent
        if trouble:
            return Result(0, block.numbers, [], f"{command}: {trouble}", taken)
        printed += output

    absent: list[str] = []
    gone_total = 0
    for claim in block.claims:
        gone = missing(claim, printed)
        gone_total += len(gone)
        if gone:
            shown = " ".join(gone[:12])
            more = "" if len(gone) <= 12 else f" and {len(gone) - 12} more"
            absent.append(f"line {claim.line}: {shown}{more}")
    return Result(block.numbers - gone_total, block.numbers, absent, "", taken)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default="docs/algorithms.md")
    parser.add_argument("--only", default="", help="run commands containing this")
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every command, which takes about an hour",
    )
    parser.add_argument("--list", action="store_true", help="say what would run")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    blocks = [block for block in read(ROOT / args.doc) if block.numbers]
    if args.only:
        blocks = [
            block
            for block in blocks
            if any(args.only in command for command in block.commands)
        ]

    if args.list or not (args.only or args.all):
        total = sum(block.numbers for block in blocks)
        print(
            f"{len(blocks)} console blocks in {args.doc} have tables under\n"
            f"them, stating {total} numbers between them.\n"
        )
        rows = [
            [f"{block.line}", f"{len(block.claims)}", f"{block.numbers}", block.label]
            for block in blocks
        ]
        for line in table(
            ["line", "tables", "numbers", "command"],
            rows,
            align=["right", "right", "right", "left"],
        ):
            print(f"  {line}")
        if not args.list:
            print("\nRun with --only <text> for one, or --all for every one of them.")
        return 0

    started = time.perf_counter()
    stale = 0
    broken = 0
    for number, block in enumerate(blocks, start=1):
        # Said before the command runs rather than after. Some of these take
        # minutes, and a run that prints nothing for an hour looks the same
        # as a run that has hung.
        print(f"\n[{number}/{len(blocks)}] {block.label}", flush=True)

        result = check(block, args.timeout)
        if result.trouble:
            print(f"  could not check it. {result.trouble}", flush=True)
            broken += 1
            continue
        print(
            f"  {result.found} of {result.claimed} numbers still printed, "
            f"in {result.seconds:.0f}s",
            flush=True,
        )
        for line in result.absent:
            print(f"  {line}", flush=True)
        stale += bool(result.absent)

    checked = len(blocks) - broken
    print(
        f"\n{checked - stale} of {checked} blocks print every number the page "
        f"states for them, in {time.perf_counter() - started:.0f}s."
    )
    if stale:
        print(
            "A block that prints none of its numbers is more likely a table\n"
            "attached to the wrong command than a table that is wholly wrong.\n"
            "This cannot tell those apart, so it reports both and calls\n"
            "neither of them stale."
        )
    if broken:
        # A documented command that will not run is a defect whatever the
        # numbers under it say, so this is the one thing here worth an exit
        # code. Numbers that moved need a person to read them.
        print(f"{broken} of {len(blocks)} would not run at all.")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
