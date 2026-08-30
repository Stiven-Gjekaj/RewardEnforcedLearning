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

Every command in a ```console block is run once. Then each table on the page is
put against every one of those outputs, and the one that accounts for most of
the table's numbers is the command that table came from.

Matching rather than position, and the first version did it by position. It
took the tables that followed a block and asked whether that block's commands
printed their numbers. Two thirds of what it reported was a table sitting under
a command it did not come from, because `## The same table, four more
environments` writes one command and then four tables, and the other three
commands are not on the page at all.

Matching finds those. A table whose numbers appear in the output of a command
written further up the page is attributed to it and reported as clean. A table
that no documented command accounts for is reported as exactly that, which is
the state the page is really in for a table whose command it never names.

## What it can and cannot see

It asks whether each number appears anywhere in an output. That is weaker than
asking whether it is the right number in the right cell, and it is the question
that can be asked of a page which nowhere says which column of which run a cell
came from. A number that moved disappears from every output, so it is caught. A
number that swapped places with another number in the same table is not.

Three things it reports that are not the page being wrong.

**A table that rounds.** The digits are compared as text, so a page writing
1,970,224,597,202 where the command prints 1970224597202.702 states a number
the output does not contain. Matching a rounded number would mean deciding how
near is near enough, and a rule loose enough to catch a truncation is loose
enough to confirm a number that really moved. The page is written to print what
the command prints instead.

**A timing.** `measure_engine.py` reports seconds, and seconds belong to the
machine. Those lines will differ on every run and there is nothing to fix.

**A command that takes longer than the budget.** Nothing ran, so nothing
accounts for its tables, and they are reported as unaccounted for. The summary
names those commands separately.
"""

from __future__ import annotations

import argparse
import json
import math
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

#: How a run says it gave up rather than failed. A command that takes longer
#: than the budget is a fact about the budget, and reporting it beside a
#: command that will not run would make the second one unfindable.
TIMED_OUT = "gave up after"

#: What a table says to be left alone with, and the reason it gives. Written
#: as an HTML comment above the table, so it is invisible in the rendered page
#: and in front of the reader who is editing the table.
#:
#:     <!-- not checked: these are seconds and belong to the machine -->
#:
#: There has to be a way to say this or the tool dies of its own noise. Three
#: tables on this page report timings, which differ on every machine and every
#: run, and a report that lists the same twenty unfixable numbers every time
#: teaches a reader to stop reading it.
#:
#: The reason is required and is printed in the report, because a table that
#: exempts itself without saying why is how a real defect gets hidden.
EXEMPT = "<!-- not checked"

#: What separates the column named in a marker from the reason for it, so that
#: one table can say "the time column belongs to the machine" and have the
#: other five columns still checked. Exempting the whole table over one column
#: would drop the model steps and the returns beside it, which are the numbers
#: the section is actually about.
COLUMN = ", column "

#: How much of a table one command has to account for before it is called the
#: command that table came from. Below this it is a coincidence: three numbers
#: of seventy is what any output shares with any table, because 0.5 and 10 and
#: 2 turn up everywhere.
#:
#: Half is a judgement and not a measurement. It is high enough that no pair on
#: this page reaches it by accident and low enough that a table which has drifted
#: in a few cells is still attributed and reported as drifted rather than as
#: unaccounted for.
ENOUGH = 0.5


def heading_of(row: str) -> str:
    """A table row as a key, with the pipes and the case taken off."""
    return row.strip("|").strip().lower()


@dataclass
class Claim:
    """One table in the documentation, and the numbers it states."""

    #: The line the table starts on, for a report a reader can navigate by.
    line: int
    #: The command written closest above it. A hint and not an answer: the
    #: report says when the command that accounts for a table is not this one.
    nearest: str = ""
    #: Why this table is not checked, when it says so itself. Empty otherwise.
    exempt: str = ""
    #: The one column that is not checked, when the marker names one. The rest
    #: of the table is checked as usual and `exempt` stays empty.
    skipped: str = ""
    rows: list[str] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        """Which column `skipped` is, counting from the left, or minus one.

        Read off the heading row rather than given, so that a marker naming a
        column that is not there is a marker that does nothing, and the report
        says the table is short of numbers rather than quietly passing.
        """
        if not self.skipped or not self.rows:
            return -1
        headings = [cell.strip().lower() for cell in self.rows[0].strip("|").split("|")]
        wanted = self.skipped.strip().lower()
        return headings.index(wanted) if wanted in headings else -1

    @property
    def numbers(self) -> list[str]:
        found: list[str] = []
        drop = self.dropped
        for row in self.rows:
            if set(row) <= set("|-: "):
                # The rule under a heading row. It carries no claim.
                continue
            cells = row.strip("|").split("|")
            for index, cell in enumerate(cells):
                if index == drop:
                    continue
                found.extend(NUMBER.findall(cell.replace(",", "").replace("*", "")))
        return found


def is_this_script(command: str) -> bool:
    """Whether a command runs this script.

    The page documents this script the way it documents every other, in a
    console block, and a block is a list of things to run. Running this one
    from inside itself would spend three hours to say nothing, and the version
    of it with `--all` would do that recursively.
    """
    return Path(__file__).name in command


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


def read(path: Path) -> tuple[list[str], list[Claim]]:
    """Every command in a markdown file, and every table, kept apart.

    Which table came from which command is not decided here. It is decided by
    running the commands and seeing which output accounts for which table.
    """
    lines = path.read_text().splitlines()
    commands: list[str] = []
    claims: list[Claim] = []
    claim: Claim | None = None
    # Whether the line before this one was part of a table. A table whose
    # heading says it holds commands is skipped whole, and without this the
    # rule under that heading would open a new table of its own.
    inside = False
    exempt = ""
    skipped = ""
    index = 0

    while index < len(lines):
        text = lines[index].strip()

        if text.startswith(EXEMPT):
            # A reason worth giving does not always fit on one line, and a
            # marker that silently does nothing when it wraps is worse than
            # no marker: the table stops being exempt and nothing says so.
            said = text[len(EXEMPT) :]
            while "-->" not in said and index + 1 < len(lines):
                index += 1
                said += " " + lines[index].strip()
            said = " ".join(said.split("-->")[0].split())

            column = ""
            if said.startswith(COLUMN.strip()):
                column, _, said = said[len(COLUMN) - 1 :].partition(":")
            else:
                said = said.removeprefix(":")

            reason = said.strip() or "no reason given"
            if column.strip():
                skipped, exempt = column.strip(), ""
            else:
                skipped, exempt = "", reason
            index += 1
            continue

        if text.startswith("```"):
            language = text.strip("`")
            fence: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                fence.append(lines[index])
                index += 1
            index += 1
            claim, inside = None, False
            if language == "console":
                for command in commands_in(fence):
                    if command not in commands and not is_this_script(command):
                        commands.append(command)
            continue

        if text.startswith("|"):
            if not inside:
                inside = True
                claim = (
                    None
                    if heading_of(text) in NOT_RESULTS
                    else Claim(
                        index + 1,
                        commands[-1] if commands else "",
                        exempt=exempt,
                        skipped=skipped,
                    )
                )
                if claim is not None:
                    claims.append(claim)
                exempt, skipped = "", ""
            if claim is not None:
                claim.rows.append(text)
        elif text:
            claim, inside = None, False
            exempt, skipped = "", ""
        else:
            claim, inside = None, False

        index += 1

    return commands, [claim for claim in claims if claim.numbers]


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
        return "", time.perf_counter() - started, f"{TIMED_OUT} {seconds:.0f}s"
    except FileNotFoundError as missing_program:
        return "", time.perf_counter() - started, f"could not run it: {missing_program}"

    taken = time.perf_counter() - started
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()
        return done.stdout, taken, f"exit {done.returncode}: {tail[-1] if tail else ''}"
    return done.stdout, taken, ""


def missing(wanted: list[str], available: list[str]) -> list[str]:
    """The numbers a table states that an output does not contain.

    Counted with multiplicity. A table that states -13.00 four times and an
    output that prints it twice is two numbers short, and reporting it as
    nothing short would hide exactly the kind of change this is for.
    """
    left = list(available)
    absent: list[str] = []
    for number in wanted:
        if number in left:
            left.remove(number)
        else:
            absent.append(number)
    return absent


def attribute(claim: Claim, printed: dict[str, list[str]]) -> tuple[str, list[str]]:
    """The command that accounts for most of a table, and what it still misses.

    Ties go to the command written nearest above the table, so a page that
    does say where a table came from is taken at its word whenever the numbers
    do not disagree with it.
    """
    wanted = claim.numbers
    best: tuple[str, list[str]] = ("", wanted)
    needed = math.ceil(len(wanted) * ENOUGH)
    for command, available in printed.items():
        absent = missing(wanted, available)
        if len(wanted) - len(absent) < needed:
            # It accounts for too little of the table to be where the table
            # came from. Without this the best of forty eight coincidences is
            # named as a source, and a small integer that every output happens
            # to print is enough to make one.
            continue
        if len(absent) < len(best[1]) or (
            len(absent) == len(best[1]) and command == claim.nearest
        ):
            best = (command, absent)
    return best


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
    parser.add_argument(
        "--cache",
        default="",
        help=(
            "a file to keep what each command printed in. Commands already "
            "in it are not run again, and the ones that are run are added to "
            "it. A whole run is about three hours and the answer changes as "
            "soon as the page is edited, so checking a fix should not cost "
            "three hours again"
        ),
    )
    args = parser.parse_args()

    commands, claims = read(ROOT / args.doc)
    if args.only:
        commands = [command for command in commands if args.only in command]
        claims = [claim for claim in claims if args.only in claim.nearest]

    if args.list or not (args.only or args.all):
        stated = sum(len(claim.numbers) for claim in claims if not claim.exempt)
        left = sum(len(claim.numbers) for claim in claims if claim.exempt)
        print(
            f"{len(commands)} commands and {len(claims)} tables in {args.doc}.\n"
            f"{stated} numbers are checked and {left} are not.\n"
        )
        for line in table(
            ["line", "numbers", "not checked", "nearest command above it"],
            [
                [
                    f"{claim.line}",
                    f"{len(claim.numbers)}",
                    claim.exempt
                    or (f"column {claim.skipped}" if claim.skipped else ""),
                    claim.nearest,
                ]
                for claim in claims
            ],
            align=["right", "right", "left", "left"],
        ):
            print(f"  {line}")
        if not args.list:
            print("\nRun with --only <text> for one, or --all for every one of them.")
        return 0

    started = time.perf_counter()
    kept = Path(args.cache) if args.cache else None
    printed: dict[str, list[str]] = {}
    if kept is not None and kept.exists():
        # Only the commands this page still asks for. A cache that outlived
        # the command it was made for would otherwise account for a table
        # with the output of something the page no longer says to run.
        held = json.loads(kept.read_text())
        printed = {name: held[name] for name in commands if name in held}
        print(f"{len(printed)} of {len(commands)} commands come from {kept}.\n")

    broken: list[str] = []
    slow: list[str] = []

    for number, command in enumerate(commands, start=1):
        if command in printed:
            continue
        print(f"[{number}/{len(commands)}] {command}", flush=True)
        output, spent, trouble = run(command, args.timeout)
        if trouble.startswith(TIMED_OUT):
            print(f"    {trouble}, so nothing below is checked against it", flush=True)
            slow.append(command)
            continue
        if trouble:
            print(f"    could not run it. {trouble}", flush=True)
            broken.append(command)
            continue
        print(f"    {spent:.0f}s", flush=True)
        printed[command] = NUMBER.findall(output.replace(",", ""))
        if kept is not None:
            # Written after every command rather than at the end, because a
            # run this long is one a person interrupts, and a cache that only
            # exists if the whole thing finished is a cache for nobody.
            kept.write_text(json.dumps(printed, indent=1, sort_keys=True))

    clean = 0
    orphans = 0
    exempt: list[Claim] = []
    print("\n")
    for claim in claims:
        if claim.exempt:
            exempt.append(claim)
            continue

        command, absent = attribute(claim, printed)
        stated = len(claim.numbers)
        if not absent:
            clean += 1
            if command != claim.nearest:
                print(
                    f"line {claim.line}: all {stated} numbers, but from\n"
                    f"  {command}\n"
                    f"  rather than the command above it, which is\n"
                    f"  {claim.nearest}"
                )
            continue

        if len(absent) == stated:
            orphans += 1
            print(
                f"line {claim.line}: no command on this page prints any of its "
                f"{stated} numbers."
            )
            continue

        shown = " ".join(absent[:12])
        more = "" if len(absent) <= 12 else f" and {len(absent) - 12} more"
        print(
            f"line {claim.line}: {stated - len(absent)} of {stated} numbers, best "
            f"from\n  {command}\n  missing {shown}{more}"
        )

    checked = len(claims) - len(exempt)
    print(
        f"\n{clean} of {checked} tables are wholly accounted for by a command "
        f"on the page,\nand {orphans} by no command at all, "
        f"in {time.perf_counter() - started:.0f}s."
    )

    if exempt:
        print(f"\n{len(exempt)} tables ask not to be checked, and say why:")
        for claim in exempt:
            print(f"  line {claim.line}: {claim.exempt}")
    if slow:
        print(
            f"\n{len(slow)} commands took longer than the {args.timeout:.0f}s budget."
            f"\nA table of theirs reads as unaccounted for above, because nothing\n"
            f"ran to account for it. Raise --timeout to check them:"
        )
        for command in slow:
            print(f"  {command}")

    if broken:
        # A documented command that will not run is a defect whatever the
        # numbers say, so this is the one thing here worth an exit code. A
        # command that ran out of time is not: that is this script's budget
        # rather than anything about the page.
        print(f"\n{len(broken)} commands would not run at all:")
        for command in broken:
            print(f"  {command}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
