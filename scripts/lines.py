#!/usr/bin/env python3
"""Count the lines of each part of the project.

The README carries a table of these. A table of numbers written by hand goes
stale on the first change nobody remembers to make, so it is produced instead.

    python scripts/lines.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AREAS = (
    ("Core", ("rel/core.py", "rel/rng.py", "rel/spaces.py", "rel/registry.py")),
    ("Environments", ("rel/envs",)),
    ("Agents", ("rel/agents",)),
    ("Network", ("rel/nn",)),
    ("Running", ("rel/training.py", "rel/metrics.py", "rel/schedules.py")),
    ("Interface", ("rel/ui",)),
    ("Command line", ("rel/cli.py", "rel/cli_gaming.py", "rel/__main__.py")),
)


def count(paths: tuple[str, ...]) -> tuple[int, int]:
    """How many files and how many lines these paths hold."""
    files = 0
    lines = 0
    for name in paths:
        target = ROOT / name
        found = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in found:
            files += 1
            lines += len(path.read_text(encoding="utf-8").splitlines())
    return files, lines


def main() -> int:
    print(f"{'area':<14} {'files':>6} {'lines':>7}")
    print(f"{'-' * 14} {'-' * 6} {'-' * 7}")

    total_files = 0
    total_lines = 0
    for name, paths in AREAS:
        files, lines = count(paths)
        total_files += files
        total_lines += lines
        print(f"{name:<14} {files:>6} {lines:>7}")

    print(f"{'-' * 14} {'-' * 6} {'-' * 7}")
    print(f"{'Total':<14} {total_files:>6} {total_lines:>7}")

    tests, test_lines = count(("tests",))
    scripts, script_lines = count(("scripts",))
    print()
    print(f"{'Tests':<14} {tests:>6} {test_lines:>7}")
    print(f"{'Scripts':<14} {scripts:>6} {script_lines:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
