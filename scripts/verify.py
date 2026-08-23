#!/usr/bin/env python3
"""Run everything that continuous integration runs, in the same order.

The order matters. A formatting fault and a type fault often come from the same
edit, and the formatter reports it in one line while the type checker reports it
in twenty. Stopping at the first failure keeps the short message first.
"""

from __future__ import annotations

import subprocess
import sys

GATE: list[list[str]] = [
    ["ruff", "check", "."],
    ["ruff", "format", "--check", "."],
    ["mypy", "rel"],
    ["pytest"],
]


def main() -> int:
    for command in GATE:
        print(f"$ {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"\n{command[0]} failed. The rest of the gate did not run.")
            return result.returncode
    print("\nThe gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
