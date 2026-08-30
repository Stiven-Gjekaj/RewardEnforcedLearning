#!/usr/bin/env python3
"""Run what continuous integration runs, in the order that reports it best.

The order matters. A formatting fault and a type fault often come from the same
edit, and the formatter reports it in one line while the type checker reports it
in twenty. Stopping at the first failure keeps the short message first. The link
check goes first because it is the cheapest of them and its message is one line.

One job of the workflow is missing from this, and on purpose. `standalone`
builds an interpreter with nothing but this package in it and trains an agent
in it, to prove that the readme is right that the package imports nothing
outside the standard library. That needs a fresh virtual environment and a
network, and a developer already has both the package and its tools installed.
`tests/test_layering.py` reads the imports and says the same thing without
either.
"""

from __future__ import annotations

import subprocess
import sys

GATE: list[list[str]] = [
    ["bash", "scripts/check-links.sh"],
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
