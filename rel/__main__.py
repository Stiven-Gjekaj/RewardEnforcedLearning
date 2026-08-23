"""Lets `python -m rel` do what `rel` does."""

from __future__ import annotations

import sys

from rel.cli import main

if __name__ == "__main__":
    sys.exit(main())
