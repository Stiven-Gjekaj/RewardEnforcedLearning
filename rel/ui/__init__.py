"""Drawing. Nothing here decides anything.

`rel/envs/` and `rel/agents/` never import this module, and a test in
`tests/test_layering.py` holds that. A whole experiment therefore runs with no
terminal attached, which is what lets continuous integration run the suite and
what lets a result be produced by a machine with no display.

Everything here returns a string or a list of strings. Only `rel.ui.live`
writes anywhere, and it checks whether it is writing to a terminal first.
"""

from __future__ import annotations

from rel.ui.chart import histogram, line_chart, sparkline
from rel.ui.colour import Palette, palette_for
from rel.ui.grid import best_values, difference_map, policy_map, value_map
from rel.ui.live import Live
from rel.ui.table import pairs, table

__all__ = [
    "Live",
    "Palette",
    "best_values",
    "difference_map",
    "histogram",
    "line_chart",
    "pairs",
    "palette_for",
    "policy_map",
    "sparkline",
    "table",
    "value_map",
]
