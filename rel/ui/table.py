"""Columns that line up, including when the cells carry colour.

`str.ljust` counts an escape code as characters, so a coloured cell comes out
several columns short and every column to the right of it walks left. The width
here is the width that is really printed.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.ui.colour import Palette, pad, width_of

RULE = "-"


def table(
    headings: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    align: Sequence[str] | None = None,
    palette: Palette | None = None,
    gap: str = "  ",
    rule: bool = True,
) -> list[str]:
    """A table as lines of text.

    `align` is one of `left`, `right` or `centre` for each column. Numbers read
    far better on the right, and the caller says so rather than this guessing
    from the contents, because a column of numbers with one dash in it would
    then change alignment.
    """
    palette = palette or Palette(enabled=False)
    aligns = list(align or ["left"] * len(headings))
    if len(aligns) != len(headings):
        raise ValueError("A table needs one alignment for each column.")

    widths = [width_of(heading) for heading in headings]
    for row in rows:
        if len(row) != len(headings):
            raise ValueError(
                f"A row has {len(row)} cells and the table has {len(headings)} columns."
            )
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], width_of(cell))

    lines = [
        gap.join(
            pad(palette.paint(heading, "bold"), widths[index], aligns[index])
            for index, heading in enumerate(headings)
        ).rstrip()
    ]

    if rule:
        lines.append(gap.join(RULE * width for width in widths))

    for row in rows:
        lines.append(
            gap.join(
                pad(cell, widths[index], aligns[index])
                for index, cell in enumerate(row)
            ).rstrip()
        )

    return lines


def pairs(
    items: Sequence[tuple[str, str]], palette: Palette | None = None
) -> list[str]:
    """A list of name and value lines, with the names padded to line up."""
    palette = palette or Palette(enabled=False)
    if not items:
        return []

    width = max(width_of(name) for name, _ in items)
    return [
        f"{pad(palette.paint(name, 'grey'), width)}  {value}" for name, value in items
    ]


__all__ = ["pairs", "table"]
