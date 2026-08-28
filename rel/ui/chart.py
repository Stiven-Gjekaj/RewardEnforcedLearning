"""Drawing a curve in a terminal, out of braille dots.

A braille character holds two columns of four dots, so a block of characters
gives eight times the resolution of a block of blocks. Forty characters across
and ten down is 80 by 40 dots, which is enough to see the shape of a learning
curve and small enough to read at a glance.

## Why not one of the block drawing characters

The eighth blocks give one column of eight steps per character. That is fine
for a bar chart and poor for a line, because a line that rises by less than an
eighth of a character in a column does not move at all, and a curve that is
flattening out is exactly a curve that rises by very little.

## Colour and overlapping series

A braille character carries dots from up to eight places at once, so two series
that cross share a character and only one colour can be given to it. The
character takes the colour of whichever series put the most dots in it.

With colour switched off the series still all appear, and which is which has to
be read off the legend. That is a real limit, and the alternative of drawing
each series in its own chart is offered by the caller rather than decided here.

## Nothing decorative hides the run

A reference mark and a band are both drawn behind the curves rather than merely
before them: a cell holding any real series takes that series' colour, a cell
holding only marks takes grey, and a cell holding only a band takes the band's
colour dimmed.

A band is its two edges rather than the filled area between them. Filled reads
better in colour and it is the same braille dot as a curve, so with colour
switched off it swallows the line it is the spread of. Two edges read the same
either way, and the answer stays visible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rel.metrics import moving_average
from rel.ui.colour import Palette

#: The bit each dot sets, indexed by column then row inside one character.
#:
#: The braille block numbers its dots 1, 2, 3, 7 down the left and 4, 5, 6, 8
#: down the right, which is a nineteenth century numbering with the fourth row
#: added a century later. That is why the last row of each column is not next
#: to the other three.
DOTS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))
BLANK = 0x2800

BLOCKS = " ▁▂▃▄▅▆▇█"

#: The series number a reference mark is drawn under. Negative, so that it can
#: never collide with a real series.
MARK = -1

#: Band `n` is drawn under `BAND - n`, so bands are -2 and below and never
#: collide with a mark or a series either.
BAND = -2


def sparkline(values: Sequence[float], width: int = 40) -> str:
    """A curve one line high, from the block characters.

    Good for a number in a table. For anything a reader has to judge a shape
    from, use `line_chart`.
    """
    if not values:
        return ""

    points = _resample(values, width)
    low = min(points)
    high = max(points)
    span = high - low

    if span <= 0.0:
        return BLOCKS[len(BLOCKS) // 2] * len(points)

    steps = len(BLOCKS) - 1
    return "".join(
        BLOCKS[min(steps, max(0, round((value - low) / span * steps)))]
        for value in points
    )


def line_chart(
    series: Mapping[str, Sequence[float]],
    *,
    width: int = 60,
    height: int = 12,
    palette: Palette | None = None,
    marks: Mapping[str, float] | None = None,
    bands: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    label_width: int = 9,
    title: str = "",
    clip: float = 0.0,
) -> list[str]:
    """A chart of one or more curves, as lines of text.

    `marks` draws a horizontal line at a value and names it, which is how the
    best possible return goes on the same picture as the run.

    `bands` draws the spread of a series behind it, as a lowest and a highest
    curve. A mean line on its own says nothing about how much the runs behind
    it disagreed, and in this project that has been the wrong answer twice.
    The two edges are drawn rather than the area between them, so the band
    reads the same with colour switched off.

    `clip` is a share of the values, from 0 to 1, to leave off the bottom of
    the range. A learning curve usually starts far below where it ends, so a
    chart drawn over the whole range spends nine tenths of its height on the
    first twenty episodes and squashes everything worth looking at into one
    row. Clipping puts the bottom of the axis above those episodes, draws them
    on the bottom row, and says so with a `<` on the label. Nothing is hidden:
    the label says the axis does not reach that far.
    """
    palette = palette or Palette(enabled=False)
    marks = marks or {}
    bands = bands or {}

    drawn = {name: values for name, values in series.items() if values}
    if not drawn:
        return ["(nothing to draw)"]

    # A band reaches further than the curve inside it, so the axis has to
    # cover it or the band would be drawn flat against the top and bottom.
    spread = {
        f"{name} band {edge}": points
        for name, pair in bands.items()
        if name in drawn
        for edge, points in zip(("low", "high"), pair, strict=True)
    }
    low, high = _bounds({**drawn, **spread}, marks, clip)
    below = any(value < low for points in drawn.values() for value in points)
    dot_columns = width * 2
    dot_rows = height * 4

    # Which series has put how many dots in each character cell, and the bits
    # that make up the character itself.
    bits = [[BLANK] * width for _ in range(height)]
    owners: list[list[dict[int, int]]] = [
        [{} for _ in range(width)] for _ in range(height)
    ]

    # The bands go on first and the curves last. What decides which colour a
    # shared cell takes is `_winner` rather than this order, and drawing in
    # this order keeps the two agreeing.
    for index, name in enumerate(drawn):
        pair = bands.get(name)
        if pair is None:
            continue
        for edge in pair:
            behind: int | None = None
            for column, value in enumerate(_resample(edge, dot_columns)):
                row = _dot_row(value, low, high, dot_rows)
                for filled in _between(behind, row):
                    _set(bits, owners, column, filled, BAND - index, width, height)
                behind = row

    # The marks go on next so that a curve crossing one is drawn over it.
    # A reference line that hides the run is worse than no reference line.
    for value in marks.values():
        row = _dot_row(value, low, high, dot_rows)
        for column in range(0, dot_columns, 4):
            _set(bits, owners, column, row, MARK, width, height)

    for index, (_name, values) in enumerate(drawn.items()):
        points = _resample(values, dot_columns)
        previous: int | None = None
        for column, value in enumerate(points):
            row = _dot_row(value, low, high, dot_rows)
            for filled in _between(previous, row):
                _set(bits, owners, column, filled, index, width, height)
            previous = row

    lines = []
    if title:
        lines.append(title)

    axis = _axis_labels(low, high, height, label_width, below)
    for row in range(height):
        cells = []
        for column in range(width):
            character = chr(bits[row][column])
            counts = owners[row][column]
            if character == chr(BLANK) or not counts:
                cells.append(" ")
                continue
            winner = _winner(counts)
            if winner == MARK:
                cells.append(palette.paint(character, "grey"))
            elif winner <= BAND:
                cells.append(palette.faint_series(character, BAND - winner))
            else:
                cells.append(palette.series(character, winner))
        lines.append(f"{axis[row]} |{''.join(cells)}")

    lines.append(" " * label_width + " +" + "-" * width)

    if len(drawn) > 1 or marks:
        legend = [palette.series(f"{name}", index) for index, name in enumerate(drawn)]
        legend += [
            palette.paint(f"{name} = {value:g}", "grey")
            for name, value in marks.items()
        ]
        lines.append(" " * (label_width + 2) + "   ".join(legend))

    return lines


def _winner(counts: Mapping[int, int]) -> int:
    """Which series a shared cell belongs to.

    A real series first, then a mark, then a band. Most dots decides only
    within one of those, because a band fills a whole column of dots and a
    curve fills one or two, and a picture of how sure the answer is must not
    cover the answer.
    """
    real = {key: count for key, count in counts.items() if key >= 0}
    if real:
        return max(real, key=lambda key: real[key])
    if MARK in counts:
        return MARK
    return max(counts, key=lambda key: counts[key])


def _bounds(
    series: Mapping[str, Sequence[float]],
    marks: Mapping[str, float],
    clip: float = 0.0,
) -> tuple[float, float]:
    values = [value for points in series.values() for value in points]

    if clip > 0.0 and len(values) > 4:
        ordered = sorted(values)
        low = ordered[min(len(ordered) - 1, int(len(ordered) * clip))]
    else:
        low = min(values)

    high = max(max(values), *marks.values()) if marks else max(values)
    low = min([low, *marks.values()]) if marks else low
    if high - low < 1e-12:
        # A flat line still has to be drawn somewhere. Putting it in the middle
        # of an invented range is honest, because the axis labels show that the
        # range is invented.
        pad = max(abs(high), 1.0) * 0.05
        return low - pad, high + pad
    return low, high


def _resample(values: Sequence[float], count: int) -> list[float]:
    """Stretch or squash a series onto exactly `count` points.

    Squashing takes the mean of each block rather than every nth value.
    Sampling would show whichever episodes happened to land on a sample, and a
    learning curve is noisy enough that the picture would change with the
    number of columns rather than with the run.
    """
    if count <= 0:
        return []
    if len(values) == count:
        return list(values)

    if len(values) < count:
        return [
            float(values[min(len(values) - 1, index * len(values) // count)])
            for index in range(count)
        ]

    points = []
    for index in range(count):
        start = index * len(values) // count
        stop = max(start + 1, (index + 1) * len(values) // count)
        block = values[start:stop]
        points.append(sum(block) / len(block))
    return points


def _dot_row(value: float, low: float, high: float, rows: int) -> int:
    """The dot row a value belongs on, held inside the chart.

    A value below the axis is drawn on the bottom row rather than dropped. A
    curve with a hole in it reads as missing data.
    """
    share = (value - low) / (high - low)
    row = round((1.0 - share) * (rows - 1))
    return min(max(row, 0), rows - 1)


def _between(previous: int | None, row: int) -> range:
    """Every dot row to fill so that a steep step stays a connected line.

    Without this a curve that jumps looks like a scatter of dots, and a reader
    cannot tell a jump from a gap in the data.
    """
    if previous is None:
        return range(row, row + 1)
    if previous == row:
        return range(row, row + 1)
    return range(min(previous, row), max(previous, row) + 1)


def _set(
    bits: list[list[int]],
    owners: list[list[dict[int, int]]],
    column: int,
    row: int,
    series: int,
    width: int,
    height: int,
) -> None:
    cell_column, inside_column = divmod(column, 2)
    cell_row, inside_row = divmod(row, 4)
    if not (0 <= cell_column < width and 0 <= cell_row < height):
        return

    bits[cell_row][cell_column] |= DOTS[inside_column][inside_row]
    counts = owners[cell_row][cell_column]
    counts[series] = counts.get(series, 0) + 1


def _axis_labels(
    low: float, high: float, height: int, width: int, below: bool = False
) -> list[str]:
    """A label on the top row, the bottom row and the middle one.

    The bottom label carries a `<` when the axis does not reach the lowest
    value in the data, so a reader is never shown a floor that is not the
    floor.
    """
    labels = [" " * width] * height
    middle = height // 2

    for row, value in ((0, high), (height - 1, low), (middle, (low + high) / 2)):
        labels[row] = _short(value).rjust(width)

    if below:
        labels[height - 1] = ("<" + _short(low)).rjust(width)
    return labels


def _short(value: float) -> str:
    """A number in as few characters as it can be read in."""
    if value == 0:
        return "0"
    size = abs(value)
    if size >= 100_000 or size < 0.001:
        return f"{value:.1e}"
    if size >= 1000:
        return f"{value:.0f}"
    if size >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def bar(value: float, low: float, high: float, width: int = 20) -> str:
    """A horizontal bar, to an eighth of a character."""
    if high <= low:
        return " " * width
    share = min(max((value - low) / (high - low), 0.0), 1.0)

    eighths = round(share * width * 8)
    full, part = divmod(eighths, 8)
    drawn = "█" * full
    if part:
        drawn += BLOCKS[part]
    return drawn.ljust(width)


def histogram(values: Sequence[float], buckets: int = 10, width: int = 24) -> list[str]:
    """A count of how often each range of values came up."""
    if not values:
        return []

    low = min(values)
    high = max(values)
    if high <= low:
        return [f"{_short(low)} {'█' * width} {len(values)}"]

    counts = [0] * buckets
    for value in values:
        index = min(buckets - 1, int((value - low) / (high - low) * buckets))
        counts[index] += 1

    tallest = max(counts)
    lines = []
    for index, count in enumerate(counts):
        edge = low + (high - low) * index / buckets
        lines.append(f"{_short(edge):>9} {bar(count, 0, tallest, width)} {count:>6}")
    return lines


def smooth(values: Sequence[float], window: int) -> list[float]:
    """A moving mean, for a curve that is too noisy to read.

    This is `rel.metrics.moving_average`. It was written out here once, with a
    note saying that one line was not worth an import, and that was wrong: two
    copies of a rule are two places for it to be changed.
    """
    return moving_average(values, window)


def spread_of(values: Sequence[float]) -> float:
    """The standard deviation, for a caption under a chart."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


__all__ = [
    "BLOCKS",
    "DOTS",
    "MARK",
    "bar",
    "histogram",
    "line_chart",
    "smooth",
    "sparkline",
    "spread_of",
]
