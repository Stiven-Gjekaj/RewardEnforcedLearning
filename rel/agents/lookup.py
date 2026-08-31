"""Features that are looked up rather than worked out.

A tile coder and a radial basis both compute their features from a point. This
one does not compute anything: it is handed a table with one row for each
state and it returns the row.

That sounds like it gives up the whole benefit of function approximation, and
for a table of one-hot rows it does: that is a tabular agent with extra steps.
The benefit comes back the moment two states share a row, or share part of
one. Then what is learned about either of them moves the other, which is
generalisation, and the table says exactly which states generalise to which.

## What it is for

Two things in this project, and they pull in opposite directions.

**State aggregation** groups nearby states into one feature and is the
friendliest approximation there is. A thousand states in ten groups is ten
weights, every state in a group shares all of its learning, and the estimate is
the average of the group.

**Baird's counterexample** is a table chosen so that off-policy bootstrapping
over it diverges. Seven states, eight weights, and enough overlap between the
rows that an update meant to lower one state's estimate raises it. The table is
the counterexample: the environment pays nothing anywhere, so nothing else in
it can be blamed.

Both are the same class because both are the same thing, and having the friendly
case and the hostile one built out of one object is the point. An approximation
is not safe or unsafe on its own. What it is used with decides.

## Zeros are dropped

A row is stored as the features that are not zero and their values, which is
what `Coder` asks for. For state aggregation that is one feature out of ten and
the saving is real. For a dense row it is the whole row and the saving is
nothing, which costs one pass at construction and never again.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.agents.linear import Encoded


class Lookup:
    """A row of features for each state, handed in rather than computed."""

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        if not rows:
            raise ValueError("A lookup table needs a row for at least one state.")

        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise ValueError(
                f"Every row is a weight for the same features, so every row is "
                f"the same length. These are {sorted(widths)}."
            )
        if widths == {0}:
            raise ValueError("A lookup table needs at least one feature.")

        self.rows: tuple[tuple[float, ...], ...] = tuple(
            tuple(float(value) for value in row) for row in rows
        )
        self._encoded: tuple[Encoded, ...] = tuple(
            (
                tuple(index for index, value in enumerate(row) if value != 0.0),
                tuple(value for value in row if value != 0.0),
            )
            for row in self.rows
        )

    @property
    def features(self) -> int:
        """How many weights an agent needs for each action."""
        return len(self.rows[0])

    @property
    def states(self) -> int:
        """How many states the table has a row for."""
        return len(self.rows)

    def encode(self, observation: int) -> Encoded:
        """The row for this state, with its zeros dropped."""
        if not 0 <= observation < len(self._encoded):
            raise IndexError(
                f"{observation} is outside a table of {len(self._encoded)} states."
            )
        return self._encoded[observation]

    def squared_length(self, values: Sequence[float]) -> float:
        """The features of a state dotted with themselves, for the step size."""
        return sum(value * value for value in values)

    def starting_weight(self, optimism: float) -> float:
        """The weight that makes a state nothing is known about worth this.

        A tile coder can always answer this, because every point lights the
        same number of features. A table cannot in general: two rows that add
        up to different numbers reach different values from the same weight, so
        there is no one weight that makes every state worth `optimism`.

        So it answers when the rows agree and raises when they do not, rather
        than returning a number that is right for some states. Zero is always
        answerable, and it is the answer every caller in this project asks for.
        """
        if optimism == 0.0:
            return 0.0

        totals = {sum(row) for row in self.rows}
        if len(totals) != 1:
            raise ValueError(
                "The rows of this table do not add up to the same number, so "
                "no single weight makes every state worth the same. An "
                "optimistic start needs a table whose rows agree."
            )

        total = totals.pop()
        if total == 0.0:
            raise ValueError("A row of zeros is worth zero whatever the weights are.")
        return optimism / total

    def __repr__(self) -> str:
        return f"Lookup(states={self.states}, features={self.features})"


def aggregated(states: int, groups: int) -> Lookup:
    """One weight for each group of states, and every state in exactly one.

    The friendly use of this class, and the smallest approximation there is.
    A thousand states in ten groups is ten weights: every state in a group
    shares all of its learning with the others, and the estimate of any of
    them is the estimate of the group.

    ## Where the boundaries go

    State `s` of `n` goes into group `s * groups // n`, so the groups are as
    even as whole numbers allow and the remainder is spread rather than piled
    onto the last one. Eleven states in three groups gives four, four and
    three, and not four, four and three the other way round.

    ## What it can and cannot say

    The estimate of a group is one number, so the value function it can write
    down is a staircase. On a walk whose true values run in a straight line
    the best a staircase can do is a known amount of error, and how much
    depends only on how many steps it has. `scripts/measure_aggregation.py`
    measures the error an agent reaches against the error the staircase cannot
    do better than.
    """
    if states < 1:
        raise ValueError("There is nothing to group.")
    if not 1 <= groups <= states:
        raise ValueError(
            f"{groups} groups for {states} states. There is at least one "
            f"group and no more of them than there are states."
        )

    rows = []
    for state in range(states):
        row = [0.0] * groups
        row[state * groups // states] = 1.0
        rows.append(row)
    return Lookup(rows)


__all__ = ["Lookup", "aggregated"]
