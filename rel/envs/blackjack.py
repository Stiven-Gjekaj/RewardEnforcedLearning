"""Blackjack against a dealer who plays by a rule, with the model written out.

Sutton and Barto, example 5.1. The player holds cards and decides whether to
draw another. Going past 21 loses at once. Sticking hands over to the dealer,
who draws while under 17 and then stops, and the higher total that is not past
21 wins.

The book uses this to introduce Monte Carlo, because "the dealer then plays out
his whole hand" is awkward to write as a one step model and easy to sample.
This writes it out anyway.

**Which is the point of having it here.** An environment that knows its own
model can be solved exactly, and an agent that learns from episodes can then be
scored against the answer rather than against another agent. Every other
episodic environment in this project is a grid, where the model is obvious.
This one is a card game where it is not, and the value of a Monte Carlo agent
is exactly the thing it lets a reader see.

## The cards

Drawn with replacement, which is the book's assumption: an infinite deck, so
the chance of each card does not depend on what has gone before. Ten, jack,
queen and king are all worth ten, so a ten comes up four times in thirteen and
every other card once.

An ace is worth eleven where that does not go past 21, and one where it does.
A hand holding an ace counted as eleven is called soft, and a soft hand that
goes past 21 becomes hard rather than losing.

## The states

`(sum, showing, soft)`: what the player holds, the dealer's face up card, and
whether the player's hand is soft. The sum runs from 12 to 21, because below
12 no card can lose and drawing is free, so the deal simply carries on until
the player is at 12 or more. Ten cards the dealer can show and two states of
the ace give 200, and one more state for a hand that is over.

## Two things the book has and this does not

**A natural pays nothing extra.** Twenty one on the first two cards is a
natural, and the book pays it before the hand is played. Here it is a hand of
21 like any other. The rule changes what the deal is worth and does not change
what to do, because the only move at 21 is to stick either way, so the policy
this solves for is the book's and the value of the deal is not.

**Doubling and splitting are not here.** The book leaves them out too, and the
two actions are the whole of what it measures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache

from rel.core import NO_INFO, EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete

#: Every card and how often it comes up, from an infinite deck. An ace is a
#: one here and is counted as eleven where that fits.
CARDS: tuple[tuple[int, float], ...] = tuple(
    (card, 4.0 / 13.0 if card == 10 else 1.0 / 13.0) for card in range(1, 11)
)

#: The shares alone, for drawing. Kept beside `CARDS` rather than built on
#: every draw, because a hand is a handful of draws and a run is millions.
SHARES: tuple[float, ...] = tuple(share for _, share in CARDS)

#: The lowest sum the player is ever asked about. Below it no card can lose,
#: so the deal carries on rather than asking.
LOWEST = 12

#: The sum that stands for a hand past 21, for the dealer's tally.
BUST = 22

STICK = 0
HIT = 1


def drawn(total: int, soft: bool, card: int) -> tuple[int, bool]:
    """The hand after one more card, with the ace counted whichever way fits.

    An ace is eleven where that does not go past 21. A hand that goes past 21
    while holding an ace counted as eleven counts it as one instead, which is
    what makes a soft hand survive a card that a hard one would lose to.
    """
    if card == 1 and total + 11 <= 21:
        total, soft = total + 11, True
    else:
        total += card

    if total > 21 and soft:
        total, soft = total - 10, False
    return total, soft


@cache
def dealer_totals(total: int, soft: bool) -> tuple[tuple[int, float], ...]:
    """What the dealer ends on, from a hand of this much.

    Drawn while under 17 and stopped at 17 or more, soft or not, which is the
    book's rule. A hand past 21 is reported as `BUST`.
    """
    if total >= 17:
        return ((min(total, BUST), 1.0),)

    ends: dict[int, float] = {}
    for card, chance in CARDS:
        after, still_soft = drawn(total, soft, card)
        for end, rest in dealer_totals(after, still_soft):
            ends[end] = ends.get(end, 0.0) + chance * rest
    return tuple(sorted(ends.items()))


@cache
def player_start() -> tuple[tuple[tuple[int, bool], float], ...]:
    """The hands the player is first asked about, and how often each comes up.

    Cards are dealt until the sum is 12 or more. That cannot go past 21: the
    largest hand below 12 is eleven, and the largest card is an ace counted as
    eleven, which would be 22 and is counted as one instead.
    """
    hands: dict[tuple[int, bool], float] = {}

    def deal(total: int, soft: bool, chance: float) -> None:
        if total >= LOWEST:
            hands[total, soft] = hands.get((total, soft), 0.0) + chance
            return
        for card, share in CARDS:
            after, still_soft = drawn(total, soft, card)
            deal(after, still_soft, chance * share)

    deal(0, False, 1.0)
    return tuple(sorted(hands.items()))


class Blackjack(TabularEnv):
    """Draw or stick against a dealer who draws while under 17."""

    def __init__(self, rng: Rng) -> None:
        super().__init__(rng)

        #: One more than the hands that can be held, for the hand that is over.
        self.over = 200
        self.observation_space = Discrete(201)
        self.action_space = Discrete(2)
        self.action_names = ("stick", "hit")
        self.spec = EnvSpec(
            name="blackjack",
            summary=(
                "Draw or stick against a dealer who draws while under 17. "
                "Winning pays 1, a draw nothing and losing -1."
            ),
            max_episode_steps=20,
        )

        self.total = LOWEST
        self.soft = False
        self.showing = 1
        self.finished = False
        self._hands = 0
        self._won = 0.0

    # -- Reading a state ----------------------------------------------------

    def fold(self, total: int, showing: int, soft: bool) -> int:
        """The state number for a hand, a face up card and a soft ace."""
        return ((total - LOWEST) * 10 + (showing - 1)) * 2 + int(soft)

    def unfold(self, state: int) -> tuple[int, int, bool]:
        """The hand, the face up card and the soft ace of a state number."""
        rest, soft = divmod(state, 2)
        hand, showing = divmod(rest, 10)
        return hand + LOWEST, showing + 1, bool(soft)

    def paid(self, total: int, dealer: int) -> float:
        """What a stick with this hand pays against this dealer tally."""
        if dealer == BUST or total > dealer:
            return 1.0
        return 0.0 if total == dealer else -1.0

    # -- The model ----------------------------------------------------------

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if state == self.over:
            return (Outcome(1.0, self.over, 0.0, True),)

        total, showing, soft = self.unfold(state)
        if action == STICK:
            # The dealer plays his whole hand out, so a stick is one step
            # here whatever it is at the table.
            paid: dict[float, float] = {}
            for end, chance in dealer_totals(*drawn(0, False, showing)):
                got = self.paid(total, end)
                paid[got] = paid.get(got, 0.0) + chance
            return tuple(
                Outcome(chance, self.over, got, True) for got, chance in paid.items()
            )

        landings: dict[tuple[int, float, bool], float] = {}
        for card, chance in CARDS:
            after, still_soft = drawn(total, soft, card)
            if after > 21:
                branch = (self.over, -1.0, True)
            else:
                branch = (self.fold(after, showing, still_soft), 0.0, False)
            landings[branch] = landings.get(branch, 0.0) + chance
        return tuple(
            Outcome(chance, landed, got, ended)
            for (landed, got, ended), chance in landings.items()
        )

    def start_states(self) -> Sequence[tuple[float, int]]:
        starts = []
        for (total, soft), chance in player_start():
            for card, share in CARDS:
                starts.append((chance * share, self.fold(total, card, soft)))
        return tuple(starts)

    def terminal_states(self) -> frozenset[int]:
        """The hand that is over, and nothing else.

        Named rather than read off the model. Every branch out of a hand of 21
        ends the hand, so a model reader would call 21 an ending and never
        sweep it, and 21 is the hand worth the most.
        """
        return frozenset({self.over})

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        # Cards until the sum is 12 or more, which is the deal `player_start`
        # lists. Below 12 no card can lose, so nothing is asked of the player
        # until then.
        self.total, self.soft = 0, False
        while self.total < LOWEST:
            self.total, self.soft = drawn(self.total, self.soft, self._card())
        self.showing = self._card()
        self.finished = False
        return self.fold(self.total, self.showing, self.soft)

    def _card(self) -> int:
        """One card, drawn from the shares the model lists."""
        return CARDS[self.rng.weighted_index(SHARES)][0]

    def _step(self, action: int) -> Step[int]:
        if action == STICK:
            dealer, soft = drawn(0, False, self.showing)
            while dealer < 17:
                dealer, soft = drawn(dealer, soft, self._card())
            got = self.paid(self.total, min(dealer, BUST))
            return self._over(got)

        self.total, self.soft = drawn(self.total, self.soft, self._card())
        if self.total > 21:
            return self._over(-1.0)
        return Step(
            observation=self.fold(self.total, self.showing, self.soft),
            reward=0.0,
            terminated=False,
            truncated=False,
        )

    def _over(self, got: float) -> Step[int]:
        self.finished = True
        self._hands += 1
        self._won += got
        return Step(observation=self.over, reward=got, terminated=True, truncated=False)

    # -- What the run really did --------------------------------------------

    def audit(self) -> Mapping[str, float]:
        """The share of hands won, which the reward already says.

        Here because a return averages a win, a draw and a loss into one
        number, and a policy that draws often is not the same as one that wins
        as often.
        """
        if self._hands == 0:
            return NO_INFO
        return {"hands": float(self._hands), "won_a_hand": self._won / self._hands}

    def render(self) -> str:
        if self.finished:
            return "the hand is over"
        ace = " with a soft ace" if self.soft else ""
        return f"player {self.total}{ace}, dealer shows {self.showing}"


__all__ = [
    "BUST",
    "CARDS",
    "HIT",
    "LOWEST",
    "STICK",
    "Blackjack",
    "dealer_totals",
    "drawn",
    "player_start",
]
