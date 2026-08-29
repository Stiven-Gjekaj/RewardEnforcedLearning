"""How an agent turns the numbers it keeps into the action it takes.

Every tabular agent in this project explored the same way: take the best
action, except that with probability epsilon take one at random. That rule is
the default because it is the one the textbook derivations assume, and it is
the weakest part of the agents that use it. It explores by ignoring everything
the agent knows, so a mistake it has made a thousand times and a move it has
never tried are equally likely to be the one it tries.

Writing the rule out as an object rather than as four lines inside `act` puts
the alternatives beside it. It also puts them in one place, so an agent that
reasons about its own policy keeps reasoning about the right one: expected
SARSA, tree backup and off-policy Monte Carlo all read
`policy_probabilities`, and every rule here answers that question exactly
rather than approximately.

## Every rule answers two questions

`choose` says what to do now. `probabilities` says how likely each action was,
which is what an agent correcting for its own exploration needs. A rule that
answered only the first would leave the agents above computing the wrong
correction, and they would still run.

## The draws are counted

`EpsilonGreedy` draws exactly as the old code did: one `chance`, then either
one `below` over the actions or the tie-break inside `argmax`. That is not a
detail. The seed reaches every part of a run here, so a rule that spent one
extra draw would move every number in the documentation while computing the
same policy.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from rel.rng import Rng
from rel.schedules import Schedule, as_schedule

#: What a rule is told about the counts, when it asked to be told.
Counts = Sequence[int] | None


def argmax(rng: Rng, scores: Sequence[float]) -> int:
    """The index of the best score, with ties broken at random.

    An untouched table holds the same number everywhere, so every action ties.
    Taking the first one turns the starting policy into "always go up", and on
    a grid that is not a neutral start: it explores one direction far more than
    the others for as long as the tie lasts.

    The draw happens only when there is a tie, which is what the tabular agents
    have always done.
    """
    best = max(scores)
    tied = [index for index, value in enumerate(scores) if value == best]
    if len(tied) == 1:
        return tied[0]
    return tied[rng.below(len(tied))]


def greedy_probabilities(scores: Sequence[float]) -> list[float]:
    """The distribution of a rule that always takes the best action.

    The tied actions share the whole of it. This is what breaking a tie at
    random means, written as a distribution rather than as a draw.
    """
    best = max(scores)
    tied = [index for index, value in enumerate(scores) if value == best]
    share = 1.0 / len(tied)

    probabilities = [0.0] * len(scores)
    for index in tied:
        probabilities[index] = share
    return probabilities


class Rule(ABC):
    """A way of choosing an action from the scores of the actions."""

    #: Whether the agent has to keep a visit count for each state and action.
    #:
    #: Counting is not free. It is a second dictionary the same size as the
    #: table, written on every step, and only one rule here reads it. An agent
    #: asks its rule rather than counting always.
    needs_counts = False

    @abstractmethod
    def choose(
        self,
        rng: Rng,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> int:
        """The index of the action to take now.

        The index is into the scores. The caller adds the start of the action
        space, so a rule never needs to know that actions might not start at
        zero.
        """

    @abstractmethod
    def probabilities(
        self,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> list[float]:
        """How likely `choose` is to return each index, exactly.

        Exactly, not approximately. Expected SARSA takes the average over this
        instead of a sample, and an off-policy correction divides by it. An
        approximation here is a bias there, and it would be a bias nothing in
        the output would show.
        """


class EpsilonGreedy(Rule):
    """The best action, except that now and then take any action.

    The one dial is how often "now and then" is. It can be a schedule, so a run
    can explore hard at the start and stop later.

    This rule is the default for every tabular agent, and the numbers in the
    documentation were measured with it.
    """

    __slots__ = ("epsilon",)

    def __init__(self, epsilon: float | Schedule = 0.1) -> None:
        self.epsilon = as_schedule(epsilon)

    def choose(
        self,
        rng: Rng,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> int:
        if rng.chance(self.epsilon(episodes)):
            return rng.below(len(scores))
        return argmax(rng, scores)

    def probabilities(
        self,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> list[float]:
        # Written out rather than built on `greedy_probabilities`, because
        # `(1 - epsilon) / tied` and `(1 - epsilon) * (1 / tied)` are not the
        # same float. They differ for five tied actions at an epsilon of 0.1.
        # No environment here has five actions, so nothing would have moved,
        # and a rule whose output depends on which of two equal expressions
        # was typed is not a rule anybody should have to check.
        epsilon = self.epsilon(episodes)
        count = len(scores)
        best = max(scores)
        tied = [index for index, value in enumerate(scores) if value == best]

        share = epsilon / count
        probabilities = [share] * count
        for index in tied:
            probabilities[index] += (1.0 - epsilon) / len(tied)
        return probabilities

    def __repr__(self) -> str:
        return f"EpsilonGreedy(epsilon={self.epsilon(0):g})"


class Softmax(Rule):
    """Every action, with a probability that rises with what it is worth.

    Epsilon-greedy explores by ignoring what the agent knows, so the second
    best action and the worst one are equally likely to be the one it tries.
    This ranks them instead. An action worth a little less than the best comes
    up often and one worth far less comes up rarely.

    The dial is the temperature. A high one flattens the ordering towards
    uniform and a low one sharpens it towards greedy. It can be a schedule, and
    cooling is the usual way to use it.

    ## The temperature is in the units of the value

    This is the weakness of the rule and it is not a small one. Epsilon means
    the same thing on every problem: it is a probability. A temperature does
    not. It divides a difference between two values, so what counts as hot
    depends on how far apart the values of that environment are.

    On the cliff walk the values run to about -100 and neighbouring actions
    differ by ones, so a temperature of one already follows the ordering
    closely and a temperature of 0.1 is greedy in all but name. On a grid where
    every reward is 0 or 1 the same two numbers are almost uniform. A setting
    carried from one environment to another is not the same setting.
    """

    __slots__ = ("temperature",)

    def __init__(self, temperature: float | Schedule = 1.0) -> None:
        self.temperature = as_schedule(temperature)

    def choose(
        self,
        rng: Rng,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> int:
        shares = self.probabilities(scores, counts, episodes, steps)
        return rng.weighted_index(shares)

    def probabilities(
        self,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> list[float]:
        temperature = self.temperature(episodes)
        if temperature < 0.0:
            raise ValueError("A temperature is zero or above.")
        if temperature == 0.0:
            # The limit rather than a division by zero. A schedule that cools
            # to nothing should end greedy, not raise on its last episode.
            return greedy_probabilities(scores)

        # Every exponent is zero or below, so nothing here can overflow. The
        # far ends underflow to zero instead, which is the right answer: an
        # action worth forty less at a temperature of one is not going to be
        # taken.
        best = max(scores)
        weights = [math.exp((score - best) / temperature) for score in scores]
        total = sum(weights)
        return [weight / total for weight in weights]

    def __repr__(self) -> str:
        return f"Softmax(temperature={self.temperature(0):g})"


class CountBonus(Rule):
    """The best action, where best counts how little the action has been tried.

    Each action gets its value plus a term that grows with how long it has been
    since it was taken here and shrinks as it is taken more. An action never
    taken in this state scores infinity, so the first few visits to a state try
    everything once before ranking anything.

    Nothing here is random except a tie. That is the point: epsilon-greedy and
    softmax explore by putting chance into the choice, and this explores by
    putting the reason to explore into the score.

    ## This is the bandit rule, once per state

    `UpperConfidenceBandit` does this over the levers of one bandit, where the
    guarantee it is famous for holds. Treating each state as its own bandit
    does not carry the guarantee across, and the reason is worth being plain
    about: a bandit's arms pay out on their own, and a grid's actions pay out
    through the states they lead to.

    So this drives the agent to try every action of the states it stands in,
    and nothing in it drives the agent towards a state it has never reached.
    Once every action of a state has been taken a few times the bonus there is
    small and the agent is greedy again, whatever is still unexplored two rooms
    away. Optimistic initialisation is the rule in this file that does the
    other thing, and `docs/algorithms.md` measures the two side by side.

    ## What it costs

    A second dictionary the size of the table, written on every step. Nothing
    else here keeps one, which is why `needs_counts` exists.
    """

    __slots__ = ("confidence",)

    needs_counts = True

    def __init__(self, confidence: float = 2.0) -> None:
        if confidence < 0.0:
            raise ValueError("A confidence is zero or above.")
        self.confidence = confidence

    def bonused(self, scores: Sequence[float], counts: Counts) -> list[float]:
        """The scores with the exploration bonus added to each."""
        if counts is None:
            raise ValueError(
                "This rule explores by counting and was given no counts. "
                "The agent reads `needs_counts` to decide whether to keep them."
            )

        # The count of visits to this state, which is what each action's own
        # count is being compared against. The plus one is so that the first
        # visit takes the log of one rather than of zero.
        span = math.log(sum(counts) + 1)
        return [
            math.inf
            if count == 0
            else score + self.confidence * math.sqrt(span / count)
            for score, count in zip(scores, counts, strict=True)
        ]

    def choose(
        self,
        rng: Rng,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> int:
        return argmax(rng, self.bonused(scores, counts))

    def probabilities(
        self,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> list[float]:
        return greedy_probabilities(self.bonused(scores, counts))

    def __repr__(self) -> str:
        return f"CountBonus(confidence={self.confidence:g})"


__all__ = [
    "CountBonus",
    "Counts",
    "EpsilonGreedy",
    "Rule",
    "Softmax",
    "argmax",
    "greedy_probabilities",
]
