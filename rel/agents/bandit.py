"""Agents for a row of levers, where there is no state and no future.

Everything here answers one question: pull the lever that has paid best, or
pull one that has not been tried enough to know? Four answers, and they differ
in what they do with the not knowing.

    epsilon greedy   ignores it, and wanders at random now and then
    optimistic       expects too much of everything, so it tries everything
    upper confidence picks the lever with the highest plausible value
    gradient         keeps preferences rather than values, and pushes them

## Every episode is a new problem

`start_episode` clears what the agent has learned.

An episode of the bandit environment is one set of levers, drawn fresh, and an
agent that carried its estimates into the next episode would be answering a
question about levers that no longer exist. A run of two hundred episodes is
therefore two hundred independent problems, which is the average that the
chapter plots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rel.agents.base import Agent, Transition
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule
from rel.spaces import Discrete


class BanditAgent(Agent[int]):
    """The parts that every lever puller shares."""

    def __init__(self, rng: Rng, actions: Discrete, *, optimism: float = 0.0) -> None:
        super().__init__(rng, actions)
        self.optimism = optimism
        self.estimates: list[float] = [optimism] * actions.n
        self.counts: list[int] = [0] * actions.n
        self.pulls = 0

    def start_episode(self) -> None:
        self.estimates = [self.optimism] * self.actions.n
        self.counts = [0] * self.actions.n
        self.pulls = 0

    def action_values(self, observation: int) -> Sequence[float]:
        return self.estimates

    def greedy(self, observation: int) -> int:
        return self.actions.start + self._argmax(self.estimates)

    def _argmax(self, scores: Sequence[float]) -> int:
        best = max(scores)
        tied = [index for index, score in enumerate(scores) if score == best]
        if len(tied) == 1:
            return tied[0]
        return tied[self.rng.below(len(tied))]

    def _record(self, transition: Transition[int, int]) -> int:
        """Count the pull and give back the index of the lever."""
        index = transition.action - self.actions.start
        self.counts[index] += 1
        self.pulls += 1
        return index


class EpsilonGreedyBandit(BanditAgent):
    """Pulls the best lever so far, and a random one now and then.

    With `step_size` left as `None` the estimate is the average of every reward
    that lever has paid. That is right while the levers stay put and wrong as
    soon as they move: the thousandth reward counts for a thousandth, so the
    estimate cannot follow anything.

    With a number, the estimate keeps a fixed weight on the newest reward, so
    it forgets at a fixed rate and can follow a lever that drifts. On the
    drifting bandit the two come apart plainly, and the reward curve shows it
    only faintly. The share of best choices shows it at once.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        epsilon: float | Schedule = 0.1,
        step_size: float | Schedule | None = None,
        optimism: float = 0.0,
    ) -> None:
        super().__init__(rng, actions, optimism=optimism)
        self.epsilon = as_schedule(epsilon)
        self.averaging = step_size is None
        self.step_size = as_schedule(0.0 if step_size is None else step_size)

    def act(self, observation: int) -> int:
        if self.rng.chance(self.epsilon(self.pulls)):
            return self.actions.start + self.rng.below(self.actions.n)
        return self.greedy(observation)

    def observe(self, transition: Transition[int, int]) -> None:
        super().observe(transition)
        index = self._record(transition)
        error = transition.reward - self.estimates[index]
        if self.averaging:
            self.estimates[index] += error / self.counts[index]
        else:
            self.estimates[index] += self.step_size(self.pulls) * error


class OptimisticBandit(EpsilonGreedyBandit):
    """Expects far too much of every lever, and is disappointed into trying all.

    No random exploration at all. Every lever starts above anything it can
    really pay, so pulling one lowers its estimate below the untried ones and
    the agent moves on. By the time it has been round them all it knows
    something about each.

    It explores hard at the start and then stops, which is exactly right for a
    problem that stays still and useless for one that moves. A lever that
    becomes the best one after the first hundred pulls is never looked at
    again.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        optimism: float = 5.0,
        step_size: float | Schedule = 0.1,
    ) -> None:
        super().__init__(
            rng, actions, epsilon=0.0, step_size=step_size, optimism=optimism
        )


class UpperConfidenceBandit(BanditAgent):
    """Pulls the lever with the highest value it could plausibly have.

    The score is the estimate plus a term that grows with how long it has been
    since the lever was tried and shrinks as it is tried more. A lever that has
    never been pulled has an infinite score, so the first sweep tries them all.

    This explores where exploring is worth something, rather than uniformly.
    The cost is that it needs a count for each lever, which is why it does not
    carry over to a problem with states.
    """

    def __init__(self, rng: Rng, actions: Discrete, *, confidence: float = 2.0) -> None:
        super().__init__(rng, actions)
        self.confidence = confidence

    def act(self, observation: int) -> int:
        untried = [index for index, count in enumerate(self.counts) if count == 0]
        if untried:
            return self.actions.start + untried[self.rng.below(len(untried))]

        span = math.log(self.pulls + 1)
        scores = [
            estimate + self.confidence * math.sqrt(span / count)
            for estimate, count in zip(self.estimates, self.counts, strict=True)
        ]
        return self.actions.start + self._argmax(scores)

    def observe(self, transition: Transition[int, int]) -> None:
        super().observe(transition)
        index = self._record(transition)
        self.estimates[index] += (
            transition.reward - self.estimates[index]
        ) / self.counts[index]


class GradientBandit(BanditAgent):
    """Keeps a preference for each lever rather than an estimate of its value.

    The preferences are turned into probabilities by a softmax, and after each
    pull the preference of the lever that was pulled goes up by how much the
    reward beat the average so far, while every other preference goes down.

    Nothing here estimates what a lever pays. A preference is not a value and
    cannot be read as one: adding the same number to all of them changes
    nothing at all. What it learns is an ordering.

    The baseline is the average reward so far. Without it the agent still
    learns, and it learns far more slowly whenever the rewards are all large
    and positive, because then every pull looks good and the preferences all
    rise together.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        step_size: float | Schedule = 0.1,
        baseline: bool = True,
    ) -> None:
        super().__init__(rng, actions)
        self.step_size = as_schedule(step_size)
        self.baseline = baseline
        self.average = 0.0

    def start_episode(self) -> None:
        super().start_episode()
        self.average = 0.0

    def probabilities(self) -> list[float]:
        """The softmax over the preferences.

        The largest preference is taken off before the exponential. That
        changes nothing about the answer and stops `exp` overflowing, which it
        does once a preference passes about seven hundred.
        """
        largest = max(self.estimates)
        weights = [math.exp(value - largest) for value in self.estimates]
        total = sum(weights)
        return [weight / total for weight in weights]

    def act(self, observation: int) -> int:
        return self.actions.start + self.rng.weighted_index(self.probabilities())

    def greedy(self, observation: int) -> int:
        return self.actions.start + self._argmax(self.estimates)

    def observe(self, transition: Transition[int, int]) -> None:
        super().observe(transition)
        shares = self.probabilities()
        index = self._record(transition)

        advantage = transition.reward - (self.average if self.baseline else 0.0)
        self.average += (transition.reward - self.average) / self.pulls

        rate = self.step_size(self.pulls)
        for lever in range(self.actions.n):
            taken = 1.0 if lever == index else 0.0
            self.estimates[lever] += rate * advantage * (taken - shares[lever])


__all__ = [
    "BanditAgent",
    "EpsilonGreedyBandit",
    "GradientBandit",
    "OptimisticBandit",
    "UpperConfidenceBandit",
]
