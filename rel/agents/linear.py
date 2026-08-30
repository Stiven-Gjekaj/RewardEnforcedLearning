"""Agents that keep a weight for each switch instead of a row for each state.

The value of an action is the sum of the weights of the switches that are on.
There is no table of states, so an environment with real numbered observations
works, and two nearby observations share most of their switches and therefore
most of what has been learned about either of them.

## Semi-gradient

The update moves the weights towards the same target the tabular agents use,
by the gradient of the value with respect to the weights. For a sum over
switches that gradient is one at each active switch and zero everywhere else,
so the update adds the same number to each active weight.

It is called semi-gradient because the target contains the value of the next
state, which also depends on the weights, and that dependence is ignored.
Following it as well is possible and it is not what these methods do. Ignoring
it is what makes the update cheap and it is why these methods have no
convergence guarantee off-policy.

## The step size

A step size is divided by the features of the point, dotted with themselves.

The reason is arithmetic rather than taste. The value is a sum over the active
features, so adding `a` to each of eight active weights moves the value by
`8a`. A step size of 0.5 with eight grids would move the value four times the
error and the weights would grow without limit. `step_size` here means the
share of the error the value moves by, which is the same thing it means for the
tabular agents, and the division happens inside.

Each coder answers for itself, in `squared_length`. For a tile coder the answer
is the number of grids whatever the point is. For a radial basis it changes
from point to point, because the values do.

## Two coders, one agent

The agent asks a coder for two things: which features are on, and how strongly.
A tile coder answers with a one for each switch, because a switch is on or off.
A radial basis answers with how close the point is to each centre.

Nothing about the agent knows which it is talking to. `Coder` below is the
whole of what it needs, and both satisfy it without either one importing the
other or this module.

**A run over a tile coder gives the same numbers it gave before this was
generalised, to the last bit.** Multiplying a weight by one is exact, adding
zero terms in the same order is exact, and a step size divided by `grids` is
the same float whether `grids` arrived as an attribute or as an answer. The
digests in `docs/algorithms.md` are the ones from before the change.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol

from rel.agents.base import Agent, Transition, rows_of
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule
from rel.spaces import Discrete

Observation = tuple[float, ...]

#: The features that are on for a point, and how strongly each one is on.
#: Two parallel tuples rather than pairs, because the agent walks them together
#: in the inner loop of every step and pairs would allocate a tuple for each.
Encoded = tuple[tuple[int, ...], tuple[float, ...]]


class Coder(Protocol):
    """Everything a linear agent needs from whatever makes its features.

    Written here, next to the only thing that uses it, rather than in a module
    of its own. A coder does not have to know it is one: `TileCoder` and
    `RadialBasis` both satisfy this without importing it, and adding a third
    means writing four methods rather than joining a hierarchy.
    """

    @property
    def features(self) -> int:
        """How many weights an agent needs for each action."""

    def encode(self, observation: Sequence[float]) -> Encoded:
        """Which features are on for this point, and how strongly."""

    def squared_length(self, values: Sequence[float]) -> float:
        """The features of a point dotted with themselves, for the step size."""

    def starting_weight(self, optimism: float) -> float:
        """The weight that makes a state nothing is known about worth this."""


class LinearAgent(Agent[Observation]):
    """The parts that every agent over a coder shares."""

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        coder: Coder,
        *,
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.0,
        optimism: float = 0.0,
    ) -> None:
        super().__init__(rng, actions)

        self.coder = coder
        self.step_size = as_schedule(step_size)
        self.discount = discount
        self.epsilon = as_schedule(epsilon)

        # One weight for each feature, for each action. Started at the share of
        # the optimistic value that one feature carries, so that the value of
        # an unseen state is the optimistic value rather than a multiple of it.
        # How large a share that is depends on the coder, so the coder says.
        share = coder.starting_weight(optimism)
        self.weights: list[list[float]] = [
            [share] * coder.features for _ in range(actions.n)
        ]

    def learned(self) -> Iterator[str]:
        # The weights, one line per action. There is no table of states here:
        # a coder turns a state into features and the weights are per feature,
        # so what this agent knows is the weights and nothing else.
        return rows_of(dict(enumerate(self.weights)))

    def current_epsilon(self) -> float:
        return self.epsilon(self.episodes)

    def current_step_size(self, values: Sequence[float]) -> float:
        return self.step_size(self.steps) / self.coder.squared_length(values)

    def value(self, encoded: Encoded, action: int) -> float:
        row = self.weights[action - self.actions.start]
        indices, values = encoded
        return sum(
            row[index] * value for index, value in zip(indices, values, strict=True)
        )

    def action_values(self, observation: Observation) -> Sequence[float]:
        encoded = self.coder.encode(observation)
        return [self.value(encoded, action) for action in self.actions]

    def act(self, observation: Observation) -> int:
        if self.rng.chance(self.current_epsilon()):
            return self.actions.start + self.rng.below(self.actions.n)
        return self.greedy(observation)

    def greedy(self, observation: Observation) -> int:
        return self._best(self.coder.encode(observation))

    def _best(self, encoded: Encoded) -> int:
        scores = [self.value(encoded, action) for action in self.actions]
        best = max(scores)
        tied = [index for index, score in enumerate(scores) if score == best]
        if len(tied) == 1:
            return self.actions.start + tied[0]
        return self.actions.start + tied[self.rng.below(len(tied))]

    def _nudge(self, encoded: Encoded, action: int, error: float) -> None:
        row = self.weights[action - self.actions.start]
        indices, values = encoded
        change = self.current_step_size(values) * error
        for index, value in zip(indices, values, strict=True):
            row[index] += change * value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.coder!r})"


class SemiGradientSarsa(LinearAgent):
    """SARSA over a coder, which is a tile coder or a radial basis.

    Over a tile coder this is the agent the mountain car chapter uses. Like
    the tabular SARSA it
    waits for the next action to be taken rather than choosing it early, so
    nothing has to be true about the loop for the update to be right.
    """

    def __init__(self, *args: object, **options: object) -> None:
        super().__init__(*args, **options)  # type: ignore[arg-type]
        self._held: Transition[Observation] | None = None

    def observe(self, transition: Transition[Observation]) -> None:
        super().observe(transition)

        held = self._held
        if held is not None:
            self._learn(held, transition.action)

        if transition.terminated:
            self._learn(transition, None)
            self._held = None
        elif transition.truncated:
            self._learn(transition, self.act(transition.next_observation))
            self._held = None
        else:
            self._held = transition

    def end_episode(self) -> None:
        if self._held is not None:
            self._learn(self._held, None)
            self._held = None
        super().end_episode()

    def _learn(
        self, transition: Transition[Observation], next_action: int | None
    ) -> None:
        encoded = self.coder.encode(transition.observation)
        current = self.value(encoded, transition.action)

        if next_action is None:
            target = transition.reward
        else:
            ahead = self.coder.encode(transition.next_observation)
            target = transition.reward + self.discount * self.value(ahead, next_action)

        self._nudge(encoded, transition.action, target - current)


class SemiGradientQ(LinearAgent):
    """Q-learning over a coder, which is a tile coder or a radial basis.

    Off-policy, so it needs no held transition: the target is the best value in
    the next state whatever the policy does there.
    """

    def observe(self, transition: Transition[Observation]) -> None:
        super().observe(transition)

        encoded = self.coder.encode(transition.observation)
        current = self.value(encoded, transition.action)

        if transition.terminated:
            target = transition.reward
        else:
            ahead = self.coder.encode(transition.next_observation)
            best = max(self.value(ahead, action) for action in self.actions)
            target = transition.reward + self.discount * best

        self._nudge(encoded, transition.action, target - current)


__all__ = ["Coder", "Encoded", "LinearAgent", "SemiGradientQ", "SemiGradientSarsa"]
