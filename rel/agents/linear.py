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

A step size for a tile coder is divided by the number of grids.

The reason is arithmetic rather than taste. The value is a sum over one switch
per grid, so adding `a` to each of eight active weights moves the value by
`8a`. A step size of 0.5 with eight grids would move the value four times the
error and the weights would grow without limit. `step_size` here means the
share of the error the value moves by, which is the same thing it means for the
tabular agents, and the division happens inside.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.agents.base import Agent, Transition
from rel.agents.tiles import TileCoder
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule
from rel.spaces import Discrete

Observation = tuple[float, ...]


class LinearAgent(Agent[Observation]):
    """The parts that every agent over a tile coder shares."""

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        coder: TileCoder,
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

        # One weight for each switch, for each action. Started at the share of
        # the optimistic value that one switch carries, so that the value of an
        # unseen state is the optimistic value rather than a multiple of it.
        share = optimism / coder.grids
        self.weights: list[list[float]] = [
            [share] * coder.features for _ in range(actions.n)
        ]

    def current_epsilon(self) -> float:
        return self.epsilon(self.episodes)

    def current_step_size(self) -> float:
        return self.step_size(self.steps) / self.coder.grids

    def value(self, switches: Sequence[int], action: int) -> float:
        row = self.weights[action - self.actions.start]
        return sum(row[switch] for switch in switches)

    def action_values(self, observation: Observation) -> Sequence[float]:
        switches = self.coder.active(observation)
        return [self.value(switches, action) for action in self.actions]

    def act(self, observation: Observation) -> int:
        if self.rng.chance(self.current_epsilon()):
            return self.actions.start + self.rng.below(self.actions.n)
        return self.greedy(observation)

    def greedy(self, observation: Observation) -> int:
        return self._best(self.coder.active(observation))

    def _best(self, switches: Sequence[int]) -> int:
        scores = [self.value(switches, action) for action in self.actions]
        best = max(scores)
        tied = [index for index, score in enumerate(scores) if score == best]
        if len(tied) == 1:
            return self.actions.start + tied[0]
        return self.actions.start + tied[self.rng.below(len(tied))]

    def _nudge(self, switches: Sequence[int], action: int, error: float) -> None:
        row = self.weights[action - self.actions.start]
        change = self.current_step_size() * error
        for switch in switches:
            row[switch] += change

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.coder!r})"


class SemiGradientSarsa(LinearAgent):
    """SARSA over a tile coder.

    This is the agent the mountain car chapter uses. Like the tabular SARSA it
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
        switches = self.coder.active(transition.observation)
        current = self.value(switches, transition.action)

        if next_action is None:
            target = transition.reward
        else:
            ahead = self.coder.active(transition.next_observation)
            target = transition.reward + self.discount * self.value(ahead, next_action)

        self._nudge(switches, transition.action, target - current)


class SemiGradientQ(LinearAgent):
    """Q-learning over a tile coder.

    Off-policy, so it needs no held transition: the target is the best value in
    the next state whatever the policy does there.
    """

    def observe(self, transition: Transition[Observation]) -> None:
        super().observe(transition)

        switches = self.coder.active(transition.observation)
        current = self.value(switches, transition.action)

        if transition.terminated:
            target = transition.reward
        else:
            ahead = self.coder.active(transition.next_observation)
            best = max(self.value(ahead, action) for action in self.actions)
            target = transition.reward + self.discount * best

        self._nudge(switches, transition.action, target - current)


__all__ = ["LinearAgent", "SemiGradientQ", "SemiGradientSarsa"]
