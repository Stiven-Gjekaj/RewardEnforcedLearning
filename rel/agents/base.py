"""What every agent has to do, and the table that most of them keep.

An agent chooses an action from an observation, and it is told what happened.
That is the whole contract. It never sees the environment, never reads the
audit, and never learns which environment it is in.

That last point is a rule and not a habit. An agent that could tell a cliff
from a lake would be tuned for each, and the numbers in the documentation would
stop meaning what they say. `tests/test_layering.py` reads the imports of every
module in `rel/agents/` and fails if one of them names an environment.

## Terminated and truncated, again

Every learning rule here bootstraps: it replaces the rest of the episode with
what it currently believes that rest is worth. The one case where it must not
is a state the environment says has no future.

    target = reward                                   if terminated
    target = reward + discount * value_of(next_state) otherwise

A step limit is not termination. The state at the limit has a future, and an
agent that treats the limit as an ending teaches itself that a long episode
ends in a worthless place. On the cart pole, where the limit is the best
outcome the environment has, that turns the correct answer into the one the
agent avoids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule
from rel.spaces import Discrete


@dataclass(frozen=True, slots=True)
class Transition(Generic[ObsT]):
    """One step, as the agent sees it."""

    observation: ObsT
    action: int
    reward: float
    next_observation: ObsT
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class Agent(ABC, Generic[ObsT]):
    """Something that chooses actions and learns from what follows."""

    def __init__(self, rng: Rng, actions: Discrete) -> None:
        self.rng = rng
        self.actions = actions
        self.episodes = 0
        self.steps = 0

    @abstractmethod
    def act(self, observation: ObsT) -> int:
        """The action to take now, exploration included."""

    @abstractmethod
    def greedy(self, observation: ObsT) -> int:
        """The action this agent believes is best, with no exploration.

        This is what an evaluation run uses. Reporting the learning policy
        instead mixes the quality of what was learned with the cost of still
        exploring, and the two move in opposite directions near the end of a
        run.
        """

    def start_episode(self) -> None:
        """Called after the environment resets and before the first action.

        Almost nothing needs this. A bandit agent does: every episode of that
        environment is a new set of levers, so an agent that carried its
        estimates across would be answering a question about the last problem.
        """

    def observe(self, transition: Transition[ObsT]) -> None:
        """Take in one step. The default learns nothing."""
        self.steps += 1

    def end_episode(self) -> None:
        """Called after the last step of an episode."""
        self.episodes += 1

    def action_values(self, observation: ObsT) -> Sequence[float] | None:
        """What this agent thinks each action is worth here, if it can say.

        A renderer draws a policy and a value map from this. An agent that
        keeps no such number returns `None` and is drawn as a policy only.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class RandomAgent(Agent[ObsT]):
    """Chooses with equal probability and learns nothing.

    This is the baseline every result is measured against. An agent that has
    learned nothing still returns a number, and a curve that rises can rise
    because the environment got easier. The only way to know is to run this one
    on the same environment and the same seed.
    """

    def act(self, observation: ObsT) -> int:
        return self.actions.start + self.rng.below(self.actions.n)

    def greedy(self, observation: ObsT) -> int:
        return self.act(observation)


class TabularAgent(Agent[ObsT]):
    """An agent that keeps one number for each observation and action.

    The table is a dictionary rather than an array. The cost is a little speed
    and the gain is that an observation of any hashable shape works, so the
    same class covers a grid cell, a lever and a pair of numbers that has been
    rounded.

    ## Ties are broken at random

    An untouched table holds the same number everywhere, so every action ties.
    Taking the first one turns the starting policy into "always go up", and on
    a grid that is not a neutral start: it explores one direction far more than
    the others for as long as the tie lasts. Breaking ties with the agent's own
    source of chance costs one draw and removes the bias.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> None:
        super().__init__(rng, actions)

        if not 0.0 <= discount <= 1.0:
            raise ValueError("The discount is between 0 and 1.")

        self.step_size = as_schedule(step_size)
        self.discount = discount
        self.epsilon = as_schedule(epsilon)
        self.optimism = optimism

        self.q: dict[ObsT, list[float]] = {}

    # -- The table ----------------------------------------------------------

    def values(self, observation: ObsT) -> list[float]:
        """The row for this observation, made on first sight."""
        row = self.q.get(observation)
        if row is None:
            row = [self.optimism] * self.actions.n
            self.q[observation] = row
        return row

    def action_scores(self, observation: ObsT) -> Sequence[float]:
        """The numbers the policy ranks actions by.

        For almost every agent this is the table. Double Q-learning keeps two
        tables and acts on their sum, so it overrides this one method and gets
        the greedy rule, the exploration probabilities and the renderer to
        agree with it for free.
        """
        return self.values(observation)

    def action_values(self, observation: ObsT) -> Sequence[float] | None:
        return self.action_scores(observation)

    def best_value(self, observation: ObsT) -> float:
        return max(self.values(observation))

    # -- The policy ---------------------------------------------------------

    def current_epsilon(self) -> float:
        return self.epsilon(self.episodes)

    def current_step_size(self) -> float:
        return self.step_size(self.steps)

    def act(self, observation: ObsT) -> int:
        if self.rng.chance(self.current_epsilon()):
            return self.actions.start + self.rng.below(self.actions.n)
        return self.greedy(observation)

    def greedy(self, observation: ObsT) -> int:
        return self.actions.start + self._argmax(self.action_scores(observation))

    def _argmax(self, row: Sequence[float]) -> int:
        best = max(row)
        tied = [index for index, value in enumerate(row) if value == best]
        if len(tied) == 1:
            return tied[0]
        return tied[self.rng.below(len(tied))]

    def policy_probabilities(self, observation: ObsT) -> list[float]:
        """How likely this agent is to take each action here.

        Expected SARSA needs this, and so does an off-policy correction. It is
        written once here so that a change to the exploration rule reaches
        every agent that reasons about it.
        """
        epsilon = self.current_epsilon()
        count = self.actions.n
        row = self.action_scores(observation)

        best = max(row)
        tied = [index for index, value in enumerate(row) if value == best]
        share = epsilon / count
        probabilities = [share] * count
        for index in tied:
            probabilities[index] += (1.0 - epsilon) / len(tied)
        return probabilities

    # -- The target ---------------------------------------------------------

    def bootstrap(self, transition: Transition[ObsT], value: float) -> float:
        """The learning target: the reward, plus the future unless there is none.

        A terminated episode has no future and the value is dropped. A
        truncated one does have a future, and dropping it there is the fault
        this method exists to make hard to write by accident.
        """
        if transition.terminated:
            return transition.reward
        return transition.reward + self.discount * value

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(step_size={self.current_step_size():g}, "
            f"discount={self.discount:g}, epsilon={self.current_epsilon():g})"
        )


__all__ = ["Agent", "RandomAgent", "TabularAgent", "Transition"]
