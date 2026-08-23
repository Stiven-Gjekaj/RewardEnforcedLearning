"""Learning from a model of the environment that the agent builds itself.

Q-learning uses each step once and throws it away. Dyna keeps it. After every
real step it remembers what happened, and then it replays a handful of
remembered steps and learns from those as well.

Nothing about the replayed step is imagined. It is a step that really happened,
applied again. The gain is that one real step can move many table entries: on a
maze where only the goal pays, the first episode teaches one cell, and the
first episode with fifty planning steps teaches a path.

## What the model can and cannot hold

The model here maps one state and action to one result. That is right for an
environment where the same action from the same state always does the same
thing, and it is wrong for one where it does not. On a slippery lake the model
remembers the last slip and replays it as if it were certain.

This is the classic version, and the limit is deliberate rather than hidden:
`tags=("tabular",)` in the environment table says which environments have a
model, and this agent is not one of the things that reads it. Dyna on the lake
learns something, and it learns it from a model that is confidently wrong.
"""

from __future__ import annotations

import math

from rel.agents.base import TabularAgent, Transition
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete


class DynaQ(TabularAgent[ObsT]):
    """Q-learning, plus `planning_steps` replays of remembered steps per step."""

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        planning_steps: int = 10,
        step_size: float | Schedule = 0.1,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )
        if planning_steps < 0:
            raise ValueError("The number of planning steps must not be below zero.")
        self.planning_steps = planning_steps

        #: state and action to reward, landing state and whether it ended.
        self.model: dict[tuple[ObsT, int], tuple[float, ObsT, bool]] = {}
        self._seen: list[tuple[ObsT, int]] = []

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)
        self._apply(transition)
        self._remember(transition)
        self._plan()

    def _apply(self, transition: Transition[ObsT]) -> None:
        row = self.values(transition.observation)
        offset = transition.action - self.actions.start
        target = self.bootstrap(
            transition, self.best_value(transition.next_observation)
        )
        row[offset] += self.current_step_size() * (target - row[offset])

    def _remember(self, transition: Transition[ObsT]) -> None:
        key = (transition.observation, transition.action)
        if key not in self.model:
            self._seen.append(key)
        self.model[key] = (
            transition.reward,
            transition.next_observation,
            transition.terminated,
        )

    def _plan(self) -> None:
        if not self._seen:
            return
        for _ in range(self.planning_steps):
            observation, action = self._seen[self.rng.below(len(self._seen))]
            reward, landed, terminated = self.model[(observation, action)]
            self._replay(observation, action, reward, landed, terminated)

    def _replay(
        self,
        observation: ObsT,
        action: int,
        reward: float,
        landed: ObsT,
        terminated: bool,
    ) -> None:
        row = self.values(observation)
        offset = action - self.actions.start
        target = (
            reward if terminated else reward + self.discount * self.best_value(landed)
        )
        row[offset] += self.current_step_size() * (target - row[offset])


class DynaQPlus(DynaQ[ObsT]):
    """Dyna-Q that pays a small bonus for a step it has not tried in a while.

    The bonus is `kappa` times the square root of how many steps have passed
    since that state and action were last taken for real. It is added during
    planning only, so the agent's behaviour is not driven by it directly. What
    it does is keep the value of a long unused corner of the table from
    settling, which sends the agent back to look.

    This is what finds a shortcut that opens after the agent has already
    learned a route. Plain Dyna-Q has no reason to go and look, and it can keep
    the old route for thousands of episodes.

    Every action of a visited state goes into the model, including actions
    never taken there, modelled as doing nothing and paying nothing. Without
    that, the bonus can only be paid for something already tried, and the
    corner that was never entered stays invisible.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        kappa: float = 0.001,
        planning_steps: int = 10,
        step_size: float | Schedule = 0.1,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> None:
        super().__init__(
            rng,
            actions,
            planning_steps=planning_steps,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )
        self.kappa = kappa
        self.last_taken: dict[tuple[ObsT, int], int] = {}

    def _remember(self, transition: Transition[ObsT]) -> None:
        for action in self.actions:
            key = (transition.observation, action)
            if key not in self.model:
                self._seen.append(key)
                # Never tried here. Modelled as doing nothing, which is the
                # entry the bonus needs in order to grow.
                self.model[key] = (0.0, transition.observation, False)

        super()._remember(transition)
        self.last_taken[(transition.observation, transition.action)] = self.steps

    def _replay(
        self,
        observation: ObsT,
        action: int,
        reward: float,
        landed: ObsT,
        terminated: bool,
    ) -> None:
        since = self.steps - self.last_taken.get((observation, action), 0)
        super()._replay(
            observation,
            action,
            reward + self.kappa * math.sqrt(since),
            landed,
            terminated,
        )


__all__ = ["DynaQ", "DynaQPlus"]
