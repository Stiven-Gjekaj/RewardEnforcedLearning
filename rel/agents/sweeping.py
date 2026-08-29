"""Dyna that replays the step that matters rather than one drawn at random.

`dyna-q` replays remembered steps uniformly. On a maze where only the goal
pays, almost every one of those replays is a step in a corner nothing has
changed about, and it teaches nothing. The one replay that would teach
something is the step leading into the cell whose value has just moved.

Prioritised sweeping keeps a queue of steps ordered by how much they would
change, replays the largest, and then asks which steps lead into the cell it
just changed and puts those on the queue too. Work follows the change backwards
through the model rather than being scattered over it.

## The predecessor table

Working backwards needs to know which state and action pairs land in a given
state. That is not in the model, which points the other way, so it is kept
alongside it and filled in as steps arrive.

## Every update comes off the queue

The real step is pushed rather than applied. So is everything else, and the
count of replays is the count of updates the agent has made in total, which is
what the comparison against `dyna-q` in `docs/algorithms.md` is about.

The consequence is that this agent with no planning steps learns nothing at
all, where `dyna-q` with no planning steps is plain Q-learning. That is a trap
rather than a feature, so the setting is refused.

## Why the queue is capped

The queue holds one entry per state and action, so it cannot be larger than the
model. It can still be large, and both the push and the pop walk it, so a run
on a big model spends its time in the queue rather than in the environment.
`CAP` bounds it, and the entry dropped is the smallest, which is the one whose
replay would have changed the least.

No environment in this repository comes near that bound: the largest grid here
has 54 cells and four actions. The bound is there because `TabularAgent` takes
an observation of any hashable shape, including a pair of real numbers that has
been rounded, and the number of those is not small.

## Why a threshold

A step whose replay would change a value by almost nothing is not worth a slot.
`threshold` is what "almost nothing" means, and it is an argument rather than a
constant because it has to be small against the rewards the environment really
pays, and nothing about the agent knows what those are. That is the same
argument the exploration bonus of Dyna-Q+ has, and it is made in the same place
for the same reason.

The threshold is also why the agent stops. Once every step in the model would
change by less than it, the queue empties and stays empty, and the agent does
no planning at all until a real step surprises it again. `dyna-q` in the same
position keeps making its full quota of replays forever.
"""

from __future__ import annotations

from rel.agents.base import Transition
from rel.agents.dyna import DynaQ
from rel.agents.explore import Rule
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete

#: How many entries the queue holds. Beyond this the smallest is dropped, and
#: the smallest is the replay that would have changed the least.
CAP = 2000


class PrioritisedSweeping(DynaQ[ObsT]):
    """Dyna-Q whose planning follows the change backwards through the model.

    The model and its bookkeeping are inherited. What is replaced is which
    steps get replayed: `DynaQ` draws them uniformly and this takes the largest
    change first, then queues whatever leads into what it changed.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        planning_steps: int = 10,
        threshold: float = 1e-4,
        step_size: float | Schedule = 0.1,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            planning_steps=planning_steps,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        if planning_steps < 1:
            raise ValueError(
                "Prioritised sweeping needs at least one planning step. "
                "Every update it makes comes off the queue, so with none it "
                "would learn nothing."
            )
        if threshold < 0.0:
            raise ValueError("The threshold must not be below zero.")
        self.threshold = threshold

        #: A state, to every state and action that has ever landed in it.
        self.leading_to: dict[ObsT, set[tuple[ObsT, int]]] = {}
        #: A state and action, to how much replaying it would change its value.
        self.queue: dict[tuple[ObsT, int], float] = {}
        #: How many replays this agent has made, which is how many updates it
        #: has made in total. The measurement in `docs/algorithms.md` is about
        #: this rather than about episodes, because the whole claim is that
        #: fewer of them are needed.
        self.replays = 0

    # -- The queue ----------------------------------------------------------

    def change_from(self, observation: ObsT, action: int) -> float:
        """How far a replay of this step would move its own value."""
        remembered = self.model.get((observation, action))
        if remembered is None:
            return 0.0

        reward, landed, terminated = remembered
        target = (
            reward if terminated else reward + self.discount * self.best_value(landed)
        )
        offset = action - self.actions.start
        return abs(target - self.peek(observation)[offset])

    def push(self, observation: ObsT, action: int, change: float) -> None:
        """Put a step on the queue, if it is worth a slot."""
        if change <= self.threshold:
            return

        key = (observation, action)
        if change <= self.queue.get(key, 0.0):
            return
        self.queue[key] = change

        if len(self.queue) > CAP:
            smallest = min(self.queue, key=lambda entry: self.queue[entry])
            del self.queue[smallest]

    def pop(self) -> tuple[ObsT, int] | None:
        """The step whose replay would change the most, and take it off."""
        if not self.queue:
            return None
        key = max(self.queue, key=lambda entry: self.queue[entry])
        del self.queue[key]
        return key

    # -- The loop -----------------------------------------------------------

    def _remember(self, transition: Transition[ObsT]) -> None:
        super()._remember(transition)
        if not transition.terminated:
            self.leading_to.setdefault(transition.next_observation, set()).add(
                (transition.observation, transition.action)
            )

    def observe(self, transition: Transition[ObsT]) -> None:
        # The real step goes on the queue rather than being applied directly,
        # so that everything this agent does to the table happens in `_plan`
        # and the count of replays is the count of updates.
        super(DynaQ, self).observe(transition)
        self._remember(transition)
        self.push(
            transition.observation,
            transition.action,
            self.change_from(transition.observation, transition.action),
        )
        self._plan()

    def _plan(self) -> None:
        for _ in range(self.planning_steps):
            key = self.pop()
            if key is None:
                return

            observation, action = key
            reward, landed, terminated = self.model[key]
            self._replay(observation, action, reward, landed, terminated)
            self.replays += 1

            # Whatever leads into the cell just changed is now worth a look.
            for before, taken in self.leading_to.get(observation, set()):
                self.push(before, taken, self.change_from(before, taken))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(planning_steps={self.planning_steps}, "
            f"threshold={self.threshold:g}, queued={len(self.queue)})"
        )


__all__ = ["CAP", "PrioritisedSweeping"]
