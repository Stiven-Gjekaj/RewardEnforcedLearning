"""Eligibility traces: one dial between one step learning and Monte Carlo.

`n-step-sarsa` in this project chooses a whole number of steps and credits
exactly those. That is the crude version of what is here, and the sweep behind
its default says why: `n` and the step size trade against each other, so the
one dial that reads as "how far back does credit reach" is really two dials
that have to be turned together.

A trace replaces the whole number with a decay. Every state and action the
agent has visited keeps a number that says how much of the current error
belongs to it. The number falls by the discount times `trace_decay` at every
step, so credit reaches back for ever and fades rather than stopping.

At `trace_decay` zero only the cell just visited is updated, and the agent is
the one step agent it was built on. Raising it lengthens the reach smoothly.
The chapters call this dial lambda.

## Three ways to bump a trace

A trace has to be raised when its cell is visited, and there is more than one
defensible way to do it.

`accumulating` adds one. A cell visited twice in quick succession is credited
twice, which is what the forward view of the return says, and which can push a
trace above one and make an update larger than the error it came from.

`replacing` sets it to one. A cell visited twice is credited once, which is
steadier and which the literature found works better on most problems.

`dutch` is between them: it keeps one minus the step size of what was there and
adds one. It is the one that makes the backward view match the forward view
exactly for a linear method.

## Why the traces are pruned

A trace never reaches zero, only smaller. Left alone the table of them grows to
hold every state and action the agent has ever taken, and each step then walks
all of it. `CUTOFF` drops a trace once it is small enough that the update it
would make is below what a float is keeping honestly anyway.
"""

from __future__ import annotations

from typing import Literal

from rel.agents.base import Transition
from rel.agents.td import Sarsa
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete

#: A trace smaller than this is dropped. The update it would make is the error
#: times the step size times the trace, so at a step size of one and an error
#: of ten this is a change of a millionth. Keeping it costs a dictionary entry
#: and a multiply on every step for the rest of the run.
CUTOFF = 1e-6

Kind = Literal["accumulating", "replacing", "dutch"]
KINDS: tuple[Kind, ...] = ("accumulating", "replacing", "dutch")


class SarsaLambda(Sarsa[ObsT]):
    """SARSA with a trace on every cell it has visited.

    The deferral comes from `Sarsa` and is the reason this inherits from it
    rather than from the table directly: the update needs the action that was
    really taken next, and promising one before the loop takes it is the fault
    that class was written to avoid.

    What is added is the trace. `Sarsa` writes the error into one cell. This
    writes it into every cell that has a trace, in proportion to it.
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
        trace_decay: float = 0.9,
        traces: Kind = "replacing",
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )

        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("trace_decay is between 0 and 1.")
        if traces not in KINDS:
            raise ValueError(f"traces is one of {KINDS}. {traces!r} is not.")

        self.trace_decay = trace_decay
        self.traces = traces
        self.e: dict[tuple[ObsT, int], float] = {}

    # -- The traces ---------------------------------------------------------

    def bump(self, observation: ObsT, action: int) -> None:
        """Raise the trace on the cell that was just visited."""
        key = (observation, action)
        was = self.e.get(key, 0.0)

        if self.traces == "accumulating":
            self.e[key] = was + 1.0
        elif self.traces == "replacing":
            self.e[key] = 1.0
        else:
            self.e[key] = (1.0 - self.current_step_size()) * was + 1.0

    def fade(self) -> None:
        """Decay every trace, and drop the ones that no longer say anything."""
        decay = self.discount * self.trace_decay
        if decay <= 0.0:
            self.e.clear()
            return

        faded: dict[tuple[ObsT, int], float] = {}
        for key, value in self.e.items():
            value *= decay
            if value >= CUTOFF:
                faded[key] = value
        self.e = faded

    def start_episode(self) -> None:
        # Credit does not reach across an ending. An episode that begins with
        # the last one's traces still on the table would write this episode's
        # error into cells that belong to a run that is over.
        self.e.clear()
        super().start_episode()

    # -- The update ---------------------------------------------------------

    def error(self, transition: Transition[ObsT], next_action: int | None) -> float:
        """How wrong the value of the cell just left turned out to be."""
        row = self.values(transition.observation)
        index = transition.action - self.actions.start

        if next_action is None:
            target = transition.reward
        else:
            offset = next_action - self.actions.start
            target = self.bootstrap(
                transition, self.peek(transition.next_observation)[offset]
            )

        return target - row[index]

    def _learn(self, transition: Transition[ObsT], next_action: int | None) -> None:
        delta = self.error(transition, next_action)
        self.bump(transition.observation, transition.action)

        step = self.current_step_size()
        for (observation, action), trace in self.e.items():
            self.values(observation)[action - self.actions.start] += (
                step * delta * trace
            )

        self.fade()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(trace_decay={self.trace_decay:g}, "
            f"traces={self.traces!r}, step_size={self.current_step_size():g})"
        )


class WatkinsQLambda(SarsaLambda[ObsT]):
    """Q-learning with traces, cut whenever the agent explores.

    A trace says that a cell is partly responsible for what is happening now.
    Q-learning learns about the greedy policy, so as soon as the agent takes an
    action the greedy policy would not have taken, what happens next is no
    longer evidence about anything that came before it. Watkins' answer is to
    cut every trace at that moment and start again.

    The cost is that on a policy that explores often, the traces rarely get
    long, and the dial does less than it does on SARSA. That is a real cost and
    it is the reason the tree backup method exists.
    """

    def error(self, transition: Transition[ObsT], next_action: int | None) -> float:
        row = self.values(transition.observation)
        index = transition.action - self.actions.start

        if next_action is None:
            target = transition.reward
        else:
            target = self.bootstrap(
                transition, self.best_value(transition.next_observation)
            )

        return target - row[index]

    def _learn(self, transition: Transition[ObsT], next_action: int | None) -> None:
        # Whether the action taken next was the greedy one has to be read
        # before the update, because the update changes what greedy means.
        explored = next_action is not None and next_action != self.greedy(
            transition.next_observation
        )

        super()._learn(transition, next_action)

        if explored:
            self.e.clear()


__all__ = ["CUTOFF", "KINDS", "Kind", "SarsaLambda", "WatkinsQLambda"]
