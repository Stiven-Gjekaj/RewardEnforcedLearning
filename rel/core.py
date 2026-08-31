"""The contract between an environment and an agent.

An environment gives an observation, takes an action, and pays a reward. An
agent reads the observation and chooses the action. Neither one imports the
other, and a test in `tests/test_layering.py` holds that.

## Terminated is not truncated

A step reports two different endings, and an agent has to treat them
differently.

`terminated` means the episode ended inside the rules of the environment. The
pole fell over. The agent reached the goal. There is no future from here, so
the value of what comes next is zero.

`truncated` means the episode was stopped from outside. A step limit ran out.
The state that the agent is in still has a future, and its value is not zero.
An agent that bootstraps through a truncation learns the correct value. An
agent that treats truncation as termination teaches itself that the state at
the limit is worthless, and then avoids the long path that would have won.

This is worth the two fields. A cart pole run that is cut off at 500 steps is
the best outcome the environment has, and one flag would record it as the same
outcome as dropping the pole.

## Why the base class does the stepping

`Env.step` is not overridden. It validates the action, calls `_step`, counts
the step and decides truncation. The subclass writes `_step` and never sees the
counter.

Two faults go away with this. An environment cannot forget to truncate, and an
environment cannot accept an action outside its space and do something quiet
with it. The second one has a habit of turning into an agent that looks broken.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from rel.rng import Rng
from rel.spaces import Discrete, Space

ObsT = TypeVar("ObsT")

#: What an environment accepts as an action.
#:
#: A whole number for every environment here but the ones whose action space is
#: a box, where it is a tuple of floats. It is a parameter rather than a union
#: so that an agent over whole numbers can subtract the start of its action
#: space without narrowing anything first, which is what it did before there
#: was more than one kind of action.
ActT = TypeVar("ActT")

# An empty info map, shared. A step that reports nothing extra hands this back
# rather than building a dictionary that nothing reads.
NO_INFO: Mapping[str, float] = {}


@dataclass(frozen=True, slots=True)
class Step(Generic[ObsT]):
    """What one action gave back.

    `info` holds numbers only. That is a deliberate limit: everything on this
    channel gets averaged over episodes by the recorder, and a string cannot be
    averaged. A value that is not a number belongs on the environment itself,
    where a caller can ask for it by name.
    """

    observation: ObsT
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, float] = field(default_factory=lambda: NO_INFO)

    @property
    def done(self) -> bool:
        """True if the episode is over, whichever way it ended."""
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class EnvSpec:
    """What an environment says about itself before it is run.

    `solved_return` is the average return that counts as solved. It is a
    reference point for a report, not a rule that anything enforces, and it is
    `None` where the project has no defensible number to put there.
    """

    name: str
    summary: str
    max_episode_steps: int | None = None
    solved_return: float | None = None

    #: Whether an episode of this can end inside the rules at all.
    #:
    #: A bandit and a boat race have no ending. Every episode of one runs to
    #: its step limit, and a report that said "twenty of twenty ran out of
    #: steps" about them would be describing the environment rather than the
    #: policy. On an environment that can end, the same line means the policy
    #: stopped moving, which is worth saying loudly.
    ends: bool = True

    #: The discount this environment is meant to be run at.
    #:
    #: An environment that never ends has to be discounted. Without it the
    #: return of a run that goes on for ever is unbounded, dynamic programming
    #: does not converge, and a learned value grows until it stops meaning
    #: anything. Saying so here means a caller who gives no discount gets one
    #: that works rather than a failure to explain.
    suggested_discount: float = 1.0


#: How many significant figures of a float are written down for a digest.
#: Twelve is far more than enough to catch a change in behaviour and short
#: enough that the text stays small.
DIGEST_FIGURES = 12


def encoded(value: Any) -> str:
    """One value, as text that does not depend on how Python prints it.

    This is here rather than beside the digest that uses it because both sides
    write values down: a run hashes the observations an environment produced,
    and an agent hashes the table it learned. One spelling means the two agree
    and neither has to know about the other.
    """
    if isinstance(value, tuple):
        return ",".join(encoded(item) for item in value)
    if isinstance(value, float):
        return f"{value:.{DIGEST_FIGURES}g}"
    return str(value)


@dataclass(frozen=True, slots=True)
class Outcome:
    """One branch of a transition, for an environment that knows its own model.

    Dynamic programming needs this. Learning does not, and an environment is
    free to give no model at all.
    """

    probability: float
    observation: int
    reward: float
    terminated: bool


class Env(ABC, Generic[ObsT, ActT]):
    """An environment that an agent can act in.

    Two parameters: what it hands back and what it accepts. Almost every
    environment here accepts a whole number and says so by subclassing
    `DiscreteEnv` below, which is the same class with the second parameter
    already filled in.
    """

    #: The values `reset` and `step` return.
    observation_space: Space[ObsT]

    #: The values `step` accepts.
    action_space: Space[ActT]

    spec: EnvSpec

    def __init__(self, rng: Rng) -> None:
        self.rng = rng
        self._elapsed = 0
        self._started = False

    # -- The contract -------------------------------------------------------

    def reset(self) -> ObsT:
        """Begin a new episode and give the first observation."""
        self._elapsed = 0
        self._started = True
        return self._reset()

    def step(self, action: ActT) -> Step[ObsT]:
        """Take one action and report what happened."""
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"{action!r} is not an action of {self.action_space!r}.")

        outcome = self._step(action)
        self._elapsed += 1

        limit = self.spec.max_episode_steps
        if not outcome.terminated and limit is not None and self._elapsed >= limit:
            outcome = Step(
                observation=outcome.observation,
                reward=outcome.reward,
                terminated=False,
                truncated=True,
                info=outcome.info,
            )

        if outcome.done:
            self._started = False

        return outcome

    @abstractmethod
    def _reset(self) -> ObsT:
        """Put the environment into a start state and give the observation."""

    @abstractmethod
    def _step(self, action: ActT) -> Step[ObsT]:
        """Apply one action. Never set `truncated`; the base class does that."""

    # -- What the reward does not say ---------------------------------------

    def audit(self) -> Mapping[str, float]:
        """What the episode really did, whether or not the reward paid for it.

        A reward is a number that an environment chose to pay. It is not a
        score. Where the two come apart, this is where the difference is
        reported: laps actually completed, vases actually broken, minutes
        actually spent inside the comfortable range.

        An agent never sees this. It is measured for the report only, which is
        the point. An agent that could read it would optimise it, and then the
        gap this project is about would close for the wrong reason.
        """
        return NO_INFO

    # -- Optional extras ----------------------------------------------------

    def render(self) -> str:
        """The environment as lines of text.

        This returns a string. It never writes to a terminal and never chooses
        a colour, so an environment stays runnable with no display attached.
        `rel.ui` takes this and decorates it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not render.")

    @property
    def elapsed(self) -> int:
        """Steps taken in the episode running now."""
        return self._elapsed

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.rng.seed})"


class DiscreteEnv(Env[ObsT, int], ABC):
    """An environment whose actions are whole numbers, which is all but one.

    Everything above this line is written without knowing what an action is.
    Everything below it counts the actions or walks them, which only a discrete
    space can do, so the narrowing is written down once here.
    """

    action_space: Discrete


class TabularEnv(DiscreteEnv[int], ABC):
    """An environment with a finite state set that can describe its own model.

    This is what dynamic programming needs, and it is what makes a claim like
    "the agent reached 94 percent of the best possible return" checkable. The
    best possible return comes from value iteration over these transitions
    rather than from a number somebody remembered.

    Writing the model out is extra work for the environment, and it earns its
    keep twice: `rel.agents.dp` solves the environment exactly, and
    `tests/test_model_agrees_with_stepping.py` drives the environment for
    thousands of steps and checks that what it does matches what it says it
    does. A model that drifts from the code is worse than no model.
    """

    observation_space: Discrete

    @abstractmethod
    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        """Every branch that this action can take from this state.

        The probabilities must add up to one.
        """

    @abstractmethod
    def start_states(self) -> Sequence[tuple[float, int]]:
        """The start distribution, as probability and state pairs."""

    def terminal_states(self) -> frozenset[int]:
        """States that no action leaves.

        The default reads the model: a state is terminal when every branch out
        of it says so. An environment with an unusual shape can say otherwise.
        """
        terminal: set[int] = set()
        for state in range(self.observation_space.n):
            branches = [
                outcome
                for action in self.action_space
                for outcome in self.transitions(state, action)
            ]
            if branches and all(outcome.terminated for outcome in branches):
                terminal.add(state)
        return frozenset(terminal)


__all__ = [
    "DIGEST_FIGURES",
    "NO_INFO",
    "DiscreteEnv",
    "Env",
    "EnvSpec",
    "ObsT",
    "Outcome",
    "Step",
    "TabularEnv",
    "encoded",
]
