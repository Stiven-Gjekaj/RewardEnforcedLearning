"""Estimating what a fixed policy is worth, without trying to improve it.

Every other agent here controls: it keeps a number for each action, ranks them,
and acts on the ranking. These do not. They follow a policy they were handed
and estimate one number for each state, which is the other half of the subject
and the half the rest of this project takes for granted.

    V(s) <- V(s) + step * (target - V(s))

What changes between them is the target. TD uses the next state's estimate and
Monte Carlo waits for the return, and everything in between is how far to look
before giving up and using an estimate.

## Why this is worth having separately

Control mixes two questions. When an agent's policy improves, its estimates
chase a moving target, and a measurement of how good the estimates are cannot
say whether the answer moved or the estimate did. Holding the policy still
separates them.

It also makes one problem exactly checkable. On `walk` the true values are
arithmetic rather than another computation, so the error of an estimate is a
number and not a comparison. `docs/algorithms.md` measures TD against Monte
Carlo there.

## A constant step size does not converge

Every one of these tracks rather than converges while the step size is a
constant: it settles into a band around the answer whose width is proportional
to the step size. That is the right behaviour for a problem that moves and the
wrong behaviour for one that does not, and the random walk does not. A
measurement of how close a method gets has to decay the step size or it is
measuring the step size.

## What a predictor does about actions

It follows the policy it was given, and `greedy` and `act` are the same thing
because there is no exploration to do: a predictor is not choosing.

The policy is a mapping from state to action, or `None` for one that draws
uniformly. A uniform policy is the classic setting for the random walk and it
cannot be written as a mapping, which is why the argument takes either.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from rel.agents.base import Agent, Transition
from rel.agents.traces import CUTOFF, KINDS, Kind
from rel.core import DIGEST_FIGURES, encoded
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule
from rel.spaces import Discrete


class Predictor(Agent[int]):
    """Follows a fixed policy and keeps one number for each state.

    The table is a dictionary rather than an array, for the same reason the
    control agents' is: a state is anything hashable, and a missing entry is
    the starting value rather than an error.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        policy: Mapping[int, int] | None = None,
        *,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        start_value: float = 0.0,
    ) -> None:
        super().__init__(rng, actions)

        if not 0.0 <= discount <= 1.0:
            raise ValueError("The discount is between 0 and 1.")

        self.policy = dict(policy) if policy is not None else None
        self.step_size = as_schedule(step_size)
        self.discount = discount
        self.start_value = start_value

        self.v: dict[int, float] = {}

    # -- The table ----------------------------------------------------------

    def value(self, observation: int) -> float:
        """What this state is worth, without adding it to the table.

        Reading is free of side effects here for the same reason it is for a
        control agent: a value map has to tell a state worth nothing from a
        state nothing is known about, and `knows` answers that off the table.
        """
        return self.v.get(observation, self.start_value)

    def knows(self, observation: int) -> bool:
        return observation in self.v

    def state_value(self, observation: int) -> float:
        return self.value(observation)

    #: A predictor keeps no number for an action, so there is no policy map to
    #: draw from its numbers and no action to rank.
    def action_values(self, observation: int) -> Sequence[float] | None:
        return None

    def learned(self) -> Iterator[str]:
        for state in sorted(self.v, key=encoded):
            yield f"{encoded(state)}|{self.v[state]:.{DIGEST_FIGURES}g}"

    # -- The policy it was handed -------------------------------------------

    def act(self, observation: int) -> int:
        if self.policy is None:
            return self.actions.start + self.rng.below(self.actions.n)

        action = self.policy.get(observation)
        if action is None:
            raise ValueError(
                f"The policy this predictor was given says nothing about "
                f"state {observation}."
            )
        return action

    def greedy(self, observation: int) -> int:
        # The same thing. A predictor is not choosing, so there is no
        # exploration to leave out.
        return self.act(observation)

    # -- Learning -----------------------------------------------------------

    def current_step_size(self) -> float:
        return self.step_size(self.steps)

    def _move(self, observation: int, target: float) -> None:
        current = self.value(observation)
        self.v[observation] = current + self.current_step_size() * (target - current)

    def target_from(self, transition: Transition[int]) -> float:
        """The one step target: the reward, plus the future unless there is none."""
        if transition.terminated:
            return transition.reward
        return transition.reward + self.discount * self.value(
            transition.next_observation
        )

    def error_against(self, truth: Mapping[int, float]) -> float:
        """The root mean square error of this table against the real values.

        Over the states `truth` names, and the caller chooses those. The states
        an agent can never be in do not belong in it: their entries never move
        off the starting value, so scoring them measures the starting value and
        not the learning. On the random walk that is the two endings, and
        including them buries the answer.

        A state the agent has not seen yet is counted at its starting value
        rather than left out, because leaving it out would report an agent that
        has seen one state as perfect.
        """
        if not truth:
            return 0.0
        squares = sum((self.value(state) - real) ** 2 for state, real in truth.items())
        return float((squares / len(truth)) ** 0.5)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(states={len(self.v)}, "
            f"step_size={self.current_step_size():g})"
        )


class TemporalDifference(Predictor):
    """TD(0). One step, and the rest of the episode replaced by an estimate.

    The oldest idea in the subject and the one the rest of it is built on: do
    not wait to find out what happened, use what you already believe about
    where you ended up.
    """

    def observe(self, transition: Transition[int]) -> None:
        super().observe(transition)
        self._move(transition.observation, self.target_from(transition))


class NStepTD(Predictor):
    """TD that waits n steps before replacing the rest with an estimate.

    At n of one this is TD(0) and at n past the length of the episode it is
    every visit Monte Carlo, and everything between is the same dial the
    control agents have.

    The buffer holds the states and rewards of the window. A state leaves it
    when the window has arrived, which is n steps later or at the end of the
    episode, whichever comes first.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        policy: Mapping[int, int] | None = None,
        *,
        n: int = 2,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        start_value: float = 0.0,
    ) -> None:
        super().__init__(
            rng,
            actions,
            policy,
            step_size=step_size,
            discount=discount,
            start_value=start_value,
        )
        if n < 1:
            raise ValueError("A step count is at least one.")
        self.n = n

        self._states: list[int] = []
        self._rewards: list[float] = []
        self._landed = 0
        self._ended = False
        #: How many of the buffered states have been credited already.
        #:
        #: Counted rather than worked out from the length, because an episode
        #: shorter than `n` never fills the window at all and the arithmetic
        #: that says which states are still owed an update then says none of
        #: them are.
        self._credited = 0

    def start_episode(self) -> None:
        self._states = []
        self._rewards = []
        self._ended = False
        self._credited = 0

    def observe(self, transition: Transition[int]) -> None:
        super(Predictor, self).observe(transition)

        self._states.append(transition.observation)
        self._rewards.append(transition.reward)
        self._landed = transition.next_observation
        self._ended = transition.terminated

        if len(self._states) >= self.n:
            self._update(len(self._states) - self.n)
            self._credited = len(self._states) - self.n + 1

        if transition.done:
            # The episode stopped and the window never filled for what is
            # left. Everything still owed an update gets one from what there
            # was, including an episode too short to have filled the window at
            # all, which is the case that counting rather than calculating is
            # here for.
            for index in range(self._credited, len(self._states)):
                self._update(index)
            self.start_episode()

    def _update(self, first: int) -> None:
        total = 0.0
        weight = 1.0
        for reward in self._rewards[first : first + self.n]:
            total += weight * reward
            weight *= self.discount

        # The tail. A terminated episode has none, and a window that stopped
        # short of the end is worth what the agent believes about where it
        # stopped.
        reaches_end = first + self.n >= len(self._states)
        if not (reaches_end and self._ended):
            total += weight * self.value(
                self._landed if reaches_end else self._states[first + self.n]
            )

        self._move(self._states[first], total)


class MonteCarloPrediction(Predictor):
    """Waits for the episode to end and credits states with the return.

    `first_visit` decides whether a state visited twice in one episode is
    credited once or twice. The two differ on a walk that doubles back, which
    the random walk does constantly, and they converge to the same answer.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        policy: Mapping[int, int] | None = None,
        *,
        first_visit: bool = True,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        start_value: float = 0.0,
    ) -> None:
        super().__init__(
            rng,
            actions,
            policy,
            step_size=step_size,
            discount=discount,
            start_value=start_value,
        )
        self.first_visit = first_visit
        self._episode: list[Transition[int]] = []

    def start_episode(self) -> None:
        self._episode = []

    def observe(self, transition: Transition[int]) -> None:
        super(Predictor, self).observe(transition)
        self._episode.append(transition)
        if transition.done:
            self._credit()
            self._episode = []

    def _credit(self) -> None:
        if not self._episode:
            return

        # The tail a step limit took away is worth what the agent believes
        # about the state it was stopped in. Treating it as zero would teach
        # the agent that a long episode ends somewhere worthless.
        last = self._episode[-1]
        running = 0.0 if last.terminated else self.value(last.next_observation)

        returns: list[float] = []
        for transition in reversed(self._episode):
            running = transition.reward + self.discount * running
            returns.append(running)
        returns.reverse()

        seen: set[int] = set()
        for index, transition in enumerate(self._episode):
            if self.first_visit and transition.observation in seen:
                continue
            seen.add(transition.observation)
            self._move(transition.observation, returns[index])


class TDLambda(Predictor):
    """TD with a trace on every state, which fades rather than stops.

    n-step TD reaches back a whole number of steps. This reaches back over all
    of them at once, weighted by `trace_decay` raised to the distance, which is
    the same dial done smoothly.

    ## Two collapses

    At a decay of zero the only trace left is the state just visited, so this
    is TD(0), cell for cell. At a decay of one with accumulating traces and no
    discount, every state of the episode keeps a full trace and each one ends
    up moved by the whole return, which is every visit Monte Carlo done as the
    episode goes rather than at the end of it.

    The first is exact and tested. The second is exact only when the step size
    is small enough that the updates inside an episode do not change the errors
    that follow, which is the standard caveat on the equivalence and the reason
    the test for it uses a tiny step size.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        policy: Mapping[int, int] | None = None,
        *,
        trace_decay: float = 0.9,
        traces: Kind = "replacing",
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        start_value: float = 0.0,
    ) -> None:
        super().__init__(
            rng,
            actions,
            policy,
            step_size=step_size,
            discount=discount,
            start_value=start_value,
        )
        if not 0.0 <= trace_decay <= 1.0:
            raise ValueError("The trace decay is between 0 and 1.")
        if traces not in KINDS:
            raise ValueError(f"traces is one of {KINDS}. {traces!r} is not.")

        self.trace_decay = trace_decay
        self.traces = traces
        self.e: dict[int, float] = {}

    def start_episode(self) -> None:
        # A trace does not reach across an ending. The state before an episode
        # boundary had nothing to do with what happens after it.
        self.e.clear()

    def bump(self, observation: int) -> None:
        """Raise the trace on the state that was just visited."""
        was = self.e.get(observation, 0.0)
        if self.traces == "accumulating":
            self.e[observation] = was + 1.0
        elif self.traces == "replacing":
            self.e[observation] = 1.0
        else:
            self.e[observation] = (1.0 - self.current_step_size()) * was + 1.0

    def fade(self) -> None:
        """Decay every trace, and drop the ones that no longer say anything."""
        decay = self.discount * self.trace_decay
        if decay <= 0.0:
            self.e.clear()
            return

        faded: dict[int, float] = {}
        for state, value in self.e.items():
            value *= decay
            if value >= CUTOFF:
                faded[state] = value
        self.e = faded

    def observe(self, transition: Transition[int]) -> None:
        super(Predictor, self).observe(transition)

        error = self.target_from(transition) - self.value(transition.observation)
        self.bump(transition.observation)

        step_size = self.current_step_size()
        for state, trace in self.e.items():
            self.v[state] = self.value(state) + step_size * error * trace

        self.fade()
        if transition.done:
            self.start_episode()


__all__ = [
    "MonteCarloPrediction",
    "NStepTD",
    "Predictor",
    "TDLambda",
    "TemporalDifference",
]
