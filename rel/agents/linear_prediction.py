"""Prediction over features rather than a table, and what it costs.

Every prediction agent in `rel.agents.prediction` keeps one number for each
state. These keep one weight for each feature, which is the same move the
control agents in `rel.agents.linear` make, and it is the move that lets a
method work on a problem with more states than a table can hold.

It also removes the guarantee. A tabular TD agent converges. A linear one
converges on-policy, and off-policy it can run away, and the three ingredients
it takes are known:

    function approximation   more states than weights, so states share
    bootstrapping            a target built from the estimate, not the return
    off-policy               data from one policy, a question about another

Any two of the three are safe. All three together are the **deadly triad**, and
`rel.envs.baird` is seven states that pay nothing on which `SemiGradientTD`
below diverges. `GradientTD` is the same problem with one term added, and it
does not.

## The step size means the same thing here as everywhere else

Divided by the features of the state dotted with themselves, as in
`rel.agents.linear`, so `step_size` is the share of the error the value moves
by rather than a number whose right size depends on the coder.

On Baird's counterexample every row has a squared length of five, so a step
size here is exactly five times the one in the literature. A run at 0.05 here
is a run at 0.01 there, and nothing else about the comparison changes.

## Why the two policies are handed in

A predictor does not choose. It is told what policy collected the data and what
policy the question is about, and when those are the same object this is
ordinary on-policy prediction with every ratio equal to one.

Handing in both, rather than deriving the behaviour one from an exploration
setting, is what makes the off-policy case something a caller states rather
than something that happens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

from rel.agents.base import Agent, Transition, rows_of
from rel.agents.linear import Coder, Encoded
from rel.agents.off_policy import ratio
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule
from rel.spaces import Discrete

#: How likely a policy is to take each action here. A function of the state,
#: because a policy that is not is the special case rather than the rule.
Policy = Callable[[ObsT], Sequence[float]]


def fixed(shares: Sequence[float]) -> Policy[ObsT]:
    """A policy that takes each action as often wherever it stands.

    Both of Baird's policies are this, and so is the uniform policy the random
    walk is usually predicted under. A policy that reads the state is still a
    `Policy`, and nothing here knows the difference.
    """
    if not shares:
        raise ValueError("A policy needs a share for each action.")
    if any(share < 0.0 for share in shares):
        raise ValueError("A share of an action is not negative.")
    if abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError(f"The shares of a policy add up to one, not {sum(shares)}.")

    held = tuple(shares)

    def shares_at(observation: ObsT) -> Sequence[float]:
        return held

    return shares_at


class LinearPredictor(Agent[ObsT]):
    """What a fixed policy is worth, as a weight for each feature.

    The weights are one vector rather than one for each action, because a
    predictor answers about a state and not about a choice made in it.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        coder: Coder[ObsT],
        behaviour: Policy[ObsT],
        target: Policy[ObsT] | None = None,
        *,
        step_size: float | Schedule = 0.05,
        discount: float = 1.0,
        start_value: float = 0.0,
    ) -> None:
        super().__init__(rng, actions)

        if not 0.0 <= discount <= 1.0:
            raise ValueError("The discount is between 0 and 1.")

        self.coder = coder
        self.behaviour = behaviour
        #: The same object when the caller gave none, so `is` tells the
        #: on-policy case from an off-policy one that happens to agree.
        self.target = behaviour if target is None else target
        self.step_size = as_schedule(step_size)
        self.discount = discount

        self.weights: list[float] = [
            coder.starting_weight(start_value)
        ] * coder.features

    # -- What it believes ---------------------------------------------------

    def worth(self, encoded: Encoded) -> float:
        """The value of a state whose features are already worked out."""
        indices, values = encoded
        return sum(
            self.weights[index] * value
            for index, value in zip(indices, values, strict=True)
        )

    def value(self, observation: ObsT) -> float:
        return self.worth(self.coder.encode(observation))

    def state_value(self, observation: ObsT) -> float:
        return self.value(observation)

    #: A predictor keeps no number for an action, so there is nothing to rank.
    def action_values(self, observation: ObsT) -> Sequence[float] | None:
        return None

    def largest_weight(self) -> float:
        """The largest weight there is, ignoring its sign.

        This is the number that runs away. A norm would do as well and would
        reach infinity sooner, and a maximum stays readable for longer.
        """
        return max(abs(weight) for weight in self.weights)

    def error_against(self, truth: Mapping[ObsT, float]) -> float:
        """The root mean square error of these estimates against the answer.

        Over the states `truth` names. On Baird's counterexample the answer is
        zero at every state, so this is the size of the estimates themselves,
        and a method that works drives it to nothing.

        The squaring is a multiplication rather than a power, which is not a
        style choice. A diverging run reaches weights near the largest float
        there is, `x ** 2` raises on those and `x * x` gives infinity, and a
        measurement of divergence that raises rather than reporting a number is
        the one case this method exists for.
        """
        if not truth:
            return 0.0
        squares = 0.0
        for state, real in truth.items():
            gap = self.value(state) - real
            squares += gap * gap
        return float((squares / len(truth)) ** 0.5)

    def learned(self) -> Iterator[str]:
        yield from rows_of({"weights": self.weights})

    # -- The policy it was handed -------------------------------------------

    def act(self, observation: ObsT) -> int:
        return self.actions.start + self.rng.weighted_index(self.behaviour(observation))

    def greedy(self, observation: ObsT) -> int:
        # The target policy, which is the one the estimates are about. An
        # evaluation run of a predictor is a run of the policy in question,
        # and on-policy the two are the same draw.
        return self.actions.start + self.rng.weighted_index(self.target(observation))

    # -- Learning -----------------------------------------------------------

    def shared_out(self, size: float, values: Sequence[float]) -> float:
        """A step size divided by the features it is about to be spread over.

        The value of a state is a sum over its active features, so a step size
        that is not divided moves the value by more than the step size says.
        `rel.agents.linear` explains it at length and does the same thing.
        """
        return size / self.coder.squared_length(values)

    def current_step_size(self, values: Sequence[float]) -> float:
        return self.shared_out(self.step_size(self.steps), values)

    def correction(self, transition: Transition[ObsT]) -> float:
        """How much more likely the target policy was to take this action."""
        return ratio(
            self.target(transition.observation),
            self.behaviour(transition.observation),
            transition.action - self.actions.start,
        )

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)

        rho = self.correction(transition)
        if rho == 0.0:
            # The target policy would never have taken this action, so the
            # step is no evidence about it. Nothing here is an optimisation:
            # every term of the update carries the ratio.
            return

        here = self.coder.encode(transition.observation)
        if not here[0]:
            # No feature is on here, so this state's value is pinned at zero
            # whatever the weights are and the gradient of it is zero as well.
            # There is nothing to move, and the step size would be divided by
            # a squared length of zero to find out.
            return

        current = self.worth(here)

        if transition.terminated:
            ahead: Encoded = ((), ())
            future = 0.0
        else:
            ahead = self.coder.encode(transition.next_observation)
            future = self.worth(ahead)

        error = transition.reward + self.discount * future - current
        self._learn(here, ahead, rho, error)

    def _learn(self, here: Encoded, ahead: Encoded, rho: float, error: float) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.coder!r})"


class SemiGradientTD(LinearPredictor[ObsT]):
    """TD(0) over a coder. The third leg of the triad, and it diverges.

    The update is the tabular one with the table taken out: each active weight
    moves by its share of the error, times the importance ratio.

        w <- w + step * rho * error * x

    It is called semi-gradient because the error contains the value of the next
    state, which also depends on the weights, and that dependence is ignored.
    Following it as well is a different method and this is not it.

    On-policy this converges to within a bounded distance of the best
    approximation there is. Off-policy it has no such guarantee, and Baird's
    counterexample is where the guarantee is missing rather than merely
    unproven: the weights grow without limit on a problem whose answer is zero.

    Whether they do depends on the discount, and sharply.
    `scripts/measure_triad.py` finds the crossing.
    """

    def _learn(self, here: Encoded, ahead: Encoded, rho: float, error: float) -> None:
        indices, values = here
        change = self.current_step_size(values) * rho * error
        for index, value in zip(indices, values, strict=True):
            self.weights[index] += change * value


class GradientTD(LinearPredictor[ObsT]):
    """The same update with the missing term put back. Called TDC.

    Semi-gradient TD follows the gradient of half the squared error and drops
    the part of it that runs through the next state's value. Off-policy that
    dropped part is what stops the correction coming back, and the fix is to
    estimate it and subtract it.

        w <- w + step * rho * (error * x - discount * x' * (x . helper))
        helper <- helper + helper_step * rho * (error - x . helper) * x

    `helper` is a second weight vector the same length as the first. It is a
    running estimate of the error as a linear function of the features, and it
    is the only new state this method carries.

    The cost is one more vector and one more step size. What it buys is
    convergence off-policy, on the same problem, from the same start, and
    `docs/algorithms.md` measures both.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        coder: Coder[ObsT],
        behaviour: Policy[ObsT],
        target: Policy[ObsT] | None = None,
        *,
        helper_step: float = 0.25,
        step_size: float | Schedule = 0.05,
        discount: float = 1.0,
        start_value: float = 0.0,
    ) -> None:
        super().__init__(
            rng,
            actions,
            coder,
            behaviour,
            target,
            step_size=step_size,
            discount=discount,
            start_value=start_value,
        )

        if helper_step < 0.0:
            raise ValueError("The helper step size is not negative.")

        self.helper_step = helper_step
        self.helper: list[float] = [0.0] * self.coder.features

    def learned(self) -> Iterator[str]:
        # Both vectors, and each says which it is. Run together, an agent that
        # had learned a thing in one would match one that learned it in the
        # other.
        yield from rows_of({"weights": self.weights})
        yield from rows_of({"helper": self.helper})

    def _learn(self, here: Encoded, ahead: Encoded, rho: float, error: float) -> None:
        indices, values = here
        step = self.current_step_size(values)

        # How much of the error the helper already explains at this state. It
        # is read before either vector moves, because both updates use it.
        explained = sum(
            self.helper[index] * value
            for index, value in zip(indices, values, strict=True)
        )

        for index, value in zip(indices, values, strict=True):
            self.weights[index] += step * rho * error * value

        # The correction, which lands on the features of the next state rather
        # than this one. That is the whole difference between the two methods.
        pull = step * rho * self.discount * explained
        for index, value in zip(*ahead, strict=True):
            self.weights[index] -= pull * value

        # Divided by the same squared length as the other step size, so both
        # settings mean a share of what they move rather than a number whose
        # right size depends on the coder. They are otherwise independent:
        # neither one is a multiple of the other.
        change = self.shared_out(self.helper_step, values) * rho * (error - explained)
        for index, value in zip(indices, values, strict=True):
            self.helper[index] += change * value


__all__ = [
    "GradientTD",
    "LinearPredictor",
    "Policy",
    "SemiGradientTD",
    "fixed",
]
