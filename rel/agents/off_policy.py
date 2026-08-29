"""Learning about one policy while following another.

Every agent in this project until now has learned about the policy it was
following, or about the greedy policy behind it. Neither can answer the
question this module is for: what is a policy worth, when the data was
collected by something else entirely?

That question is the whole of off-policy learning, and it is worth having
because the answer is almost always "we already have the data". A log of what
an old policy did, a human demonstration, a run from last week. None of it was
collected by the policy anybody now wants to evaluate.

## The two policies

The **behaviour** policy is what chooses the actions. The **target** policy is
what the agent is learning about. When they are the same policy this is the
on-policy learning the rest of the project does, and the correction below is
one at every step.

## The correction, and what it costs

A return collected under the behaviour policy is a sample of the wrong thing.
The fix is to weight it by how much more likely the target policy was to have
taken that whole sequence of actions, which is the product of the ratio at
every step.

A product of ratios is where the trouble is. A single step where the target
policy is ten times more likely than the behaviour one multiplies the whole
return by ten. Twenty such steps multiply it by ten to the twentieth.

Two estimators divide that product differently.

`ordinary` divides by the number of returns. It is unbiased, which is the
property a textbook wants, and its variance can be unbounded, which is the
property a practitioner notices.

`weighted` divides by the sum of the ratios. It is biased, and the bias goes
away as the returns pile up, and its variance is far smaller. This is the one
almost everybody uses, and `docs/algorithms.md` measures why.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rel.agents.base import TabularAgent, Transition
from rel.agents.explore import Rule
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete

Estimator = Literal["ordinary", "weighted"]
ESTIMATORS: tuple[Estimator, ...] = ("ordinary", "weighted")


def ratio(target: Sequence[float], behaviour: Sequence[float], action: int) -> float:
    """How much more likely the target policy was to take this action.

    Zero when the target policy would never take it, which cuts the whole
    return: nothing after an action the target policy does not take is evidence
    about the target policy.

    A behaviour policy that would never take an action the target policy might
    is not a behaviour policy this can learn from. That is coverage, and it is
    the one assumption off-policy learning cannot do without, so it raises
    rather than dividing by zero and carrying on.
    """
    if behaviour[action] <= 0.0:
        raise ValueError(
            "The behaviour policy never takes an action the target policy "
            "might. Without coverage there is nothing to correct."
        )
    return target[action] / behaviour[action]


class OffPolicyMonteCarlo(TabularAgent[ObsT]):
    """Monte Carlo control that learns the greedy policy from an exploring one.

    The behaviour policy is epsilon greedy, as everywhere else here. The target
    policy is the greedy one, so the agent learns what it would be worth to
    stop exploring, from data collected while exploring.

    ## Why the episode is walked backwards and cut short

    The target policy is greedy, so its probability on any action but the best
    one is zero. The moment the walk backwards reaches a step where the
    behaviour policy explored, the ratio is zero and every earlier step in the
    episode is multiplied by it.

    So the loop stops there. That is not an optimisation: it is the algorithm.
    The cost is that a long episode teaches only its tail, and the more the
    behaviour policy explores the shorter that tail is.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        step_size: float | Schedule | None = None,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
        estimator: Estimator = "weighted",
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=0.0 if step_size is None else step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator is one of {ESTIMATORS}. {estimator!r} is not.")

        self.averaging = step_size is None
        self.estimator = estimator
        #: The running denominator of each cell. A count for the ordinary
        #: estimator and a sum of the ratios for the weighted one, which is the
        #: only difference between them.
        self.weight: dict[tuple[ObsT, int], float] = {}
        self._episode: list[Transition[ObsT]] = []

    def target_probabilities(self, observation: ObsT) -> list[float]:
        """The greedy policy: all of the weight on the best action."""
        best = self.greedy(observation) - self.actions.start
        shares = [0.0] * self.actions.n
        shares[best] = 1.0
        return shares

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)
        self._episode.append(transition)

    def end_episode(self) -> None:
        if self._episode:
            self._learn()
            self._episode.clear()
        super().end_episode()

    def _learn(self) -> None:
        last = self._episode[-1]

        # A terminated episode has no future. The tail a step limit took away
        # does, and the target policy is greedy, so what it believes the
        # stopped state is worth is the best action there. Read without making
        # a row for it.
        total = 0.0 if last.terminated else max(self.peek(last.next_observation))

        importance = 1.0

        for transition in reversed(self._episode):
            total = transition.reward + self.discount * total

            key = (transition.observation, transition.action)
            offset = transition.action - self.actions.start
            row = self.values(transition.observation)

            if self.estimator == "weighted":
                weight = self.weight.get(key, 0.0) + importance
                self.weight[key] = weight
                if weight > 0.0:
                    row[offset] += (importance / weight) * (total - row[offset])
            elif self.averaging:
                count = self.weight.get(key, 0.0) + 1.0
                self.weight[key] = count
                row[offset] += (importance * total - row[offset]) / count
            else:
                row[offset] += self.current_step_size() * (
                    importance * total - row[offset]
                )

            target = self.target_probabilities(transition.observation)
            behaviour = self.policy_probabilities(transition.observation)
            importance *= ratio(target, behaviour, offset)

            if importance == 0.0:
                # The behaviour policy explored here. Nothing before this step
                # is evidence about a greedy target, so the walk stops.
                break

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(estimator={self.estimator!r}, "
            f"discount={self.discount:g}, epsilon={self.current_epsilon():g})"
        )


__all__ = ["ESTIMATORS", "Estimator", "OffPolicyMonteCarlo", "ratio"]
