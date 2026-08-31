"""Agents that learn a policy directly, rather than a value to act greedily on.

Everything else in this project learns what an action is worth and then takes
the best one. These learn how likely to make each action, and change those
likelihoods in the direction that made the return larger.

That buys two things and costs one. It gives a policy that can be genuinely
random where being predictable is bad, and it gives a smooth path from one
policy to a nearby one rather than the jump a greedy rule makes when two values
cross. It costs variance: the signal is the return of a whole episode, and one
lucky episode looks exactly like a good policy.

## What the gradient is

    change the log probability of the action that was taken,
    by how much better the return was than expected

Written out, the update to the weights is that number times the gradient of the
log probability of the action taken. Nothing about the reward is
differentiated. The environment is not a function that can be differentiated,
and it does not have to be.

## Why the returns are standardised

`normalise` takes the weights of an episode, subtracts their mean and divides
by their spread.

Subtracting the mean is a baseline, and it does not bias the gradient: the
expected gradient of a log probability is zero, so subtracting anything that
does not depend on the action leaves the direction alone and shrinks the noise.

Dividing by the spread does change the size of the step, and it is what makes
one step size work on an environment paying 1 a step and on one paying -100.
It is a real change to the algorithm rather than a tidy up, so it is a setting
rather than something done quietly.

## The graph is one step wide

An episode of five hundred steps gives a sum of five hundred terms, each with
its own graph. Building all of them and calling `backward` once would hold five
hundred graphs at the same time.

Gradients add, so the same answer comes from building one term, sending it
backwards, and letting the gradient pile up on the weights. That is what these
agents do, and it is why `rel.nn.autograd` is not built to hold a long graph.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from rel.agents.base import Agent, Transition
from rel.agents.features import Encoder
from rel.core import DIGEST_FIGURES, ObsT
from rel.nn.autograd import (
    Tensor,
    add,
    exp,
    multiply,
    scale,
    select,
    square,
    total,
)
from rel.nn.layers import PolicyNetwork, ValueNetwork
from rel.nn.optim import Adam
from rel.rng import Rng
from rel.spaces import Discrete


class Reinforce(Agent[ObsT]):
    """Waits for the episode, then pushes up whatever led to a good return.

    This is Williams' REINFORCE, with an optional learned baseline.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        encoder: Encoder,
        features: int,
        *,
        hidden: int = 16,
        step_size: float = 0.02,
        value_step_size: float = 0.05,
        discount: float = 0.99,
        baseline: bool = True,
        normalise: bool = True,
        episodic_weighting: bool = False,
        entropy: float = 0.0,
        clip: float = 1.0,
    ) -> None:
        super().__init__(rng, actions)

        self.encoder = encoder
        self.discount = discount
        self.normalise = normalise
        self.episodic_weighting = episodic_weighting
        self.entropy = entropy

        self.policy = PolicyNetwork(rng, features, actions.n, hidden)
        self.optimiser = Adam(self.policy.parameters(), step_size, clip=clip)

        self.value: ValueNetwork | None = None
        self.value_optimiser: Adam | None = None
        if baseline:
            self.value = ValueNetwork(rng, features, hidden)
            self.value_optimiser = Adam(
                self.value.parameters(), value_step_size, clip=clip
            )

        self._episode: list[Transition[ObsT, int]] = []

    # -- Acting -------------------------------------------------------------

    def probabilities(self, observation: ObsT) -> list[float]:
        return [
            math.exp(value) for value in self.policy(self.encoder(observation)).data
        ]

    def act(self, observation: ObsT) -> int:
        return self.actions.start + self.rng.weighted_index(
            self.probabilities(observation)
        )

    def greedy(self, observation: ObsT) -> int:
        shares = self.probabilities(observation)
        best = max(shares)
        tied = [index for index, share in enumerate(shares) if share == best]
        if len(tied) == 1:
            return self.actions.start + tied[0]
        return self.actions.start + tied[self.rng.below(len(tied))]

    #: These are shares and not returns. `rel.ui.grid.best_values` refuses to
    #: draw a value map from them, and the policy map is drawn as usual
    #: because it only takes the largest.
    values_are_returns = False

    def action_values(self, observation: ObsT) -> Sequence[float] | None:
        """The probabilities, which are not values and are not pretended to be.

        A renderer draws a policy from whatever this returns, and it takes the
        largest, so the policy map is right. A value map drawn from these would
        be a picture of confidence carrying the label of a value map, which is
        why `rel.ui.grid` asks for the value separately.
        """
        return self.probabilities(observation)

    # -- Learning -----------------------------------------------------------

    def observe(self, transition: Transition[ObsT, int]) -> None:
        super().observe(transition)
        self._episode.append(transition)

    def end_episode(self) -> None:
        if self._episode:
            self._learn()
            self._episode.clear()
        super().end_episode()

    def _tail(self, transition: Transition[ObsT, int]) -> float:
        """What the rest of an episode that the step limit stopped is worth.

        A cut off episode has a future. Calling that future zero is the fault
        this project warns about everywhere else, and on the cliff walk it is
        not a small one. See `_returns` for what it does to a run.

        With a baseline there is a value network to ask. Without one there is
        nothing that knows the state, so the estimate is the mean reward of the
        episode paid for ever. That needs a discount below one. The return of
        an undiscounted episode that never finishes is not a number, so that
        case gives zero, and it is the one case this method cannot answer.
        """
        if self.value is not None:
            return self.value(self.encoder(transition.next_observation)).item()
        if self.discount >= 1.0:
            return 0.0
        mean = sum(step.reward for step in self._episode) / len(self._episode)
        return mean / (1.0 - self.discount)

    def _returns(self) -> list[float]:
        """The discounted return from each step to the end of the episode.

        The last step of an episode is one of two things, and the difference
        decides what the return reaching back through it is. An episode that
        ended has no future and the return stops. An episode that the step
        limit cut off does have a future, and `_tail` estimates it.

        Reading both as an ending is what this method did for a while, and it
        is why REINFORCE never reached the goal on three seeds of six of the
        cliff walk. A failed episode there is five hundred steps of -1. Under
        a zero tail the return of the last step is -1 and the return of the
        first is -99.34, so standardising hands the last steps a weight near
        +3.2 and the first steps a weight near -0.8. The agent is told to
        repeat whatever it was doing when the limit stopped it. It circles,
        the circling is rewarded, and it circles harder.

        The guess written down at the time was that standardising removes the
        signal when every step is equally bad. The measurement says the
        opposite: the signal was large, and it pointed the wrong way.
        """
        running = 0.0
        backwards: list[float] = []
        for index, transition in enumerate(reversed(self._episode)):
            if transition.terminated:
                running = 0.0
            elif index == 0:
                running = self._tail(transition)
            running = transition.reward + self.discount * running
            backwards.append(running)
        backwards.reverse()
        return backwards

    def _targets_and_weights(
        self, features: Sequence[Sequence[float]]
    ) -> tuple[list[float], list[float]]:
        """What the value network should say, and how hard to push each action.

        Here the target is the return of the rest of the episode, and the
        weight is that return with the baseline taken off.
        """
        returns = self._returns()
        if self.value is None:
            return returns, list(returns)

        baseline = [self.value(x).item() for x in features]
        return returns, [
            total_return - guess
            for total_return, guess in zip(returns, baseline, strict=True)
        ]

    def _learn(self) -> None:
        features = [self.encoder(step.observation) for step in self._episode]
        targets, weights = self._targets_and_weights(features)

        if self.episodic_weighting:
            # The gradient of an episodic objective carries a discount for how
            # far into the episode the step is. Almost every implementation
            # drops it. This project makes it a setting rather than a silent
            # choice, and `docs/algorithms.md` has what it does to a run.
            weights = [
                weight * self.discount**step for step, weight in enumerate(weights)
            ]

        if self.normalise and len(weights) > 1:
            weights = standardised(weights)

        # `self.policy(x)` already gives log probabilities. Taking the log
        # softmax of them again is a fault this file carried for a while: it
        # still points roughly the right way, so the agent still learned and
        # the curves still went up, and nothing about a run said anything was
        # wrong. It was found by the linter noticing an unused import.
        self.optimiser.zero_grad()
        for step, x, weight in zip(self._episode, features, weights, strict=True):
            chosen = step.action - self.actions.start
            shares = self.policy(x)
            loss = scale(select(shares, chosen), -weight)

            if self.entropy > 0.0:
                # An entropy bonus. The sum of a probability times its own log
                # is the negative entropy, so adding it with a positive weight
                # pushes the policy away from certainty.
                #
                # Without it a policy that has found something good keeps
                # sharpening, and once it is nearly deterministic it stops
                # seeing what the other actions would have paid. The value
                # network is still wrong at that point, and the two together
                # walk the policy into a corner it cannot leave. On the cart
                # pole this is the difference between an agent that reaches
                # eight steps and one that reaches five hundred.
                loss = add(
                    loss,
                    scale(total(multiply(exp(shares), shares)), self.entropy),
                )

            loss.backward()
        self.optimiser.step()

        if self.value is not None and self.value_optimiser is not None:
            self.value_optimiser.zero_grad()
            for x, target in zip(features, targets, strict=True):
                predicted = self.value(x)
                difference = add(predicted, scale(Tensor([target]), -1.0))
                total(square(difference)).backward()
            self.value_optimiser.step()

    def learned(self) -> Iterator[str]:
        # Every weight of every layer, the baseline included. A network keeps
        # no table of states, so what it learned is its parameters, and two
        # runs that walked the same path can still have moved these apart.
        nets = [("policy", self.policy)] + (
            [("value", self.value)] if self.value is not None else []
        )
        for name, net in nets:
            for index, tensor in enumerate(net.parameters()):
                row = ",".join(f"{value:.{DIGEST_FIGURES}g}" for value in tensor.data)
                yield f"{name}.{index}|{row}"

    def __repr__(self) -> str:
        return f"Reinforce({self.policy!r}, baseline={self.value is not None})"


class ActorCritic(Reinforce[ObsT]):
    """Replaces the tail of the return with what the value network believes.

    REINFORCE waits for the episode to end, because it needs the whole return.
    This replaces everything after one step with what the value network
    currently says the next state is worth, which is available at once.

    The trade is the one that runs through this whole project. Waiting gives an
    unbiased number with a lot of noise in it. Bootstrapping gives a quieter
    number that is wrong by however wrong the value network is.

    ## Why this still waits for the episode to end

    The textbook version updates after every step. This one collects the
    episode and makes one update from it, and the reason is arithmetic rather
    than theory: an update touches every weight of both networks, and doing
    that once per step rather than once per episode made four hundred episodes
    of the cliff walk take 220 seconds instead of 12. Nothing about the targets
    changes. Each one is still one step of reward plus the value of where that
    step landed.

    The value is learned by semi-gradient: the target holds the value of the
    next state, and the gradient of that dependence is ignored. Following it as
    well is possible and it is not what this method does.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        encoder: Encoder,
        features: int,
        *,
        hidden: int = 16,
        step_size: float = 0.02,
        value_step_size: float = 0.05,
        discount: float = 0.99,
        normalise: bool = True,
        episodic_weighting: bool = False,
        entropy: float = 0.01,
        clip: float = 1.0,
    ) -> None:
        super().__init__(
            rng,
            actions,
            encoder,
            features,
            hidden=hidden,
            step_size=step_size,
            value_step_size=value_step_size,
            discount=discount,
            baseline=True,
            normalise=normalise,
            episodic_weighting=episodic_weighting,
            entropy=entropy,
            clip=clip,
        )

    def _targets_and_weights(
        self, features: Sequence[Sequence[float]]
    ) -> tuple[list[float], list[float]]:
        assert self.value is not None

        here = [self.value(x).item() for x in features]
        targets: list[float] = []

        for index, step in enumerate(self._episode):
            if step.terminated:
                ahead = 0.0
            elif index + 1 < len(here):
                ahead = here[index + 1]
            else:
                # The last step of an episode that was cut off rather than
                # ended. The state it stopped in has a future, and dropping it
                # is the fault this project warns about everywhere else.
                ahead = self._tail(step)
            targets.append(step.reward + self.discount * ahead)

        return targets, [
            target - guess for target, guess in zip(targets, here, strict=True)
        ]

    def __repr__(self) -> str:
        return f"ActorCritic({self.policy!r})"


def standardised(values: Sequence[float]) -> list[float]:
    """The values with their mean taken off, divided by their spread."""
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if spread > 1e-8:
        return [(value - mean) / spread for value in values]
    return [value - mean for value in values]


__all__ = ["ActorCritic", "Reinforce", "standardised"]
