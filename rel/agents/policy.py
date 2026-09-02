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

from rel.agents.base import DiscreteAgent, Transition
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


class Reinforce(DiscreteAgent[ObsT]):
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
        self._fit_value(features, targets)

    def _fit_value(
        self, features: Sequence[Sequence[float]], targets: Sequence[float]
    ) -> None:
        """One squared error step on the baseline, if there is one.

        Separate from `_learn` because `ClippedPolicy` reuses an episode
        several times for the policy and once for this, and the two would
        otherwise be the same six lines in two places.
        """
        if self.value is None or self.value_optimiser is None:
            return

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


class ClippedPolicy(Reinforce[ObsT]):
    """Reuses an episode several times, with a clip on how far any pass moves.

    `reinforce` takes one gradient step from an episode and throws it away. The
    episode cost real steps of a real environment and one step of gradient is
    very little to get for them, so the obvious thing is to take several. The
    obvious thing is also wrong: after the first step the policy is no longer
    the one that collected the episode, and the returns in it are the returns
    of a policy that no longer exists.

    ## What the clip is for

    The correction for using data from another policy is the ratio of the two
    probabilities of the action taken, and off-policy corrections built out of
    ratios are the thing this project measures going wrong elsewhere: a ratio
    of ten is a step ten times too large made on the evidence of one episode.

    A clipped objective bounds the ratio instead of trusting it. Where the
    ratio has moved past `clip_range` in the direction that would make the
    update larger, the objective stops improving, so the gradient is nothing
    and the pass leaves that step alone. Where it has not, the update is the
    ratio times the advantage, which is the off-policy gradient as written.

    So a pass can move any one step by a bounded amount, and the passes after
    the first only move the steps the earlier ones have not already moved far.

    ## Why there is no clip operation in the engine

    `min(ratio * A, clip(ratio) * A)` is a function of the ratio with two
    pieces. On one piece it is `ratio * A` and on the other it is a constant,
    and the gradient of a constant is nothing. So which piece applies is read
    off the numbers and the graph is built for that piece alone, which is the
    same gradient with no operation added to `rel.nn.autograd`.

    The clip binds when the advantage is positive and the ratio is above
    `1 + clip_range`, or the advantage is negative and the ratio is below
    `1 - clip_range`. Both mean the same thing: this pass would push further
    in a direction earlier passes have already pushed far.

    ## The old probabilities are read once

    An episode is collected under one policy, because these agents learn at the
    end of an episode and not during one. So the probabilities to compare
    against are the ones the policy has when `_learn` starts, read once before
    any pass, and they are the collecting policy exactly rather than an
    approximation of it.

    ## What it counts

    `clipped` and `considered` are how many step updates had the clip bind and
    how many there were. That share is the setting's own diagnostic: at zero
    the clip is doing nothing and the agent is plain repeated gradient, and
    near one it is doing everything and the passes past the first are wasted.
    `scripts/measure_clipped.py` reads it.
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
        passes: int = 4,
        clip_range: float = 0.2,
    ) -> None:
        if passes < 1:
            raise ValueError("An episode is used at least once.")
        if clip_range <= 0.0:
            raise ValueError("A clip range above nothing is what makes this a clip.")

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
        #: How many times each episode is walked over.
        self.passes = passes
        #: How far the ratio may move before the objective stops improving.
        self.clip_range = clip_range
        #: How many step updates had the clip bind, and how many there were.
        self.clipped = 0
        self.considered = 0

    @property
    def share_clipped(self) -> float:
        """The share of step updates the clip has bound on so far."""
        if self.considered == 0:
            return 0.0
        return self.clipped / self.considered

    def binds(self, ratio: float, weight: float) -> bool:
        """Whether the clip stops this step moving any further this pass."""
        if weight > 0.0:
            return ratio > 1.0 + self.clip_range
        if weight < 0.0:
            return ratio < 1.0 - self.clip_range
        # An advantage of exactly nothing moves the policy nowhere either way,
        # so there is nothing for a clip to stop.
        return False

    def _learn(self) -> None:
        features = [self.encoder(step.observation) for step in self._episode]
        targets, weights = self._targets_and_weights(features)

        if self.episodic_weighting:
            weights = [
                weight * self.discount**step for step, weight in enumerate(weights)
            ]

        if self.normalise and len(weights) > 1:
            weights = standardised(weights)

        chosen = [step.action - self.actions.start for step in self._episode]
        before = [
            select(self.policy(x), action).item()
            for x, action in zip(features, chosen, strict=True)
        ]

        for _ in range(self.passes):
            self.optimiser.zero_grad()
            for x, action, weight, was in zip(
                features, chosen, weights, before, strict=True
            ):
                shares = self.policy(x)
                here = select(shares, action)
                ratio = math.exp(here.item() - was)

                self.considered += 1
                stopped = self.binds(ratio, weight)
                if stopped:
                    self.clipped += 1

                loss = (
                    None if stopped else scale(exp(add(here, Tensor([-was]))), -weight)
                )
                if self.entropy > 0.0:
                    bonus = scale(total(multiply(exp(shares), shares)), self.entropy)
                    loss = bonus if loss is None else add(loss, bonus)

                if loss is not None:
                    loss.backward()
            self.optimiser.step()

        self._fit_value(features, targets)

    def __repr__(self) -> str:
        return (
            f"ClippedPolicy({self.policy!r}, passes={self.passes}, "
            f"clip_range={self.clip_range:g})"
        )


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


__all__ = ["ActorCritic", "ClippedPolicy", "Reinforce", "standardised"]
