"""A policy over a box of actions: where to aim, and how wide to spread.

Every other agent here chooses from a list. It keeps a number for each action
and ranks them, or a share for each action and draws one. Neither works when
the action is a number, because there is no list to rank and no share to keep.

What replaces it is a distribution with parameters. This one is a normal
distribution for each dimension of the action, so the policy is two numbers a
dimension:

    the mean     where to aim, from a network reading the observation
    the spread   how far from that to wander, learned and the same everywhere

An action is a draw from it, and the gradient of the log of its density is what
the policy gradient needs. Nothing else about the method changes: the weight on
each step is still how much better the step turned out than the value network
expected, and the update still pushes up whatever led to that.

## Why the mean is squashed

The network's output is put through a tanh and then stretched onto the box, so
the mean is always inside it. Without that the mean can walk out of the box,
every draw then clips to the same bound, and the gradient of a clipped action
keeps pushing it further out. That is a real failure and not a theoretical one.

A draw can still land outside and is clipped, and the log density is read at
the action the environment was given rather than at the draw. That is a bias.
It is bounded because the mean cannot leave the box, and it is the price of
not needing a second distribution to do the squashing properly.

## Why the spread is learned and not scheduled

Exploration here is the width of the distribution, so a fixed width is a fixed
exploration rate that somebody has to choose and decay. Learning it instead
lets the same gradient that moves the mean decide how sure to be, and a policy
that has found something good narrows on its own.

It is one number for each dimension rather than a number for each state,
because a spread that reads the observation gives the agent a way to make the
log density large by making the spread small, everywhere, at once.

## What the settings are, and what they were measured against

The pendulum is the only environment here with a box of actions, so every
default below is what did best on it over three seeds of four hundred
episodes, and none of them is a claim about anything else.

    step size 0.05, spread step 0.01     -1092
    step size 0.05, spread step 0.003    -1397
    step size 0.02, spread step 0.01     -1356
    step size 0.01, spread step 0.01     -1424

An entropy bonus does not help here. At 0.01 it is level, at 0.05 it is worse,
and at 0.2 the spread runs away to thirteen on a box four wide. So the default
is none, and the setting stays because the discrete actor critic needs one on
the cart pole and the reason is the same.

**None of those numbers is a policy that solved anything.** Doing nothing at
all scores -1187 on this problem. `docs/algorithms.md` has what does work on
it and what this measurement says about the method rather than about this
implementation.

## What is dropped from the log density

The constant, which is half the log of two pi for each dimension. Its gradient
is zero, so leaving it out changes no update. Nothing here prints a loss, so
nothing reads a number that would be wrong without it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from rel.agents.base import Agent, Transition
from rel.agents.features import Encoder
from rel.core import DIGEST_FIGURES, ObsT
from rel.nn.autograd import Tensor, add, exp, multiply, scale, square, tanh, total
from rel.nn.layers import QNetwork, ValueNetwork
from rel.nn.optim import Adam
from rel.rng import Rng
from rel.spaces import Box

Torque = tuple[float, ...]


class GaussianActorCritic(Agent[ObsT, Torque]):
    """An actor critic whose policy is a normal distribution over a box."""

    def __init__(
        self,
        rng: Rng,
        actions: Box,
        encoder: Encoder,
        features: int,
        *,
        hidden: int = 32,
        step_size: float = 0.05,
        value_step_size: float = 0.05,
        spread_step_size: float = 0.01,
        discount: float = 0.99,
        spread: float = 0.5,
        entropy: float = 0.0,
        normalise: bool = True,
        clip: float = 1.0,
    ) -> None:
        super().__init__(rng, actions)

        if spread <= 0.0:
            raise ValueError("A spread of zero is a policy that never explores.")

        self.box = actions
        self.encoder = encoder
        self.discount = discount
        self.entropy = entropy
        self.normalise = normalise

        # The middle of the box and half its width, which turn a tanh into a
        # mean inside it.
        self.middle = tuple(
            (low + high) / 2.0
            for low, high in zip(actions.low, actions.high, strict=True)
        )
        self.reach = tuple(
            (high - low) / 2.0
            for low, high in zip(actions.low, actions.high, strict=True)
        )

        self.means = QNetwork(rng, features, actions.dimensions, hidden)
        self.log_spread = Tensor(
            [math.log(spread)] * actions.dimensions, name="log spread"
        )
        self.value = ValueNetwork(rng, features, hidden)

        self.optimiser = Adam(self.means.parameters(), step_size, clip=clip)
        self.spread_optimiser = Adam([self.log_spread], spread_step_size, clip=clip)
        self.value_optimiser = Adam(self.value.parameters(), value_step_size, clip=clip)

        self._episode: list[Transition[ObsT, Torque]] = []

    # -- Acting -------------------------------------------------------------

    def mean(self, features: Sequence[float]) -> Tensor:
        """Where to aim, always inside the box."""
        squashed = tanh(self.means(features))
        return add(multiply(squashed, Tensor(self.reach)), Tensor(self.middle))

    def spread(self) -> tuple[float, ...]:
        """How far from the mean this policy wanders, in each dimension."""
        return tuple(math.exp(value) for value in self.log_spread.data)

    def act(self, observation: ObsT) -> Torque:
        aim = self.mean(self.encoder(observation)).data
        drawn = [
            middle + self.rng.normal(0.0, width)
            for middle, width in zip(aim, self.spread(), strict=True)
        ]
        return self.box.clip(drawn)

    def greedy(self, observation: ObsT) -> Torque:
        # The mean with no draw, which is what an evaluation run wants. Not
        # clipped: a tanh is between minus one and one, so the mean is between
        # the bounds and lands on one of them only where the tanh has
        # saturated, which the box accepts.
        return tuple(self.mean(self.encoder(observation)).data)

    #: A box has no list of actions to be worth anything, so there is nothing
    #: for a renderer to rank.
    def action_values(self, observation: ObsT) -> Sequence[float] | None:
        return None

    def state_value(self, observation: ObsT) -> float:
        return float(self.value(self.encoder(observation)).item())

    # -- Learning -----------------------------------------------------------

    def observe(self, transition: Transition[ObsT, Torque]) -> None:
        super().observe(transition)
        self._episode.append(transition)

    def end_episode(self) -> None:
        if self._episode:
            self._learn()
            self._episode.clear()
        super().end_episode()

    def _tail(self, transition: Transition[ObsT, Torque]) -> float:
        """What the rest of an episode the step limit stopped is worth."""
        return float(self.value(self.encoder(transition.next_observation)).item())

    def _targets_and_weights(
        self, features: Sequence[Sequence[float]]
    ) -> tuple[list[float], list[float]]:
        """The one step target for each state, and how hard to push each action.

        The same arithmetic as `rel.agents.policy.ActorCritic`. A step that
        ended has no future, a step the limit cut off has one and the value
        network is asked for it, and everything between reads the value of
        where it landed.
        """
        here = [float(self.value(x).item()) for x in features]
        targets: list[float] = []

        for index, step in enumerate(self._episode):
            if step.terminated:
                ahead = 0.0
            elif index + 1 < len(here):
                ahead = here[index + 1]
            else:
                ahead = self._tail(step)
            targets.append(step.reward + self.discount * ahead)

        return targets, [
            target - guess for target, guess in zip(targets, here, strict=True)
        ]

    def _learn(self) -> None:
        features = [self.encoder(step.observation) for step in self._episode]
        targets, weights = self._targets_and_weights(features)

        if self.normalise and len(weights) > 1:
            weights = standardised(weights)

        self.optimiser.zero_grad()
        self.spread_optimiser.zero_grad()
        for step, x, weight in zip(self._episode, features, weights, strict=True):
            loss = scale(self._log_density(x, step.action), -weight)

            if self.entropy > 0.0:
                # An entropy bonus, which for a normal distribution is the log
                # of its spread and a constant. Subtracting it from the loss
                # pushes the spread up, which is the same thing the entropy
                # setting of `rel.agents.policy` does and it is here for the
                # same reason: without it a policy that has found anything at
                # all narrows onto it, and on the pendulum it narrows onto a
                # constant torque before it has ever swung.
                loss = add(loss, scale(total(self.log_spread), -self.entropy))

            loss.backward()
        self.optimiser.step()
        self.spread_optimiser.step()

        self.value_optimiser.zero_grad()
        for x, target in zip(features, targets, strict=True):
            difference = add(self.value(x), scale(Tensor([target]), -1.0))
            total(square(difference)).backward()
        self.value_optimiser.step()

    def _log_density(self, features: Sequence[float], action: Torque) -> Tensor:
        """The log of how likely this policy was to draw this action.

        Without the constant, which has no gradient. What is left is minus a
        half of the squared distance in spreads, minus the log of the spread.
        """
        gap = add(Tensor(action), scale(self.mean(features), -1.0))
        in_spreads = multiply(gap, exp(scale(self.log_spread, -1.0)))
        return add(
            scale(total(square(in_spreads)), -0.5),
            scale(total(self.log_spread), -1.0),
        )

    def learned(self) -> Iterator[str]:
        # Every weight of both networks and the spread, which is a weight of
        # its own and the one a reader is most likely to want.
        for name, net in (("means", self.means), ("value", self.value)):
            for index, tensor in enumerate(net.parameters()):
                row = ",".join(f"{value:.{DIGEST_FIGURES}g}" for value in tensor.data)
                yield f"{name}.{index}|{row}"
        row = ",".join(f"{value:.{DIGEST_FIGURES}g}" for value in self.log_spread.data)
        yield f"log spread|{row}"

    def __repr__(self) -> str:
        spread = ",".join(f"{width:.3g}" for width in self.spread())
        return f"GaussianActorCritic({self.means!r}, spread={spread})"


def standardised(values: Sequence[float]) -> list[float]:
    """The values with their mean taken off, divided by their spread.

    The same helper `rel.agents.policy` has, repeated rather than imported,
    because importing it would make this module depend on the discrete policy
    agents for four lines of arithmetic.
    """
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if spread > 1e-8:
        return [(value - mean) / spread for value in values]
    return [0.0] * len(values)


__all__ = ["GaussianActorCritic", "standardised"]
