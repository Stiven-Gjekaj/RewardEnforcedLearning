"""Turning gradients into changes to the weights.

Two of them. Plain gradient descent with momentum, which is what the older
literature uses and which needs its step size choosing carefully, and Adam,
which mostly does not.

## Why the step is clipped

Both take a `clip`. A policy gradient produces the occasional enormous
gradient: one episode goes far better than the running average, its return
multiplies every log probability along it, and a single step moves the weights
further than the whole run so far. The policy after that step is not a better
policy. It is a different one, and everything learned is gone.

Clipping the length of the whole gradient, rather than each number in it, keeps
the direction and shortens the step. That distinction matters: clipping each
number separately changes which way the step points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rel.nn.autograd import Tensor


def gradient_length(parameters: Sequence[Tensor]) -> float:
    """The length of the gradient, across every parameter at once."""
    total = 0.0
    for tensor in parameters:
        for value in tensor.grad:
            total += value * value
    return math.sqrt(total)


class Optimiser:
    """What every rule for changing weights shares."""

    def __init__(self, parameters: Sequence[Tensor], clip: float = 0.0) -> None:
        if not parameters:
            raise ValueError("An optimiser needs something to optimise.")
        self.parameters = list(parameters)
        self.clip = clip
        self.steps = 0

    def zero_grad(self) -> None:
        for tensor in self.parameters:
            tensor.zero_grad()

    def _shrink(self) -> float:
        """How much to shorten this step by, to keep the gradient short."""
        if self.clip <= 0.0:
            return 1.0
        length = gradient_length(self.parameters)
        if length <= self.clip:
            return 1.0
        return self.clip / length

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimiser):
    """Move against the gradient, with a memory of the last few moves."""

    def __init__(
        self,
        parameters: Sequence[Tensor],
        step_size: float = 0.01,
        momentum: float = 0.0,
        clip: float = 0.0,
    ) -> None:
        super().__init__(parameters, clip)
        self.step_size = step_size
        self.momentum = momentum
        self.speed: list[list[float]] = [
            [0.0] * len(tensor.data) for tensor in self.parameters
        ]

    def step(self) -> None:
        self.steps += 1
        shrink = self._shrink()

        for index, tensor in enumerate(self.parameters):
            speed = self.speed[index]
            for position, gradient in enumerate(tensor.grad):
                change = gradient * shrink
                speed[position] = self.momentum * speed[position] + change
                tensor.data[position] -= self.step_size * speed[position]


class Adam(Optimiser):
    """Keeps a running average of the gradient and of its square.

    The step for each weight is the first divided by the root of the second, so
    a weight whose gradient is small and steady moves as far as one whose
    gradient is large and jumps about. That is why it needs less care over the
    step size than plain descent, and why the same step size works across
    problems that differ by orders of magnitude in reward.

    The two averages both start at zero, which makes them far too small for the
    first few steps. Both are divided by how much of that starting zero is
    left, which is the correction in the paper and is what stops the first
    step being tiny.
    """

    def __init__(
        self,
        parameters: Sequence[Tensor],
        step_size: float = 0.001,
        decay: float = 0.9,
        square_decay: float = 0.999,
        epsilon: float = 1e-8,
        clip: float = 0.0,
    ) -> None:
        super().__init__(parameters, clip)
        self.step_size = step_size
        self.decay = decay
        self.square_decay = square_decay
        self.epsilon = epsilon

        self.average: list[list[float]] = [
            [0.0] * len(tensor.data) for tensor in self.parameters
        ]
        self.square: list[list[float]] = [
            [0.0] * len(tensor.data) for tensor in self.parameters
        ]

    def step(self) -> None:
        self.steps += 1
        shrink = self._shrink()

        first_left = 1.0 - self.decay**self.steps
        second_left = 1.0 - self.square_decay**self.steps

        for index, tensor in enumerate(self.parameters):
            average = self.average[index]
            square = self.square[index]

            for position, raw in enumerate(tensor.grad):
                gradient = raw * shrink
                average[position] = (
                    self.decay * average[position] + (1.0 - self.decay) * gradient
                )
                square[position] = (
                    self.square_decay * square[position]
                    + (1.0 - self.square_decay) * gradient * gradient
                )

                corrected = average[position] / first_left
                spread = math.sqrt(square[position] / second_left)
                tensor.data[position] -= (
                    self.step_size * corrected / (spread + self.epsilon)
                )


__all__ = ["SGD", "Adam", "Optimiser", "gradient_length"]
