"""A network small enough to run in pure Python, written out.

One hidden layer and a softmax on the end. That is enough for a policy over a
handful of actions, and it is about as much as pure Python can run at a speed
worth waiting for.

## How the weights start

Each weight is drawn from a normal distribution scaled by one over the square
root of the number of inputs to its layer.

The reason is the size of the output. A unit with `n` inputs adds up `n`
products, and if each weight has a spread of one then the sum has a spread of
the square root of `n`. Ten inputs give a spread near three, `tanh` of three is
0.995, and the gradient there is 0.01. The layer starts flat and learns almost
nothing for a long time.

Dividing by the square root of `n` holds the spread near one whatever the width
is. `test_the_output_of_a_fresh_layer_is_not_saturated` measures it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rel.nn.autograd import Tensor, linear, log_softmax, relu, tanh
from rel.rng import Rng


class Linear:
    """One layer: a weight, a bias, and nothing else."""

    __slots__ = ("bias", "inputs", "outputs", "weight")

    def __init__(self, rng: Rng, inputs: int, outputs: int, gain: float = 1.0) -> None:
        if inputs < 1 or outputs < 1:
            raise ValueError("A layer needs at least one input and one output.")

        self.inputs = inputs
        self.outputs = outputs

        spread = gain / math.sqrt(inputs)
        self.weight = Tensor(
            [rng.normal(0.0, spread) for _ in range(outputs * inputs)],
            (outputs, inputs),
            name="weight",
        )
        self.bias = Tensor([0.0] * outputs, (outputs,), name="bias")

    def __call__(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]

    def __repr__(self) -> str:
        return f"Linear({self.inputs} -> {self.outputs})"


class PolicyNetwork:
    """Observation in, log probability of each action out.

    Log probabilities rather than probabilities, because the thing a policy
    gradient needs is the log and taking it afterwards throws away accuracy
    exactly where it matters: an action the policy has almost stopped taking
    has a probability that rounds to zero and a log that does not.
    """

    __slots__ = ("first", "hidden", "second")

    def __init__(self, rng: Rng, inputs: int, actions: int, hidden: int = 16) -> None:
        self.hidden = hidden
        self.first = Linear(rng, inputs, hidden)
        # A small last layer, so the policy starts near uniform. A policy that
        # starts confident explores nothing and takes a long time to be talked
        # out of it.
        self.second = Linear(rng, hidden, actions, gain=0.1)

    def __call__(self, observation: Sequence[float]) -> Tensor:
        x = Tensor(observation)
        return log_softmax(self.second(tanh(self.first(x))))

    def probabilities(self, observation: Sequence[float]) -> list[float]:
        return [math.exp(value) for value in self(observation).data]

    def parameters(self) -> list[Tensor]:
        return self.first.parameters() + self.second.parameters()

    def __repr__(self) -> str:
        return (
            f"PolicyNetwork({self.first.inputs} -> {self.hidden} -> "
            f"{self.second.outputs})"
        )


class ValueNetwork:
    """Observation in, one number out. The baseline of a policy gradient."""

    __slots__ = ("first", "hidden", "second")

    def __init__(self, rng: Rng, inputs: int, hidden: int = 16) -> None:
        self.hidden = hidden
        self.first = Linear(rng, inputs, hidden)
        self.second = Linear(rng, hidden, 1, gain=0.1)

    def __call__(self, observation: Sequence[float]) -> Tensor:
        x = Tensor(observation)
        return self.second(relu(self.first(x)))

    def parameters(self) -> list[Tensor]:
        return self.first.parameters() + self.second.parameters()

    def __repr__(self) -> str:
        return f"ValueNetwork({self.first.inputs} -> {self.hidden} -> 1)"


__all__ = ["Linear", "PolicyNetwork", "ValueNetwork"]
