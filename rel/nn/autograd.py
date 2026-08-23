"""Reverse mode automatic differentiation, over vectors.

A policy gradient method needs the gradient of the log probability of one
action with respect to every weight of a network. Writing that out by hand for
a two layer network is possible and it is the sort of thing that is wrong for
six months without anybody noticing, because a wrong gradient still points
somewhere and the returns still go up a little.

So the gradient is worked out by the same rule every time, and
`tests/test_autograd.py` checks each operation against the definition of a
derivative: move the input by a hair in each direction, see how much the output
moved, and compare that with what the operation said. Every operation here is
checked that way, at several points, including the ones near a kink.

## Vectors, not scalars

The well known small autograd libraries carry one number per node. That is the
clearest way to write one and it is about fifty times slower than it needs to
be here, because Python spends its time on the objects rather than on the
arithmetic. A node here holds a whole vector or a whole matrix, so a layer of
sixteen units is one node and not sixteen.

## The graph is one step wide

A policy gradient over an episode of five hundred steps is a sum of five
hundred terms, each with its own graph. Building all of them and calling
`backward` once would hold five hundred graphs in memory at once.

Gradients add, so the same answer comes from building one term, calling
`backward`, and letting the gradient pile up on the parameters. That is what
the agents do. Nothing here has to know about it, and it is worth knowing that
it is why nothing here is built to hold a long graph.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

Backward = Callable[[], None]


class Tensor:
    """A vector or a matrix, and how it was made.

    `shape` is `(n,)` for a vector and `(rows, columns)` for a matrix. The data
    is flat in both cases, in row order.
    """

    __slots__ = ("_backward", "_parents", "data", "grad", "name", "shape")

    def __init__(
        self,
        data: Sequence[float],
        shape: tuple[int, ...] | None = None,
        parents: tuple[Tensor, ...] = (),
        backward: Backward | None = None,
        name: str = "",
    ) -> None:
        self.data = [float(value) for value in data]
        self.shape = shape if shape is not None else (len(self.data),)

        if _size(self.shape) != len(self.data):
            raise ValueError(
                f"A tensor of shape {self.shape} holds {_size(self.shape)} "
                f"numbers and {len(self.data)} were given."
            )

        self.grad = [0.0] * len(self.data)
        self._parents = parents
        self._backward = backward
        self.name = name

    # -- Reading ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        head = ", ".join(f"{value:.4g}" for value in self.data[:4])
        more = ", ..." if len(self.data) > 4 else ""
        label = f" {self.name}" if self.name else ""
        return f"Tensor{label}({head}{more}, shape={self.shape})"

    def item(self) -> float:
        """The only number in this tensor."""
        if len(self.data) != 1:
            raise ValueError(
                f"item() needs a tensor of one number and this holds {len(self.data)}."
            )
        return self.data[0]

    # -- Gradients ----------------------------------------------------------

    def zero_grad(self) -> None:
        self.grad = [0.0] * len(self.data)

    def backward(self) -> None:
        """Work the gradient back from this tensor to every leaf.

        This tensor must hold one number, because the gradient of a vector is
        a matrix and nothing here needs one.
        """
        if len(self.data) != 1:
            raise ValueError(
                "backward() starts from one number. Reduce to a scalar first."
            )

        ordered = _topological(self)
        self.grad = [1.0]
        for node in reversed(ordered):
            if node._backward is not None:
                node._backward()


def _size(shape: tuple[int, ...]) -> int:
    total = 1
    for side in shape:
        total *= side
    return total


def _topological(root: Tensor) -> list[Tensor]:
    """Every node this one was made from, parents before children.

    The walk is a loop with its own stack rather than a recursive function. A
    graph here is at most a few dozen nodes deep, so recursion would work, and
    a stack costs one extra list and never runs out.
    """
    ordered: list[Tensor] = []
    seen: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        if expanded:
            ordered.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))

        stack.append((node, True))
        for parent in node._parents:
            if id(parent) not in seen:
                stack.append((parent, False))

    return ordered


# -- The operations ----------------------------------------------------------


def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """`weight` times `x`, plus `bias`.

    `x` is a vector of `n`, `weight` is `(m, n)` and `bias` is a vector of `m`.
    """
    rows, columns = weight.shape
    if len(x) != columns:
        raise ValueError(f"A weight of {weight.shape} cannot take {len(x)} inputs.")
    if len(bias) != rows:
        raise ValueError(f"A weight of {weight.shape} needs a bias of {rows}.")

    out = [0.0] * rows
    for row in range(rows):
        base = row * columns
        total = bias.data[row]
        for column in range(columns):
            total += weight.data[base + column] * x.data[column]
        out[row] = total

    result = Tensor(out, (rows,), parents=(x, weight, bias))

    def backward() -> None:
        for row in range(rows):
            share = result.grad[row]
            if share == 0.0:
                continue
            base = row * columns
            bias.grad[row] += share
            for column in range(columns):
                weight.grad[base + column] += x.data[column] * share
                x.grad[column] += weight.data[base + column] * share

    result._backward = backward
    return result


def tanh(x: Tensor) -> Tensor:
    """The hyperbolic tangent, one element at a time."""
    out = [math.tanh(value) for value in x.data]
    result = Tensor(out, x.shape, parents=(x,))

    def backward() -> None:
        for index, value in enumerate(out):
            x.grad[index] += (1.0 - value * value) * result.grad[index]

    result._backward = backward
    return result


def relu(x: Tensor) -> Tensor:
    """Zero below zero, and the value itself above it.

    The derivative at exactly zero does not exist. Zero is used, which is the
    usual choice and is worth saying rather than leaving for a reader to
    discover from the code.
    """
    out = [value if value > 0.0 else 0.0 for value in x.data]
    result = Tensor(out, x.shape, parents=(x,))

    def backward() -> None:
        for index, value in enumerate(x.data):
            if value > 0.0:
                x.grad[index] += result.grad[index]

    result._backward = backward
    return result


def log_softmax(x: Tensor) -> Tensor:
    """The log of the softmax, worked out without ever taking the softmax.

    Taking the softmax and then the log overflows twice over. The exponential
    of a large number is infinity, and the log of a small one is minus
    infinity, and a policy that has learned anything at all produces both.

    Subtracting the largest value first is exact: it changes nothing about the
    answer, and it makes every exponential at most one.
    """
    largest = max(x.data)
    shifted = [value - largest for value in x.data]
    total = math.log(sum(math.exp(value) for value in shifted))
    out = [value - total for value in shifted]

    result = Tensor(out, x.shape, parents=(x,))

    def backward() -> None:
        share = sum(result.grad)
        for index, value in enumerate(out):
            x.grad[index] += result.grad[index] - math.exp(value) * share

    result._backward = backward
    return result


def exp(x: Tensor) -> Tensor:
    """The exponential, one element at a time.

    The derivative of the exponential is itself, so the backward pass reads the
    result rather than recomputing anything.
    """
    out = [math.exp(value) for value in x.data]
    result = Tensor(out, x.shape, parents=(x,))

    def backward() -> None:
        for index, value in enumerate(out):
            x.grad[index] += value * result.grad[index]

    result._backward = backward
    return result


def select(x: Tensor, index: int) -> Tensor:
    """One element, as a tensor of one number."""
    if not 0 <= index < len(x):
        raise IndexError(f"{index} is outside a tensor of {len(x)}.")

    result = Tensor([x.data[index]], (1,), parents=(x,))

    def backward() -> None:
        x.grad[index] += result.grad[0]

    result._backward = backward
    return result


def scale(x: Tensor, factor: float) -> Tensor:
    """Every element times the same number."""
    result = Tensor([value * factor for value in x.data], x.shape, parents=(x,))

    def backward() -> None:
        for index in range(len(x.data)):
            x.grad[index] += factor * result.grad[index]

    result._backward = backward
    return result


def add(a: Tensor, b: Tensor) -> Tensor:
    """Two tensors of the same shape, added element by element."""
    if a.shape != b.shape:
        raise ValueError(f"{a.shape} and {b.shape} cannot be added.")

    out = [first + second for first, second in zip(a.data, b.data, strict=True)]
    result = Tensor(out, a.shape, parents=(a, b))

    def backward() -> None:
        for index, share in enumerate(result.grad):
            a.grad[index] += share
            b.grad[index] += share

    result._backward = backward
    return result


def multiply(a: Tensor, b: Tensor) -> Tensor:
    """Two tensors of the same shape, multiplied element by element."""
    if a.shape != b.shape:
        raise ValueError(f"{a.shape} and {b.shape} cannot be multiplied.")

    out = [first * second for first, second in zip(a.data, b.data, strict=True)]
    result = Tensor(out, a.shape, parents=(a, b))

    def backward() -> None:
        for index, share in enumerate(result.grad):
            a.grad[index] += b.data[index] * share
            b.grad[index] += a.data[index] * share

    result._backward = backward
    return result


def total(x: Tensor) -> Tensor:
    """Every element added together."""
    result = Tensor([sum(x.data)], (1,), parents=(x,))

    def backward() -> None:
        share = result.grad[0]
        for index in range(len(x.data)):
            x.grad[index] += share

    result._backward = backward
    return result


def square(x: Tensor) -> Tensor:
    """Every element times itself."""
    result = Tensor([value * value for value in x.data], x.shape, parents=(x,))

    def backward() -> None:
        for index, value in enumerate(x.data):
            x.grad[index] += 2.0 * value * result.grad[index]

    result._backward = backward
    return result


def zero_grads(tensors: Iterable[Tensor]) -> None:
    for tensor in tensors:
        tensor.zero_grad()


__all__ = [
    "Tensor",
    "add",
    "exp",
    "linear",
    "log_softmax",
    "multiply",
    "relu",
    "scale",
    "select",
    "square",
    "tanh",
    "total",
    "zero_grads",
]
