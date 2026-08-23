"""Turning an observation into the numbers a network can take.

A network takes a vector of real numbers. A grid gives an integer and a cart
pole gives four floats, so something has to sit between them, and where it sits
decides what the network can learn.

## A grid cell is not a number

The obvious encoding of state 37 of a grid is the number 37. It is also wrong.
A network reading it would learn that state 37 is nearly the same as state 38,
which on a grid means the cell to the right of it, and nothing like state 25,
which is the cell directly above. The layout of the numbers has nothing to do
with the layout of the grid.

One value per state, all zero except the one, says nothing that is not true. It
costs a weight for every state, which is exactly the cost of a table, and that
is the honest price of a state space with no structure the network can use.

## A box is centred, not just scaled

Each dimension is moved onto minus one to one rather than zero to one.

A network whose inputs are all positive gets a gradient on the first layer that
points the same way for every weight of a unit, so the weights of that unit can
only all rise or all fall together. Centring the inputs removes it, and it
costs one subtraction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rel.spaces import Box, Discrete, Space

#: An observation in, a vector of numbers out.
Encoder = Callable[..., Sequence[float]]


def one_hot(size: int) -> Encoder:
    """One value for each state, all zero except the one it is in."""

    def encode(observation: int) -> Sequence[float]:
        if not 0 <= observation < size:
            raise IndexError(f"{observation} is outside a space of {size}.")
        values = [0.0] * size
        values[observation] = 1.0
        return values

    return encode


def centred(box: Box) -> Encoder:
    """Each dimension moved onto the range minus one to one."""

    def encode(observation: Sequence[float]) -> Sequence[float]:
        return [2.0 * value - 1.0 for value in box.scaled(box.clip(observation))]

    return encode


def encoder_for(space: Space[object]) -> tuple[Encoder, int]:
    """The encoder for this observation space, and how many numbers it gives."""
    if isinstance(space, Discrete):
        if space.start != 0:
            raise ValueError("A one hot encoding needs a space that starts at zero.")
        return one_hot(space.n), space.n
    if isinstance(space, Box):
        return centred(space), space.dimensions
    raise TypeError(f"There is no encoding for a {type(space).__name__}.")


__all__ = ["Encoder", "centred", "encoder_for", "one_hot"]
