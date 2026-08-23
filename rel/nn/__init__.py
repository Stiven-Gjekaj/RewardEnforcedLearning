"""A network and the arithmetic to train it, written out.

Nothing here is imported by an environment, and nothing here knows what a
reward is. It is a gradient engine and two networks. `rel/agents/policy.py`
puts them to work.
"""

from __future__ import annotations

from rel.nn.autograd import Tensor, linear, log_softmax, relu, select, tanh
from rel.nn.layers import Linear, PolicyNetwork, ValueNetwork
from rel.nn.optim import SGD, Adam, Optimiser

__all__ = [
    "SGD",
    "Adam",
    "Linear",
    "Optimiser",
    "PolicyNetwork",
    "Tensor",
    "ValueNetwork",
    "linear",
    "log_softmax",
    "relu",
    "select",
    "tanh",
]
