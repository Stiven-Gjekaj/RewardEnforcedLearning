"""Tests for the layers and the two rules for changing weights."""

from __future__ import annotations

import math

import pytest

from rel.nn.autograd import Tensor
from rel.nn.layers import Linear, PolicyNetwork, ValueNetwork
from rel.nn.optim import SGD, Adam, gradient_length
from rel.rng import Rng


class TestLinear:
    def test_the_output_is_the_size_it_says(self) -> None:
        layer = Linear(Rng(1), 4, 3)
        assert len(layer(Tensor([1.0, 2.0, 3.0, 4.0]))) == 3

    def test_the_bias_starts_at_zero(self) -> None:
        # There is nothing a bias could be started at that is better than
        # nothing, and a bias drawn at random adds noise to the first steps for
        # no gain.
        assert Linear(Rng(1), 4, 3).bias.data == [0.0, 0.0, 0.0]

    def test_a_layer_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one input"):
            Linear(Rng(1), 0, 3)

    def test_the_same_seed_gives_the_same_weights(self) -> None:
        assert Linear(Rng(9), 4, 3).weight.data == Linear(Rng(9), 4, 3).weight.data

    def test_two_seeds_give_different_weights(self) -> None:
        assert Linear(Rng(9), 4, 3).weight.data != Linear(Rng(10), 4, 3).weight.data

    def test_a_wide_layer_does_not_start_saturated(self) -> None:
        # A unit with n inputs adds up n products. If each weight had a spread
        # of one, the sum would have a spread of the square root of n, and
        # tanh of three is 0.995 where the gradient is 0.01. Dividing the
        # spread by the square root of n holds the output near one whatever
        # the width is.
        for width in (4, 64, 512):
            layer = Linear(Rng(3), width, 8)
            source = Rng(4)
            inputs = Tensor([source.normal() for _ in range(width)])
            out = layer(inputs).data
            spread = math.sqrt(sum(value * value for value in out) / len(out))
            assert 0.4 < spread < 2.5, f"width {width} gave a spread of {spread:.2f}"


class TestPolicyNetwork:
    def test_the_probabilities_add_up_to_one(self) -> None:
        network = PolicyNetwork(Rng(1), 4, 3)
        assert sum(network.probabilities([0.1, -0.2, 0.3, 0.4])) == pytest.approx(1.0)

    def test_it_starts_close_to_uniform(self) -> None:
        # A policy that starts confident explores nothing and takes a long
        # time to be talked out of it.
        network = PolicyNetwork(Rng(1), 4, 3)
        source = Rng(2)
        for _ in range(50):
            shares = network.probabilities([source.normal() for _ in range(4)])
            assert max(shares) < 0.55, shares

    def test_it_gives_a_log_probability_for_each_action(self) -> None:
        out = PolicyNetwork(Rng(1), 4, 5)([0.0] * 4)
        assert len(out) == 5
        assert all(value <= 0.0 for value in out.data)

    def test_the_parameters_are_the_weights_and_the_biases(self) -> None:
        network = PolicyNetwork(Rng(1), 4, 3, hidden=8)
        sizes = sorted(len(tensor) for tensor in network.parameters())
        assert sizes == [3, 8, 8 * 3, 8 * 4]


class TestValueNetwork:
    def test_it_gives_one_number(self) -> None:
        assert len(ValueNetwork(Rng(1), 4)([0.0] * 4)) == 1

    def test_it_starts_near_zero(self) -> None:
        network = ValueNetwork(Rng(1), 4)
        source = Rng(2)
        for _ in range(20):
            value = network([source.normal() for _ in range(4)]).item()
            assert abs(value) < 1.0


def a_parameter(values: list[float], gradient: list[float]) -> Tensor:
    tensor = Tensor(values)
    tensor.grad = list(gradient)
    return tensor


class TestSGD:
    def test_it_moves_against_the_gradient(self) -> None:
        tensor = a_parameter([1.0, 2.0], [0.5, -0.5])
        SGD([tensor], step_size=0.1).step()
        assert tensor.data == pytest.approx([0.95, 2.05])

    def test_momentum_makes_the_second_step_longer(self) -> None:
        plain = a_parameter([0.0], [1.0])
        heavy = a_parameter([0.0], [1.0])
        without = SGD([plain], step_size=0.1)
        with_it = SGD([heavy], step_size=0.1, momentum=0.9)

        for _ in range(2):
            without.step()
            with_it.step()
            plain.grad = [1.0]
            heavy.grad = [1.0]

        assert heavy.data[0] < plain.data[0]

    def test_zero_grad_clears_every_parameter(self) -> None:
        tensor = a_parameter([1.0], [5.0])
        SGD([tensor]).zero_grad()
        assert tensor.grad == [0.0]

    def test_an_optimiser_with_nothing_to_do_is_refused(self) -> None:
        with pytest.raises(ValueError, match="something to optimise"):
            SGD([])


class TestAdam:
    def test_the_first_step_is_about_the_step_size(self) -> None:
        # The two averages start at zero, which makes them far too small for
        # the first few steps. The correction is what stops the first step
        # being tiny, and without it this would be nearer 0.0001.
        tensor = a_parameter([0.0], [3.0])
        Adam([tensor], step_size=0.01).step()
        assert abs(tensor.data[0] + 0.01) < 1e-6

    def test_the_size_of_the_gradient_hardly_matters(self) -> None:
        # This is what it is for. A weight with a small steady gradient moves
        # about as far as one with a large steady gradient, which is why the
        # same step size works across problems that differ by orders of
        # magnitude in reward.
        small = a_parameter([0.0], [1e-4])
        large = a_parameter([0.0], [1e4])
        for tensor in (small, large):
            optimiser = Adam([tensor], step_size=0.01)
            for _ in range(5):
                optimiser.step()
        # Not exactly equal: the small constant that keeps the division safe
        # is a larger share of a tiny gradient than of a huge one. Five steps
        # of 0.01 move both to about -0.05 and they differ in the sixth digit.
        assert abs(small.data[0] - large.data[0]) < 1e-4

    def test_it_moves_against_the_gradient(self) -> None:
        tensor = a_parameter([0.0], [-1.0])
        Adam([tensor], step_size=0.01).step()
        assert tensor.data[0] > 0.0


class TestClipping:
    def test_the_length_of_the_gradient_is_across_every_parameter(self) -> None:
        first = a_parameter([0.0], [3.0])
        second = a_parameter([0.0], [4.0])
        assert gradient_length([first, second]) == pytest.approx(5.0)

    def test_a_short_gradient_is_left_alone(self) -> None:
        tensor = a_parameter([0.0], [0.5])
        SGD([tensor], step_size=1.0, clip=10.0).step()
        assert tensor.data[0] == pytest.approx(-0.5)

    def test_a_long_gradient_is_shortened_to_the_limit(self) -> None:
        tensor = a_parameter([0.0], [50.0])
        SGD([tensor], step_size=1.0, clip=2.0).step()
        assert tensor.data[0] == pytest.approx(-2.0)

    def test_clipping_keeps_the_direction(self) -> None:
        # Clipping each number on its own would change which way the step
        # points. Clipping the length of the whole gradient does not, and the
        # difference is the whole reason to do it this way.
        first = a_parameter([0.0], [30.0])
        second = a_parameter([0.0], [40.0])
        optimiser = SGD([first, second], step_size=1.0, clip=5.0)
        optimiser.step()

        assert first.data[0] == pytest.approx(-3.0)
        assert second.data[0] == pytest.approx(-4.0)
        assert first.data[0] / second.data[0] == pytest.approx(30.0 / 40.0)


def test_a_network_learns_a_function() -> None:
    """The whole engine, end to end, on something with a known answer.

    The network is asked to say which of two inputs is larger. It is a small
    thing to learn and it needs the layer, the activation, the log softmax, the
    gradient and the optimiser all to be right at once.
    """
    rng = Rng(5)
    network = PolicyNetwork(rng, 2, 2, hidden=8)
    optimiser = Adam(network.parameters(), step_size=0.05)
    source = rng.stream("data")

    from rel.nn.autograd import log_softmax, scale, select

    for _ in range(600):
        optimiser.zero_grad()
        for _ in range(8):
            first, second = source.normal(), source.normal()
            answer = 0 if first > second else 1
            loss = scale(select(log_softmax(network([first, second])), answer), -1.0)
            loss.backward()
        optimiser.step()

    right = 0
    checking = Rng(6)
    for _ in range(200):
        first, second = checking.normal(), checking.normal()
        shares = network.probabilities([first, second])
        chosen = 0 if shares[0] > shares[1] else 1
        if chosen == (0 if first > second else 1):
            right += 1

    assert right > 180, f"{right} of 200"
