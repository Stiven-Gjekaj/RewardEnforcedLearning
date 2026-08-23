"""Tests for the gradient engine.

Every operation is checked against the definition of a derivative: move the
input by a hair in each direction, see how much the output moved, and compare
that with what the operation said.

This is the test that matters. A wrong gradient still points somewhere, so an
agent trained with one still learns a little and still produces a curve that
goes up. Nothing about a run says the gradient is wrong.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest

from rel.nn.autograd import (
    Tensor,
    add,
    linear,
    log_softmax,
    multiply,
    relu,
    scale,
    select,
    square,
    tanh,
    total,
    zero_grads,
)

#: How far to move an input when measuring a derivative. Too large and the
#: curvature shows up in the answer; too small and the difference disappears
#: into the last bits of a float. This is near the best there is for a double.
STEP = 1e-6


def check_gradient(
    make: Callable[[Sequence[Tensor]], Tensor],
    inputs: Sequence[Tensor],
    tolerance: float = 1e-5,
) -> None:
    """Compare what backward said with what moving the input actually does."""
    zero_grads(inputs)
    make(inputs).backward()
    analytic = [list(tensor.grad) for tensor in inputs]

    for index, tensor in enumerate(inputs):
        for position in range(len(tensor.data)):
            original = tensor.data[position]

            tensor.data[position] = original + STEP
            above = make(inputs).item()
            tensor.data[position] = original - STEP
            below = make(inputs).item()
            tensor.data[position] = original

            measured = (above - below) / (2.0 * STEP)
            said = analytic[index][position]
            assert abs(measured - said) < tolerance + abs(measured) * 1e-3, (
                f"input {index} position {position}: backward said {said:.8f} "
                f"and moving it gave {measured:.8f}"
            )


class TestTheTensor:
    def test_a_shape_that_does_not_match_the_data_is_refused(self) -> None:
        with pytest.raises(ValueError, match="holds 6 numbers"):
            Tensor([1.0, 2.0], (2, 3))

    def test_a_vector_takes_its_shape_from_its_length(self) -> None:
        assert Tensor([1.0, 2.0, 3.0]).shape == (3,)

    def test_item_needs_one_number(self) -> None:
        assert Tensor([4.0]).item() == 4.0
        with pytest.raises(ValueError, match="one number"):
            Tensor([1.0, 2.0]).item()

    def test_backward_starts_from_one_number(self) -> None:
        with pytest.raises(ValueError, match="starts from one number"):
            Tensor([1.0, 2.0]).backward()

    def test_the_gradient_starts_at_zero(self) -> None:
        assert Tensor([1.0, 2.0]).grad == [0.0, 0.0]

    def test_zero_grad_clears_it(self) -> None:
        tensor = Tensor([1.0, 2.0])
        tensor.grad = [5.0, 6.0]
        tensor.zero_grad()
        assert tensor.grad == [0.0, 0.0]

    def test_it_reads_back_something_useful(self) -> None:
        assert "shape=(2,)" in repr(Tensor([1.0, 2.0]))


class TestEachOperation:
    def test_linear(self) -> None:
        x = Tensor([0.5, -1.5, 2.0])
        weight = Tensor([0.1, -0.2, 0.3, 0.4, 0.5, -0.6], (2, 3))
        bias = Tensor([0.7, -0.8])
        check_gradient(lambda parts: total(linear(*parts)), [x, weight, bias])

    def test_tanh(self) -> None:
        # Including a large value, where the derivative is nearly zero, and a
        # small one, where it is nearly one.
        x = Tensor([0.0, 0.5, -1.5, 6.0, -6.0])
        check_gradient(lambda parts: total(tanh(parts[0])), [x])

    def test_relu_away_from_the_kink(self) -> None:
        x = Tensor([1.5, -2.0, 0.3, -0.1])
        check_gradient(lambda parts: total(relu(parts[0])), [x])

    def test_relu_at_the_kink_takes_the_lower_answer(self) -> None:
        # The derivative at exactly zero does not exist. Zero is used, which is
        # the usual choice, and it disagrees with a two sided measurement by
        # exactly a half. A test that measured it would fail for a reason that
        # is not a defect, so it is stated instead.
        x = Tensor([0.0])
        total(relu(x)).backward()
        assert x.grad == [0.0]

    def test_log_softmax(self) -> None:
        x = Tensor([1.0, -0.5, 2.5, 0.0])
        check_gradient(lambda parts: select(log_softmax(parts[0]), 2), [x])

    def test_log_softmax_summed(self) -> None:
        x = Tensor([0.3, -1.2, 0.8])
        check_gradient(lambda parts: total(log_softmax(parts[0])), [x])

    def test_select(self) -> None:
        x = Tensor([1.0, 2.0, 3.0])
        check_gradient(lambda parts: select(parts[0], 1), [x])

    def test_scale(self) -> None:
        x = Tensor([1.0, -2.0])
        check_gradient(lambda parts: total(scale(parts[0], -3.5)), [x])

    def test_add(self) -> None:
        a = Tensor([1.0, -2.0])
        b = Tensor([0.5, 4.0])
        check_gradient(lambda parts: total(add(parts[0], parts[1])), [a, b])

    def test_multiply(self) -> None:
        a = Tensor([1.5, -2.0])
        b = Tensor([0.5, 4.0])
        check_gradient(lambda parts: total(multiply(parts[0], parts[1])), [a, b])

    def test_square(self) -> None:
        x = Tensor([1.5, -2.0, 0.0])
        check_gradient(lambda parts: total(square(parts[0])), [x])

    def test_total(self) -> None:
        x = Tensor([1.0, 2.0, 3.0])
        check_gradient(lambda parts: total(parts[0]), [x])


class TestOperationsTogether:
    def test_a_whole_layer_and_a_softmax(self) -> None:
        # The graph a policy gradient really builds.
        x = Tensor([0.4, -0.7])
        first = Tensor([0.3, -0.1, 0.2, 0.5, -0.4, 0.6], (3, 2))
        first_bias = Tensor([0.0, 0.1, -0.1])
        second = Tensor([0.2, -0.3, 0.4, 0.1, 0.5, -0.2], (2, 3))
        second_bias = Tensor([0.05, -0.05])

        def build(parts: Sequence[Tensor]) -> Tensor:
            hidden = tanh(linear(parts[0], parts[1], parts[2]))
            return select(log_softmax(linear(hidden, parts[3], parts[4])), 1)

        check_gradient(build, [x, first, first_bias, second, second_bias])

    def test_a_node_used_twice_gets_both_shares(self) -> None:
        # A diamond. If the walk over the graph visited the shared node before
        # both of its children had passed their share back, half the gradient
        # would be lost.
        x = Tensor([2.0, 3.0])

        def build(parts: Sequence[Tensor]) -> Tensor:
            shared = tanh(parts[0])
            return total(add(scale(shared, 2.0), multiply(shared, shared)))

        check_gradient(build, [x])

    def test_a_squared_error_against_a_target(self) -> None:
        predicted = Tensor([0.4])
        target = Tensor([1.5])

        def build(parts: Sequence[Tensor]) -> Tensor:
            return total(square(add(parts[0], scale(target, -1.0))))

        check_gradient(build, [predicted])


class TestGradientsAdd:
    def test_two_backward_calls_pile_up(self) -> None:
        # This is what lets an episode of five hundred steps be learned one
        # step at a time rather than as one graph of five hundred terms.
        x = Tensor([1.0, 2.0])
        total(scale(x, 3.0)).backward()
        assert x.grad == [3.0, 3.0]
        total(scale(x, 3.0)).backward()
        assert x.grad == [6.0, 6.0]

    def test_one_graph_gives_the_same_answer_as_two(self) -> None:
        together = Tensor([1.0, -2.0])
        total(add(square(together), scale(together, 5.0))).backward()

        apart = Tensor([1.0, -2.0])
        total(square(apart)).backward()
        total(scale(apart, 5.0)).backward()

        assert together.grad == pytest.approx(apart.grad)


class TestLogSoftmaxIsSafe:
    def test_a_huge_input_does_not_overflow(self) -> None:
        # Taking the softmax and then the log overflows twice over: the
        # exponential of a large number is infinity, and the log of a small one
        # is minus infinity. A policy that has learned anything produces both.
        result = log_softmax(Tensor([900.0, 0.0, -900.0]))
        assert all(math.isfinite(value) for value in result.data)
        assert result.data[0] == pytest.approx(0.0)

    def test_the_probabilities_add_up_to_one(self) -> None:
        result = log_softmax(Tensor([1.3, -0.2, 4.0, 0.0]))
        assert sum(math.exp(value) for value in result.data) == pytest.approx(1.0)

    def test_adding_the_same_number_to_every_input_changes_nothing(self) -> None:
        first = log_softmax(Tensor([1.0, 2.0, 3.0])).data
        second = log_softmax(Tensor([101.0, 102.0, 103.0])).data
        assert first == pytest.approx(second)


class TestShapes:
    def test_a_weight_that_does_not_fit_the_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot take"):
            linear(Tensor([1.0, 2.0]), Tensor([1.0] * 9, (3, 3)), Tensor([0.0] * 3))

    def test_a_bias_that_does_not_fit_the_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs a bias of 3"):
            linear(Tensor([1.0] * 3), Tensor([1.0] * 9, (3, 3)), Tensor([0.0, 0.0]))

    def test_two_shapes_that_differ_cannot_be_added(self) -> None:
        with pytest.raises(ValueError, match="cannot be added"):
            add(Tensor([1.0, 2.0]), Tensor([1.0]))

    def test_selecting_outside_the_tensor_is_refused(self) -> None:
        with pytest.raises(IndexError):
            select(Tensor([1.0, 2.0]), 5)
