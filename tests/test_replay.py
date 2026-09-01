"""Tests for the buffer of remembered steps."""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.replay import FLOOR, Replay
from rel.rng import Rng


def step(number: int) -> Transition[int, int]:
    return Transition(number, 0, float(number), number + 1, False, False)


class TestKeeping:
    def test_it_holds_what_it_is_given(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4)
        buffer.add(step(1))
        buffer.add(step(2))
        assert len(buffer) == 2
        assert {one.observation for one in buffer.steps()} == {1, 2}

    def test_it_never_grows_past_its_capacity(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 3)
        for number in range(10):
            buffer.add(step(number))
        assert len(buffer) == 3

    def test_the_oldest_is_the_one_written_over(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 3)
        for number in range(5):
            buffer.add(step(number))
        assert {one.observation for one in buffer.steps()} == {2, 3, 4}

    def test_it_counts_everything_it_was_ever_given(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 2)
        for number in range(7):
            buffer.add(step(number))
        assert buffer.seen == 7
        assert len(buffer) == 2

    def test_a_buffer_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            Replay[int](Rng(1), 0)


class TestDrawing:
    def test_a_sample_is_the_size_asked_for(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 10)
        for number in range(10):
            buffer.add(step(number))
        assert len(buffer.sample(4)) == 4

    def test_it_draws_from_everything_it_holds(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 5)
        for number in range(5):
            buffer.add(step(number))
        drawn = {one.observation for _ in range(50) for one in buffer.sample(4).steps}
        assert drawn == {0, 1, 2, 3, 4}

    def test_a_short_buffer_gives_what_it_has(self) -> None:
        # The first few steps of a run are exactly when a buffer is short, and
        # an agent that could not learn until it was full would do nothing for
        # the first thousand steps.
        buffer: Replay[int] = Replay(Rng(1), 100)
        buffer.add(step(1))
        drawn = buffer.sample(8)
        assert len(drawn) == 8
        assert {one.observation for one in drawn.steps} == {1}

    def test_an_empty_buffer_gives_nothing(self) -> None:
        assert len(Replay[int](Rng(1), 10).sample(4)) == 0

    def test_a_sample_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            Replay[int](Rng(1), 10).sample(0)

    def test_the_same_seed_draws_the_same_sample(self) -> None:
        def drawn(seed: int) -> list[int]:
            buffer: Replay[int] = Replay(Rng(seed), 10)
            for number in range(10):
                buffer.add(step(number))
            return [one.observation for one in buffer.sample(6).steps]

        assert drawn(4) == drawn(4)
        assert drawn(4) != drawn(5)


class TestPriority:
    def test_a_power_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="zero to one"):
            Replay[int](Rng(1), 4, priority=-0.1)
        with pytest.raises(ValueError, match="zero to one"):
            Replay[int](Rng(1), 4, priority=1.5)

    def test_a_new_step_starts_at_the_largest_priority_seen(self) -> None:
        # A step nothing has been fitted to has no error yet, so starting it
        # anywhere below the top would let it sit in the buffer undrawn.
        buffer: Replay[int] = Replay(Rng(1), 4, priority=1.0)
        buffer.add(step(0))
        buffer.reprioritise([0], [9.0])
        buffer.add(step(1))
        assert buffer.priorities()[1] == pytest.approx(9.0 + FLOOR)

    def test_a_step_the_agent_got_right_keeps_a_floor(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4, priority=1.0)
        buffer.add(step(0))
        buffer.reprioritise([0], [0.0])
        assert buffer.priorities()[0] == pytest.approx(FLOOR)
        assert buffer.priorities()[0] > 0.0

    def test_the_sign_of_the_error_does_not_matter(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4, priority=1.0)
        buffer.add(step(0))
        buffer.add(step(1))
        buffer.reprioritise([0, 1], [3.0, -3.0])
        assert buffer.priorities()[0] == buffer.priorities()[1]

    def test_the_power_is_applied_to_the_error(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4, priority=0.5)
        buffer.add(step(0))
        buffer.reprioritise([0], [4.0])
        assert buffer.priorities()[0] == pytest.approx((4.0 + FLOOR) ** 0.5)

    def test_a_place_the_buffer_does_not_hold_is_refused(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4, priority=1.0)
        buffer.add(step(0))
        with pytest.raises(IndexError, match="place 2"):
            buffer.reprioritise([2], [1.0])

    def test_every_place_needs_one_error(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4, priority=1.0)
        buffer.add(step(0))
        with pytest.raises(ValueError, match="one error"):
            buffer.reprioritise([0], [1.0, 2.0])

    def test_the_batch_says_where_each_step_came_from(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 6, priority=1.0)
        for number in range(6):
            buffer.add(step(number))
        drawn = buffer.sample(20)
        assert len(drawn.places) == len(drawn.steps)
        for place, one in zip(drawn.places, drawn.steps, strict=True):
            assert buffer.steps()[place] is one

    def test_a_larger_error_is_drawn_more_often(self) -> None:
        buffer: Replay[int] = Replay(Rng(3), 4, priority=1.0)
        for number in range(4):
            buffer.add(step(number))
        buffer.reprioritise([0, 1, 2, 3], [1.0, 1.0, 1.0, 7.0])

        counted = [0, 0, 0, 0]
        for place in buffer.sample(4000).places:
            counted[place] += 1

        # Three steps at one and one at seven, so the last should take seven
        # tenths of the draws.
        assert counted[3] / 4000 == pytest.approx(0.7, abs=0.03)
        for place in range(3):
            assert counted[place] / 4000 == pytest.approx(0.1, abs=0.03)

    def test_a_step_with_the_floor_alone_is_still_reachable(self) -> None:
        buffer: Replay[int] = Replay(Rng(5), 2, priority=1.0)
        buffer.add(step(0))
        buffer.add(step(1))
        buffer.reprioritise([0, 1], [0.0, 0.0])
        # Both are at the floor, so the draw is even between them.
        assert set(buffer.sample(60).places) == {0, 1}

    def test_a_power_of_zero_draws_exactly_as_before(self) -> None:
        # The uniform path must spend the same draws it always has, or every
        # recorded run of this project moves.
        def drawn(priority: float) -> list[int]:
            buffer: Replay[int] = Replay(Rng(11), 8, priority=priority)
            for number in range(8):
                buffer.add(step(number))
            return list(buffer.sample(12).places)

        plain: Replay[int] = Replay(Rng(11), 8)
        for number in range(8):
            plain.add(step(number))
        assert drawn(0.0) == list(plain.sample(12).places)
        assert drawn(0.0) != drawn(1.0)

    def test_priority_ignores_the_weights_at_a_power_of_zero(self) -> None:
        buffer: Replay[int] = Replay(Rng(11), 4, priority=0.0)
        for number in range(4):
            buffer.add(step(number))
        buffer.reprioritise([0], [1000.0])

        counted = [0, 0, 0, 0]
        for place in buffer.sample(4000).places:
            counted[place] += 1
        for place in range(4):
            assert counted[place] / 4000 == pytest.approx(0.25, abs=0.03)

    def test_an_empty_buffer_gives_an_empty_batch(self) -> None:
        drawn = Replay[int](Rng(1), 10, priority=1.0).sample(4)
        assert len(drawn) == 0
        assert drawn.places == ()

    def test_it_says_what_it_holds(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4, priority=0.6, weighting=0.4)
        buffer.add(step(0))
        assert repr(buffer) == "Replay(1 of 4, seen 1, priority=0.6, weighting=0.4)"


class TestWeighting:
    def test_a_power_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="weighting power"):
            Replay[int](Rng(1), 4, priority=1.0, weighting=-0.1)
        with pytest.raises(ValueError, match="weighting power"):
            Replay[int](Rng(1), 4, priority=1.0, weighting=1.5)

    def test_weighting_without_a_priority_draw_is_refused(self) -> None:
        # It would correct nothing, so asking for it is a mistake worth saying.
        with pytest.raises(ValueError, match="needs one"):
            Replay[int](Rng(1), 4, weighting=0.5)

    def test_a_uniform_draw_needs_no_correction(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4)
        for number in range(4):
            buffer.add(step(number))
        assert buffer.sample(5).weights == (1.0,) * 5

    def test_no_weighting_leaves_every_step_counting_the_same(self) -> None:
        buffer: Replay[int] = Replay(Rng(1), 4, priority=1.0, weighting=0.0)
        for number in range(4):
            buffer.add(step(number))
        buffer.reprioritise([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0])
        assert buffer.sample(6).weights == (1.0,) * 6

    def test_the_step_drawn_most_often_counts_least(self) -> None:
        buffer: Replay[int] = Replay(Rng(2), 4, priority=1.0, weighting=1.0)
        for number in range(4):
            buffer.add(step(number))
        buffer.reprioritise([0, 1, 2, 3], [1.0, 1.0, 1.0, 9.0])

        drawn = buffer.sample(200)
        seen = dict(zip(drawn.places, drawn.weights, strict=True))
        assert seen[3] < seen[0]

    def test_at_full_weighting_the_two_effects_cancel(self) -> None:
        # A step drawn nine times more often than another counts one ninth as
        # much, so how much of the batch it accounts for is unchanged.
        buffer: Replay[int] = Replay(Rng(2), 2, priority=1.0, weighting=1.0)
        buffer.add(step(0))
        buffer.add(step(1))
        buffer.reprioritise([0, 1], [1.0, 9.0])

        drawn = buffer.sample(4000)
        counted = [0.0, 0.0]
        for place, weight in zip(drawn.places, drawn.weights, strict=True):
            counted[place] += weight
        assert counted[0] / counted[1] == pytest.approx(1.0, abs=0.05)

    def test_half_weighting_corrects_half_of_it(self) -> None:
        buffer: Replay[int] = Replay(Rng(2), 2, priority=1.0, weighting=0.5)
        buffer.add(step(0))
        buffer.add(step(1))
        buffer.reprioritise([0, 1], [1.0, 9.0])

        drawn = buffer.sample(4000)
        counted = [0.0, 0.0]
        for place, weight in zip(drawn.places, drawn.weights, strict=True):
            counted[place] += weight
        # Drawn nine to one and counted one to three, so three to one.
        assert counted[1] / counted[0] == pytest.approx(3.0, abs=0.2)

    def test_no_weight_is_above_one(self) -> None:
        buffer: Replay[int] = Replay(Rng(7), 6, priority=0.8, weighting=0.7)
        for number in range(6):
            buffer.add(step(number))
        buffer.reprioritise(range(6), [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
        for weight in buffer.sample(100).weights:
            assert 0.0 < weight <= 1.0

    def test_the_least_likely_step_counts_fully(self) -> None:
        buffer: Replay[int] = Replay(Rng(7), 3, priority=1.0, weighting=1.0)
        for number in range(3):
            buffer.add(step(number))
        buffer.reprioritise([0, 1, 2], [1.0, 3.0, 7.0])

        drawn = buffer.sample(300)
        seen = dict(zip(drawn.places, drawn.weights, strict=True))
        assert seen[0] == pytest.approx(1.0)

    def test_an_empty_buffer_gives_no_weights(self) -> None:
        assert Replay[int](Rng(1), 10, priority=1.0).sample(4).weights == ()
