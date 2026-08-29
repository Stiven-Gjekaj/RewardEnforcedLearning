"""Tests for the buffer of remembered steps."""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.replay import Replay
from rel.rng import Rng


def step(number: int) -> Transition[int]:
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
        drawn = {one.observation for _ in range(50) for one in buffer.sample(4)}
        assert drawn == {0, 1, 2, 3, 4}

    def test_a_short_buffer_gives_what_it_has(self) -> None:
        # The first few steps of a run are exactly when a buffer is short, and
        # an agent that could not learn until it was full would do nothing for
        # the first thousand steps.
        buffer: Replay[int] = Replay(Rng(1), 100)
        buffer.add(step(1))
        drawn = buffer.sample(8)
        assert len(drawn) == 8
        assert {one.observation for one in drawn} == {1}

    def test_an_empty_buffer_gives_nothing(self) -> None:
        assert Replay[int](Rng(1), 10).sample(4) == []

    def test_a_sample_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            Replay[int](Rng(1), 10).sample(0)

    def test_the_same_seed_draws_the_same_sample(self) -> None:
        def drawn(seed: int) -> list[int]:
            buffer: Replay[int] = Replay(Rng(seed), 10)
            for number in range(10):
                buffer.add(step(number))
            return [one.observation for one in buffer.sample(6)]

        assert drawn(4) == drawn(4)
        assert drawn(4) != drawn(5)
