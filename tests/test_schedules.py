"""Tests for how a number changes as a run goes on."""

from __future__ import annotations

import pytest

from rel.schedules import as_schedule, constant, exponential, linear


class TestConstant:
    def test_it_gives_the_same_number_for_ever(self) -> None:
        held = constant(0.3)
        assert [held(0), held(50), held(100_000)] == [0.3, 0.3, 0.3]


class TestLinear:
    def test_it_starts_and_ends_where_it_says(self) -> None:
        falling = linear(1.0, 0.1, over=100)
        assert falling(0) == 1.0
        assert falling(100) == 0.1

    def test_it_is_halfway_at_halfway(self) -> None:
        assert linear(1.0, 0.0, over=100)(50) == pytest.approx(0.5)

    def test_it_stays_at_the_end_afterwards(self) -> None:
        # A schedule that carried on past its end would go negative, and an
        # exploration rate below zero is not a smaller exploration rate.
        falling = linear(1.0, 0.1, over=100)
        assert falling(500) == 0.1

    def test_it_can_rise(self) -> None:
        rising = linear(0.0, 1.0, over=10)
        assert rising(5) == pytest.approx(0.5)

    def test_a_length_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above zero"):
            linear(1.0, 0.0, over=0)


class TestExponential:
    def test_it_halves_over_the_half_life(self) -> None:
        falling = exponential(1.0, half_life=50)
        assert falling(0) == 1.0
        assert falling(50) == pytest.approx(0.5)
        assert falling(100) == pytest.approx(0.25)

    def test_it_never_goes_below_the_floor(self) -> None:
        falling = exponential(1.0, half_life=10, floor=0.05)
        assert falling(10_000) == 0.05

    def test_a_half_life_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above zero"):
            exponential(1.0, half_life=0)


class TestAsSchedule:
    def test_a_number_becomes_a_schedule(self) -> None:
        # Every agent runs its arguments through this, so that a caller writes
        # `epsilon=0.1` for the common case and hands a schedule in for the
        # rest.
        held = as_schedule(0.25)
        assert held(0) == 0.25
        assert held(999) == 0.25

    def test_a_whole_number_becomes_a_float(self) -> None:
        assert isinstance(as_schedule(1)(0), float)

    def test_a_schedule_is_left_alone(self) -> None:
        falling = linear(1.0, 0.0, over=10)
        assert as_schedule(falling) is falling


def test_a_schedule_reaches_an_agent() -> None:
    """The whole point: an agent takes one and uses it.

    A schedule that an agent stored and never called would look right in every
    test above and do nothing at all in a run.
    """
    from rel.agents.td import QLearning
    from rel.rng import Rng
    from rel.spaces import Discrete

    agent = QLearning(Rng(1), Discrete(2), epsilon=linear(1.0, 0.0, over=10))
    assert agent.current_epsilon() == 1.0

    for _ in range(10):
        agent.end_episode()
    assert agent.current_epsilon() == 0.0
