"""Tests for the one environment here whose values are known in closed form.

Every other measurement in this project compares an agent against dynamic
programming over the same model. That is a strong check and it is still a check
against another computation. This one is a check against arithmetic, and the
test that matters most holds the two against each other.
"""

from __future__ import annotations

import pytest

from rel.envs.classic import RandomWalk, random_walk
from rel.rng import Rng


def a_walk(size: int = 5) -> RandomWalk:
    return random_walk(Rng(1), size=size)


class TestTheShape:
    def test_there_is_an_ending_at_each_end(self) -> None:
        walk = a_walk()
        assert walk.is_ending(0)
        assert walk.is_ending(6)
        assert not any(walk.is_ending(state) for state in range(1, 6))

    def test_the_model_agrees_about_which_states_end(self) -> None:
        assert a_walk().terminal_states() == frozenset({0, 6})

    def test_it_starts_in_the_middle(self) -> None:
        assert a_walk().reset() == 3

    def test_an_even_size_starts_left_of_centre(self) -> None:
        # There is no middle of an even walk. The docstring says which way it
        # goes rather than leaving a reader to find out.
        assert a_walk(size=4).reset() == 2

    def test_there_are_two_actions(self) -> None:
        walk = a_walk()
        assert walk.action_space.n == 2
        assert walk.action_names == ("left", "right")

    def test_a_walk_with_no_cells_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one cell"):
            random_walk(Rng(1), size=0)


class TestMoving:
    def test_left_goes_left_and_right_goes_right(self) -> None:
        walk = a_walk()
        walk.reset()
        assert walk.step(RandomWalk.LEFT).observation == 2
        assert walk.step(RandomWalk.RIGHT).observation == 3

    def test_reaching_the_right_ending_pays_one(self) -> None:
        walk = a_walk()
        walk.reset()
        for _ in range(2):
            walk.step(RandomWalk.RIGHT)
        step = walk.step(RandomWalk.RIGHT)
        assert step.reward == 1.0
        assert step.terminated

    def test_reaching_the_left_ending_pays_nothing(self) -> None:
        walk = a_walk()
        walk.reset()
        for _ in range(2):
            walk.step(RandomWalk.LEFT)
        step = walk.step(RandomWalk.LEFT)
        assert step.reward == 0.0
        assert step.terminated

    def test_a_step_in_the_middle_pays_nothing(self) -> None:
        walk = a_walk()
        walk.reset()
        assert walk.step(RandomWalk.LEFT).reward == 0.0


class TestTheTrueValues:
    def test_they_are_the_share_of_the_way_along(self) -> None:
        assert a_walk().true_values() == pytest.approx(
            (0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 0.0)
        )

    def test_the_endings_are_worth_nothing(self) -> None:
        values = a_walk(size=9).true_values()
        assert values[0] == 0.0
        assert values[-1] == 0.0

    def test_they_agree_with_a_sweep_over_the_model(self) -> None:
        # The check that matters. The closed form and dynamic programming are
        # two independent routes to the same numbers, and a fault in either
        # one shows up as a disagreement rather than as a plausible answer.
        walk = a_walk(size=9)
        values = [0.0] * walk.observation_space.n

        for _ in range(5000):
            fresh = list(values)
            for state in range(walk.observation_space.n):
                if walk.is_ending(state):
                    continue
                total = 0.0
                for action in walk.action_space:
                    for outcome in walk.transitions(state, action):
                        rest = (
                            0.0 if outcome.terminated else values[outcome.observation]
                        )
                        total += outcome.probability * (outcome.reward + rest)
                fresh[state] = total / walk.action_space.n
            values = fresh

        assert tuple(values) == pytest.approx(walk.true_values(), abs=1e-9)


class TestDrawingIt:
    def test_the_endings_are_walls_and_the_agent_is_a_circle(self) -> None:
        walk = a_walk()
        walk.reset()
        assert walk.render() == "|--o--|"

    def test_it_moves(self) -> None:
        walk = a_walk()
        walk.reset()
        walk.step(RandomWalk.LEFT)
        assert walk.render() == "|-o---|"
