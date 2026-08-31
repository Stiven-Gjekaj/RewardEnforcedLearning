"""Tests for the one environment here whose values are known in closed form.

Every other measurement in this project compares an agent against dynamic
programming over the same model. That is a strong check and it is still a check
against another computation. This one is a check against arithmetic, and the
test that matters most holds the two against each other.
"""

from __future__ import annotations

import pytest

from rel.agents.dp import (
    evaluate_policy,
    evaluate_shares,
    uniform_shares,
    value_iteration,
)
from rel.envs.classic import RandomWalk, long_walk, random_walk
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


class TestAStepLongerThanOneCell:
    """Sutton and Barto, example 9.1, which is the same walk with a stride.

    The action still picks the direction and the environment picks the
    distance. What is worth holding is that the model and the stepping agree
    about a step that would pass an ending, because that is the one place the
    two could differ and the closed form does not reach it.
    """

    def test_a_stride_of_one_is_the_walk_it_always_was(self) -> None:
        plain = a_walk(size=9)
        strided = RandomWalk(Rng(1).stream("env"), size=9, stride=1)
        for state in range(11):
            for action in (0, 1):
                assert plain.transitions(state, action) == strided.transitions(
                    state, action
                )

    def test_a_step_is_evenly_between_one_cell_and_the_stride(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=99, stride=4)
        landed = walk.transitions(50, RandomWalk.RIGHT)
        assert {outcome.observation for outcome in landed} == {51, 52, 53, 54}
        for outcome in landed:
            assert outcome.probability == pytest.approx(0.25)

    def test_a_step_that_would_pass_an_ending_stops_at_it(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=9, stride=4)
        landed = walk.transitions(8, RandomWalk.RIGHT)
        # From cell 8 of nine, three of the four step sizes reach the ending.
        ending = [outcome for outcome in landed if outcome.observation == 10]
        assert len(ending) == 1
        assert ending[0].probability == pytest.approx(0.75)
        assert ending[0].reward == 1.0
        assert ending[0].terminated

    def test_the_branches_add_up_wherever_it_stands(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=20, stride=7)
        for state in range(22):
            for action in (0, 1):
                total = sum(
                    outcome.probability for outcome in walk.transitions(state, action)
                )
                assert total == pytest.approx(1.0), (state, action)

    def test_nothing_ever_leaves_the_state_space(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=30, stride=11)
        policy = Rng(3).stream("policy")
        observation = walk.reset()
        for _ in range(3000):
            assert walk.observation_space.contains(observation)
            outcome = walk.step(policy.below(2))
            observation = outcome.observation
            if outcome.done:
                observation = walk.reset()

    def test_the_closed_form_refuses_a_longer_step(self) -> None:
        # It is right for a step of one cell and a little wrong for any other,
        # and a little wrong is the worst thing a reference value can be.
        with pytest.raises(ValueError, match="closed form is for a stride of one"):
            RandomWalk(Rng(1).stream("env"), size=9, stride=2).true_values()

    def test_a_stride_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one cell"):
            RandomWalk(Rng(1).stream("env"), size=9, stride=0)

    def test_the_long_walk_is_the_one_the_chapter_uses(self) -> None:
        walk = long_walk(Rng(1).stream("env"))
        assert walk.size == 1000
        assert walk.stride == 100
        assert walk.observation_space.n == 1002

    def test_the_long_walk_reaches_an_ending_inside_its_limit(self) -> None:
        """Which is the whole reason the stride exists.

        A thousand cells taking one cell a step needs about a quarter of a
        million steps to reach an ending. Every episode of that truncates and
        a run of it measures the step limit.
        """
        walk = long_walk(Rng(4).stream("env"))
        policy = Rng(4).stream("policy")
        lengths = []
        for _ in range(20):
            walk.reset()
            length = 0
            while True:
                length += 1
                if walk.step(policy.below(2)).done:
                    break
            lengths.append(length)
        assert max(lengths) < walk.spec.max_episode_steps
        assert sum(lengths) / len(lengths) < 1000


class TestTheWalkOfOneCell:
    """The smallest walk there is, and the one the default model reads wrongly.

    `TabularEnv.terminal_states` calls a state terminal when every branch out
    of it says so. A walk of one cell has one cell and both of its branches
    reach an ending, so the default called the cell an ending too. Dynamic
    programming then never swept it and reported it worth nothing, where a
    policy that always goes right from it is worth one.
    """

    def test_only_the_two_endings_are_endings(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=1)
        assert walk.terminal_states() == frozenset({0, 2})

    def test_the_cell_is_worth_a_half_under_the_uniform_policy(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=1)
        assert walk.true_values() == pytest.approx((0.0, 0.5, 0.0))
        assert evaluate_shares(walk, uniform_shares(walk)) == pytest.approx(
            (0.0, 0.5, 0.0), abs=1e-6
        )

    def test_always_going_right_from_it_is_worth_one(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=1)
        assert evaluate_policy(walk, [1, 1, 1]).values[1] == pytest.approx(1.0)

    def test_the_best_possible_policy_reaches_the_right_ending(self) -> None:
        walk = RandomWalk(Rng(1).stream("env"), size=1)
        assert value_iteration(walk).start_value == pytest.approx(1.0)

    def test_every_size_reads_only_its_endings(self) -> None:
        for size in (1, 2, 3, 9):
            walk = RandomWalk(Rng(1).stream("env"), size=size)
            assert walk.terminal_states() == frozenset({0, size + 1}), size
