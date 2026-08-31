"""Tests for cutting a box of actions into a short list of them.

Two things are worth holding. The cut has to reach both ends of the box, since
an environment driven by a motor needs full power to be reachable at all, and
everything but the action has to come from the environment inside, so that a
run of the cut version is a run of the same problem.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from rel.envs import ENVIRONMENTS
from rel.envs.levels import Levels, levelled_pendulum
from rel.envs.pendulum import Pendulum
from rel.rng import Rng
from rel.spaces import Box, Discrete


def build(levels: int = 9, seed: int = 1) -> Levels[tuple[float, ...]]:
    return levelled_pendulum(Rng(seed).stream("env"), levels=levels)


class TestTheCut:
    def test_it_is_in_the_registry(self) -> None:
        env = ENVIRONMENTS.make("pendulum-levels", Rng(1).stream("env"))
        assert isinstance(env, Levels)

    def test_the_action_space_counts_the_levels(self) -> None:
        assert build(levels=5).action_space == Discrete(5)

    def test_both_ends_of_the_box_are_reachable(self) -> None:
        # A motor that cannot be given full power is a different problem.
        for levels in (2, 3, 4, 9):
            torques = [torque[0] for torque in build(levels=levels).every_torque()]
            assert min(torques) == pytest.approx(-2.0)
            assert max(torques) == pytest.approx(2.0)

    def test_two_levels_are_the_two_ends_and_nothing_between(self) -> None:
        assert build(levels=2).every_torque() == [(-2.0,), (2.0,)]

    def test_three_levels_hold_the_middle(self) -> None:
        assert build(levels=3).every_torque() == [(-2.0,), (0.0,), (2.0,)]

    def test_the_gaps_are_even(self) -> None:
        torques = [torque[0] for torque in build(levels=9).every_torque()]
        gaps = [second - first for first, second in pairwise(torques)]
        for gap in gaps:
            assert gap == pytest.approx(gaps[0])

    def test_one_level_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least the two ends"):
            build(levels=1)

    def test_an_environment_with_nothing_to_cut_is_refused(self) -> None:
        with pytest.raises(TypeError, match="nothing to cut"):
            Levels(ENVIRONMENTS.make("cliff", Rng(1).stream("env")))  # type: ignore[arg-type]

    def test_an_action_outside_the_list_is_refused(self) -> None:
        with pytest.raises(IndexError, match="is not one of"):
            build(levels=3).torque(3)


class TestMoreThanOneDimension:
    """The cut is a number read in base `levels`, one digit per dimension.

    Nothing in this project has a two dimensional box of actions yet, and the
    arithmetic that would break on one is here rather than the day one arrives.
    """

    class TwoMotors(Pendulum):
        def __init__(self, rng: Rng) -> None:
            super().__init__(rng)
            self.action_space = Box([-2.0, 0.0], [2.0, 1.0], names=["turn", "lift"])

    def _cut(self, levels: int) -> Levels[tuple[float, ...]]:
        return Levels(self.TwoMotors(Rng(1).stream("env")), levels=levels)

    def test_the_count_is_the_levels_to_the_power_of_the_dimensions(self) -> None:
        assert self._cut(3).action_space == Discrete(9)
        assert self._cut(4).action_space == Discrete(16)

    def test_the_first_dimension_moves_fastest(self) -> None:
        cut = self._cut(2)
        assert cut.every_torque() == [
            (-2.0, 0.0),
            (2.0, 0.0),
            (-2.0, 1.0),
            (2.0, 1.0),
        ]

    def test_every_corner_is_reachable(self) -> None:
        corners = set(self._cut(3).every_torque())
        assert {(-2.0, 0.0), (2.0, 0.0), (-2.0, 1.0), (2.0, 1.0)} <= corners


class TestWhatItForwards:
    def test_the_observation_space_is_the_one_inside(self) -> None:
        cut = build()
        assert cut.observation_space is cut.inside.observation_space

    def test_the_box_a_tile_coder_divides_comes_through(self) -> None:
        cut = build()
        assert cut.tiling_space is cut.inside.tiling_space

    def test_the_spec_says_the_same_things(self) -> None:
        cut = build()
        assert cut.spec.summary == cut.inside.spec.summary
        assert cut.spec.ends == cut.inside.spec.ends
        assert cut.spec.max_episode_steps == cut.inside.spec.max_episode_steps
        assert cut.spec.suggested_discount == cut.inside.spec.suggested_discount

    def test_the_name_says_it_has_been_cut(self) -> None:
        assert build().spec.name == "pendulum-levels"

    def test_the_audit_is_the_one_inside(self) -> None:
        cut = build()
        cut.reset()
        cut.step(0)
        assert cut.audit() == cut.inside.audit()

    def test_the_drawing_is_the_one_inside(self) -> None:
        cut = build()
        cut.reset()
        assert cut.render() == cut.inside.render()

    def test_it_says_what_it_wraps(self) -> None:
        assert repr(build(levels=4)).startswith("Levels(")
        assert "levels=4" in repr(build(levels=4))


class TestARunOfItIsARunOfTheSameProblem:
    def test_a_step_is_the_step_the_torque_would_have_made(self) -> None:
        cut = build(levels=3)
        plain = Pendulum(Rng(1).stream("env"))

        seen = cut.reset()
        assert plain.reset() == seen

        for action, torque in ((0, (-2.0,)), (2, (2.0,)), (1, (0.0,))):
            one = cut.step(action)
            two = plain.step(torque)
            assert one.observation == two.observation
            assert one.reward == two.reward

    def test_both_truncate_on_the_same_step(self) -> None:
        cut = levelled_pendulum(Rng(1).stream("env"), levels=3, steps=5)
        cut.reset()
        for _ in range(4):
            assert not cut.step(1).done
        assert cut.step(1).truncated

    def test_it_can_be_started_again_after_the_limit(self) -> None:
        cut = levelled_pendulum(Rng(1).stream("env"), levels=3, steps=3)
        for _ in range(2):
            cut.reset()
            for _ in range(3):
                outcome = cut.step(1)
            assert outcome.truncated

    def test_the_reward_is_never_above_zero(self) -> None:
        cut = build()
        cut.reset()
        policy = Rng(4).stream("policy")
        for _ in range(300):
            outcome = cut.step(cut.action_space.sample(policy))
            assert outcome.reward <= 0.0
            if outcome.done:
                cut.reset()

    def test_holding_the_switch_one_way_stalls(self) -> None:
        """Which is what makes this a swinging problem rather than a lifting one.

        Held at full power from the bottom, the weight climbs to 0.659 of the
        way down and stays there however long it is held: the motor is exactly
        strong enough to balance gravity at that angle and no stronger. Held at
        nothing it does not move at all, because the bottom is where the pull
        is zero.
        """
        for action in (0, 1):
            for steps in (20, 100):
                cut = levelled_pendulum(Rng(2).stream("env"), levels=2)
                cut.reset()
                cut.inside.angle = math.pi
                cut.inside.speed = 0.0
                cut.inside.highest = -1.0
                for _ in range(steps):
                    cut.step(action)
                assert cut.audit()["highest_point"] == pytest.approx(-0.659, abs=0.002)

        still = Pendulum(Rng(2).stream("env"))
        still.reset()
        still.angle = math.pi
        still.speed = 0.0
        still.highest = -1.0
        for _ in range(100):
            still.step((0.0,))
        assert still.audit()["highest_point"] == pytest.approx(-1.0)

    def test_a_switch_that_pumps_reaches_the_top(self) -> None:
        # Full power in whichever direction it is already moving, which adds
        # energy on every step. Two levels is enough for that.
        cut = levelled_pendulum(Rng(2).stream("env"), levels=2)
        cut.reset()
        cut.inside.angle = math.pi
        cut.inside.speed = 0.01
        cut.inside.highest = -1.0
        for _ in range(200):
            cut.step(1 if cut.inside.speed >= 0.0 else 0)
        assert cut.audit()["highest_point"] > 0.999
