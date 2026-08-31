"""Tests for the one environment whose action is a number rather than a choice.

The physics tests are the load bearing ones. Every other environment here is
checked against a table or a closed form, and this one has neither, so what
holds it is a handful of positions whose behaviour is arithmetic: a weight
balanced exactly at the top stays there, a weight let go from the side falls
the way gravity points, and the cost is zero at exactly one place.
"""

from __future__ import annotations

import math

import pytest

from rel.envs import ENVIRONMENTS
from rel.envs.pendulum import Pendulum, wrapped
from rel.rng import Rng
from rel.spaces import Box

STILL: tuple[float, ...] = (0.0,)


def build(steps: int = 200, seed: int = 1) -> Pendulum:
    return Pendulum(Rng(seed).stream("env"), max_episode_steps=steps)


def placed(angle: float, speed: float = 0.0, seed: int = 1) -> Pendulum:
    """A pendulum reset and then moved to a place worth asking about."""
    env = build(seed=seed)
    env.reset()
    env.angle = angle
    env.speed = speed
    # The audit remembers the highest point since the reset, so putting the
    # weight somewhere means putting that back as well. Without it every
    # reading is about wherever the reset happened to land.
    env.highest = math.cos(angle)
    env.torque_spent = 0.0
    return env


class TestTheAngleThatWraps:
    def test_a_small_angle_is_left_alone(self) -> None:
        assert wrapped(0.5) == pytest.approx(0.5)
        assert wrapped(-0.5) == pytest.approx(-0.5)

    def test_a_whole_turn_is_nowhere(self) -> None:
        assert wrapped(2.0 * math.pi) == pytest.approx(0.0)
        assert wrapped(-2.0 * math.pi) == pytest.approx(0.0)

    def test_two_turns_and_a_bit_is_the_bit(self) -> None:
        assert wrapped(4.0 * math.pi + 0.3) == pytest.approx(0.3)

    def test_the_answer_is_always_inside_half_a_turn(self) -> None:
        for step in range(-100, 101):
            assert -math.pi <= wrapped(step * 0.37) <= math.pi


class TestTheShape:
    def test_it_is_in_the_registry(self) -> None:
        env = ENVIRONMENTS.make("pendulum", Rng(1).stream("env"))
        assert isinstance(env, Pendulum)

    def test_the_action_is_a_box_of_one_dimension(self) -> None:
        space = build().action_space
        assert isinstance(space, Box)
        assert space.dimensions == 1
        assert space.low == (-2.0,)
        assert space.high == (2.0,)

    def test_the_observation_is_the_angle_twice_over_and_the_speed(self) -> None:
        space = build().observation_space
        assert isinstance(space, Box)
        assert space.dimensions == 3
        assert space.low == (-1.0, -1.0, -8.0)

    def test_a_torque_outside_the_box_is_refused(self) -> None:
        env = build()
        env.reset()
        with pytest.raises(ValueError, match="is not an action"):
            env.step((5.0,))

    def test_it_never_ends(self) -> None:
        assert not build().spec.ends

    def test_every_episode_runs_to_the_limit(self) -> None:
        env = build(steps=7)
        env.reset()
        for _ in range(6):
            assert not env.step(STILL).done
        last = env.step(STILL)
        assert last.truncated
        assert not last.terminated


class TestThePhysics:
    def test_balanced_at_the_top_it_stays(self) -> None:
        # The one place where the pull is exactly zero, so nothing moves and
        # every step costs nothing at all.
        env = placed(0.0)
        for _ in range(50):
            outcome = env.step(STILL)
            assert outcome.reward == 0.0
        assert env.angle == 0.0
        assert env.speed == 0.0

    def test_let_go_from_the_side_it_falls(self) -> None:
        env = placed(math.pi / 2.0)
        env.step(STILL)
        # The angle counts from the top, so falling is the angle growing.
        assert env.speed > 0.0
        assert env.angle > math.pi / 2.0

    def test_it_falls_the_other_way_from_the_other_side(self) -> None:
        env = placed(-math.pi / 2.0)
        env.step(STILL)
        assert env.speed < 0.0
        assert env.angle < -math.pi / 2.0

    def test_the_motor_cannot_hold_it_up_from_the_side(self) -> None:
        """Which is the whole reason the problem needs a swing.

        Full torque against gravity at horizontal, and it still falls.
        """
        env = placed(math.pi / 2.0)
        env.step((-2.0,))
        assert env.speed > 0.0

    def test_the_speed_is_capped(self) -> None:
        env = placed(math.pi / 2.0, speed=8.0)
        for _ in range(20):
            env.step((2.0,))
            assert abs(env.speed) <= env.TOP_SPEED

    def test_the_angle_is_advanced_with_the_new_speed(self) -> None:
        # Semi-implicit Euler, as in `rel.envs.control`. With the old speed the
        # first step from rest would not move at all.
        env = placed(math.pi / 2.0)
        env.step(STILL)
        assert env.angle != pytest.approx(math.pi / 2.0)


class TestTheCost:
    def test_nothing_ever_pays(self) -> None:
        env = build()
        env.reset()
        policy = Rng(2).stream("policy")
        for _ in range(400):
            outcome = env.step(env.action_space.sample(policy))
            assert outcome.reward <= 0.0
            if outcome.done:
                env.reset()

    def test_the_only_free_step_is_the_top_at_rest_with_no_torque(self) -> None:
        assert placed(0.0).step(STILL).reward == 0.0
        assert placed(0.0).step((0.1,)).reward < 0.0
        assert placed(0.0, speed=0.1).step(STILL).reward < 0.0
        assert placed(0.1).step(STILL).reward < 0.0

    def test_a_whole_turn_costs_what_no_turn_costs(self) -> None:
        # The cost reads the wrapped angle, so a weight that has gone round is
        # charged for where it is rather than for how it got there.
        one = placed(0.3).step(STILL).reward
        round_again = placed(0.3 + 2.0 * math.pi).step(STILL).reward
        assert one == pytest.approx(round_again)

    def test_doing_nothing_beats_thrashing(self) -> None:
        """Both are what a run looks like when nothing has been learned.

        The numbers in the module docstring, over the same twenty starts.
        """
        still, random = 0.0, 0.0
        for seed in range(1, 21):
            quiet = Pendulum(Rng(seed).stream("env"))
            quiet.reset()
            noisy = Pendulum(Rng(seed).stream("env"))
            noisy.reset()
            policy = Rng(seed).stream("policy")
            for _ in range(200):
                still += quiet.step(STILL).reward
                random += noisy.step(noisy.action_space.sample(policy)).reward

        assert still / 20.0 == pytest.approx(-1187.0, abs=1.0)
        assert random / 20.0 < still / 20.0


class TestTheAudit:
    def test_the_highest_point_is_the_cosine_it_reached(self) -> None:
        env = placed(math.pi)
        for _ in range(3):
            env.step(STILL)
        assert env.audit()["highest_point"] == pytest.approx(
            math.cos(math.pi), abs=0.01
        )

    def test_the_top_reads_one(self) -> None:
        assert placed(0.0).audit()["highest_point"] == pytest.approx(1.0)

    def test_the_torque_spent_ignores_which_way(self) -> None:
        env = build()
        env.reset()
        env.step((1.5,))
        env.step((-0.5,))
        assert env.audit()["torque_spent"] == pytest.approx(2.0)

    def test_a_reset_forgets_the_last_episode(self) -> None:
        env = build()
        env.reset()
        env.step((2.0,))
        env.reset()
        assert env.audit()["torque_spent"] == 0.0


class TestTheDrawing:
    def test_the_weight_is_drawn_at_the_top_when_it_is_up(self) -> None:
        drawn = placed(0.0).render().splitlines()
        assert "o" in drawn[0]
        assert "+" in drawn[len(drawn) // 2]

    def test_the_weight_is_drawn_at_the_bottom_when_it_hangs(self) -> None:
        drawn = placed(math.pi).render().splitlines()
        assert "o" in drawn[-1]

    def test_it_is_drawn_to_one_side_and_then_the_other(self) -> None:
        left = placed(-math.pi / 2.0).render().splitlines()
        right = placed(math.pi / 2.0).render().splitlines()
        assert left[5].index("o") < right[5].index("o")
