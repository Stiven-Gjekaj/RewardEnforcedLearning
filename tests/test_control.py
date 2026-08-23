"""Tests for the two control problems.

The physics is written out here rather than called for, so these tests ask
whether it behaves like the thing it models. A cart pushed one way goes that
way. A car with an engine too weak for the hill does not climb it. Neither of
those follows from the code compiling.
"""

from __future__ import annotations

import math

from rel.envs.control import CartPole, MountainCar
from rel.rng import Rng

LEFT, RIGHT = 0, 1
BACK, COAST, FORWARD = 0, 1, 2


class TestCartPoleSetup:
    def test_it_starts_near_the_middle_and_near_upright(self) -> None:
        pole = CartPole(Rng(1), start_spread=0.05)
        for _ in range(50):
            observation = pole.reset()
            assert all(abs(value) <= 0.05 for value in observation)

    def test_the_observation_is_the_four_numbers(self) -> None:
        observation = CartPole(Rng(1)).reset()
        assert len(observation) == 4
        assert all(isinstance(value, float) for value in observation)

    def test_every_step_pays_one(self) -> None:
        # The pole is put back upright between steps. Left alone it falls in
        # about ten steps under a constant push, and then the reward stops
        # because the episode does.
        pole = CartPole(Rng(1))
        pole.reset()
        for _ in range(20):
            pole.angle = pole.spin = 0.0
            assert pole.step(LEFT).reward == 1.0


class TestCartPolePhysics:
    def test_pushing_right_moves_the_cart_right(self) -> None:
        # The pole is held upright between steps so that only the cart is
        # under test. Left to itself it falls before ten steps are up.
        pole = CartPole(Rng(2))
        pole.reset()
        pole.position = pole.speed = 0.0
        for _ in range(10):
            pole.angle = pole.spin = 0.0
            pole.step(RIGHT)
        assert pole.position > 0.0

    def test_pushing_left_moves_the_cart_left(self) -> None:
        pole = CartPole(Rng(2))
        pole.reset()
        pole.position = pole.speed = 0.0
        for _ in range(10):
            pole.angle = pole.spin = 0.0
            pole.step(LEFT)
        assert pole.position < 0.0

    def test_a_leaning_pole_falls_further(self) -> None:
        # The cart is pushed left while the pole leans right. Getting under a
        # pole that leans right means moving right, so this is the wrong way,
        # and the lean has to grow.
        pole = CartPole(Rng(3))
        pole.reset()
        pole.position = pole.speed = pole.spin = 0.0
        pole.angle = 0.02
        for _ in range(5):
            pole.step(LEFT)
        assert pole.angle > 0.02

    def test_gravity_pulls_the_pole_away_from_upright(self) -> None:
        # Gravity has to pull the pole away from upright, not towards it. A
        # sign error there gives an environment that balances itself, and every
        # agent then looks like it solved the problem on the first episode.
        #
        # Watching a leaning pole fall does not show this. Inside the twelve
        # degrees the environment allows, the term from the cart is five times
        # the term from gravity, so the pole falls the same way with the sign
        # either way round. What separates them is how much faster a bigger
        # lean falls. Right gravity: 17.4 against 14.6 radians per second
        # squared. Wrong gravity: 11.2 against 14.6, which is slower.
        def spin_after_one_step(angle: float) -> float:
            pole = CartPole(Rng(3))
            pole.reset()
            pole.position = pole.speed = pole.spin = 0.0
            pole.angle = angle
            pole.step(LEFT)
            return pole.spin

        assert spin_after_one_step(0.2) > spin_after_one_step(0.0)

    def test_the_position_moves_with_the_speed_it_ends_up_with(self) -> None:
        # This pins the integrator. Semi-implicit Euler advances the speed and
        # then moves the position by the speed it now has. Plain Euler moves
        # the position by the speed it had before, which from a standing start
        # means the cart does not move at all on the first step.
        #
        # The difference is not a rounding error. Plain Euler feeds energy into
        # a swinging system, so a pole that is only just balanced drifts
        # outward on its own, and an agent gets credit or blame for it.
        pole = CartPole(Rng(3))
        pole.reset()
        pole.position = pole.speed = pole.angle = pole.spin = 0.0
        pole.step(RIGHT)

        assert pole.speed > 0.0
        assert abs(pole.position - CartPole.TIMESTEP * pole.speed) < 1e-15
        assert abs(pole.angle - CartPole.TIMESTEP * pole.spin) < 1e-15

    def test_pushing_under_a_leaning_pole_slows_the_fall(self) -> None:
        # The other half of the sign. Pushing the cart towards the lean has to
        # turn the pole back, or nothing an agent does can help it.
        def lean_after(action: int) -> float:
            pole = CartPole(Rng(3))
            pole.reset()
            pole.position = pole.speed = pole.spin = 0.0
            pole.angle = 0.02
            for _ in range(5):
                pole.step(action)
            return pole.angle

        assert lean_after(RIGHT) < lean_after(LEFT)

    def test_the_pole_falls_when_nothing_corrects_it(self) -> None:
        pole = CartPole(Rng(4))
        pole.reset()
        pole.angle = 0.05
        pole.spin = 0.0
        steps = 0
        while True:
            outcome = pole.step(RIGHT)
            steps += 1
            if outcome.done:
                break
        assert outcome.terminated
        assert steps < 100


class TestCartPoleEndings:
    def test_it_ends_when_the_pole_passes_twelve_degrees(self) -> None:
        pole = CartPole(Rng(5))
        pole.reset()
        pole.angle = CartPole.ANGLE_LIMIT - 0.001
        pole.spin = 1.0
        outcome = pole.step(LEFT)
        assert outcome.terminated
        assert abs(pole.angle) > CartPole.ANGLE_LIMIT

    def test_it_ends_when_the_cart_leaves_the_track(self) -> None:
        pole = CartPole(Rng(6))
        pole.reset()
        pole.position = CartPole.TRACK_LIMIT - 0.01
        pole.speed = 2.0
        pole.angle = 0.0
        pole.spin = 0.0
        outcome = pole.step(RIGHT)
        assert outcome.terminated
        assert abs(pole.position) > CartPole.TRACK_LIMIT

    def test_the_step_limit_truncates_rather_than_terminates(self) -> None:
        pole = CartPole(Rng(7), max_episode_steps=10)
        pole.reset()
        pole.angle = pole.spin = pole.position = pole.speed = 0.0
        for _ in range(9):
            pole.angle = pole.spin = 0.0  # hold it upright by hand
            assert not pole.step(LEFT).done
        pole.angle = pole.spin = 0.0
        last = pole.step(LEFT)
        assert last.truncated
        assert not last.terminated


class TestCartPoleAudit:
    def test_a_pole_held_upright_reads_as_steady(self) -> None:
        pole = CartPole(Rng(8))
        pole.reset()
        for _ in range(20):
            pole.position = pole.speed = pole.angle = pole.spin = 0.0
            pole.step(LEFT)
        assert pole.audit()["steady_share"] > 0.9

    def test_a_pole_wobbling_near_the_limit_does_not(self) -> None:
        # This is the gap the audit exists for. Both of these episodes pay one
        # reward per step, and only one of them is a pole that anybody would
        # call balanced.
        pole = CartPole(Rng(9))
        pole.reset()
        for _ in range(20):
            pole.angle = CartPole.ANGLE_LIMIT * 0.9
            pole.spin = 0.0
            pole.position = 0.0
            pole.speed = 0.0
            pole.step(LEFT)
        assert pole.audit()["steady_share"] == 0.0

    def test_it_reports_zero_before_the_first_step(self) -> None:
        pole = CartPole(Rng(10))
        pole.reset()
        assert pole.audit() == {"steady_share": 0.0}


def test_every_cart_pole_observation_is_inside_the_box() -> None:
    # The box says what the environment can produce, and a tile coder and a
    # report both read it. A bound that the environment passes is a bound that
    # silently clips a real state.
    pole = CartPole(Rng(11).stream("env"))
    policy = Rng(11).stream("policy")
    for _ in range(400):
        observation = pole.reset()
        assert pole.observation_space.contains(observation)
        while True:
            outcome = pole.step(policy.below(2))
            assert pole.observation_space.contains(outcome.observation), (
                f"{outcome.observation} is outside {pole.observation_space}"
            )
            if outcome.done:
                break


class TestMountainCarSetup:
    def test_it_starts_at_rest_in_the_valley(self) -> None:
        car = MountainCar(Rng(12))
        for _ in range(50):
            position, speed = car.reset()
            assert -0.6 <= position < -0.4
            assert speed == 0.0

    def test_every_step_pays_minus_one(self) -> None:
        car = MountainCar(Rng(13))
        car.reset()
        assert all(car.step(COAST).reward == -1.0 for _ in range(20))


class TestMountainCarPhysics:
    def test_the_engine_alone_cannot_climb_the_hill(self) -> None:
        # This is the whole problem. An environment where full throttle wins is
        # not the mountain car, and an agent tested on it has been told nothing.
        car = MountainCar(Rng(14).stream("env"), max_episode_steps=5000)
        car.reset()
        while True:
            outcome = car.step(FORWARD)
            if outcome.done:
                break
        assert not outcome.terminated
        assert car.audit()["arrived"] == 0.0
        assert car.audit()["peak_position"] < MountainCar.GOAL

    def test_rocking_with_the_speed_climbs_it(self) -> None:
        # Push whichever way the car is already moving. This is the textbook
        # solution, and it has to work or the environment is unsolvable.
        car = MountainCar(Rng(15).stream("env"), max_episode_steps=5000)
        observation = car.reset()
        steps = 0
        while True:
            action = FORWARD if observation[1] >= 0.0 else BACK
            outcome = car.step(action)
            observation = outcome.observation
            steps += 1
            if outcome.done:
                break
        assert outcome.terminated
        assert steps < 500

    def test_the_speed_is_capped(self) -> None:
        # Driven straight at the cap, from a stretch of hill steep enough to
        # push it past. Watching a whole run instead proves nothing: the
        # textbook policy reaches the goal in about 120 steps and never gets
        # near the cap, so the run passes with the cap taken out.
        car = MountainCar(Rng(16))
        car.reset()
        car.position = -1.0
        car.speed = MountainCar.TOP_SPEED - 0.0001
        _, speed = car.step(FORWARD).observation
        assert speed == MountainCar.TOP_SPEED

        car.position = -1.0
        car.speed = -MountainCar.TOP_SPEED + 0.0001
        car.step(BACK)
        assert car.speed >= -MountainCar.TOP_SPEED

    def test_the_speed_stays_capped_through_a_whole_run(self) -> None:
        car = MountainCar(Rng(16).stream("env"), max_episode_steps=5000)
        observation = car.reset()
        for _ in range(3000):
            action = FORWARD if observation[1] >= 0.0 else BACK
            outcome = car.step(action)
            observation = outcome.observation
            assert abs(observation[1]) <= MountainCar.TOP_SPEED + 1e-12
            if outcome.done:
                observation = car.reset()

    def test_the_far_wall_takes_the_speed_away(self) -> None:
        # Without this the car bounces off the closed end and keeps its energy,
        # which makes the hill easy to climb and the problem a different one.
        car = MountainCar(Rng(17))
        car.reset()
        car.position = MountainCar.LOWEST + 0.001
        car.speed = -0.05
        car.step(BACK)
        assert car.position == MountainCar.LOWEST
        assert car.speed == 0.0

    def test_reaching_the_goal_ends_the_episode(self) -> None:
        car = MountainCar(Rng(18))
        car.reset()
        car.position = MountainCar.GOAL - 0.001
        car.speed = MountainCar.TOP_SPEED
        outcome = car.step(FORWARD)
        assert outcome.terminated
        assert car.audit()["arrived"] == 1.0


class TestMountainCarAudit:
    def test_it_reports_how_far_up_the_car_got(self) -> None:
        # The return of a truncated episode is the step limit whatever the car
        # did, so the reward cannot separate a car that nearly arrived from one
        # that never left the bottom.
        car = MountainCar(Rng(19))
        car.reset()
        assert car.audit()["peak_position"] < -0.3

        car.position = 0.2
        car.speed = MountainCar.TOP_SPEED
        car.step(FORWARD)
        assert car.audit()["peak_position"] > 0.2

    def test_the_peak_does_not_go_down_again(self) -> None:
        car = MountainCar(Rng(20))
        car.reset()
        car.position = 0.2
        car.speed = MountainCar.TOP_SPEED
        car.step(FORWARD)
        high = car.audit()["peak_position"]

        car.speed = -MountainCar.TOP_SPEED
        for _ in range(20):
            car.step(BACK)
        assert car.position < high
        assert car.audit()["peak_position"] == high


def test_every_mountain_car_observation_is_inside_the_box() -> None:
    car = MountainCar(Rng(21).stream("env"))
    policy = Rng(21).stream("policy")
    for _ in range(100):
        observation = car.reset()
        assert car.observation_space.contains(observation)
        while True:
            outcome = car.step(policy.below(3))
            assert car.observation_space.contains(outcome.observation)
            if outcome.done:
                break


def test_the_same_seed_gives_the_same_flight() -> None:
    def flight(seed: int) -> list[float]:
        pole = CartPole(Rng(seed).stream("env"))
        policy = Rng(seed).stream("policy")
        pole.reset()
        angles = []
        for _ in range(200):
            outcome = pole.step(policy.below(2))
            angles.append(outcome.observation[2])
            if outcome.done:
                pole.reset()
        return angles

    assert flight(22) == flight(22)
    assert flight(22) != flight(23)


def test_the_tiling_box_sits_inside_the_observation_box() -> None:
    # The tile coder spreads its tiles over the tiling box and clips anything
    # outside. A tiling box wider than the observation box would waste tiles on
    # states that cannot happen.
    for env in (CartPole(Rng(24)), MountainCar(Rng(24))):
        assert all(
            wide <= narrow
            for wide, narrow in zip(
                env.observation_space.low, env.tiling_space.low, strict=True
            )
        )
        assert all(
            narrow <= wide
            for narrow, wide in zip(
                env.tiling_space.high, env.observation_space.high, strict=True
            )
        )


def test_the_angle_limit_is_twelve_degrees() -> None:
    assert abs(math.degrees(CartPole.ANGLE_LIMIT) - 12.0) < 1e-9
