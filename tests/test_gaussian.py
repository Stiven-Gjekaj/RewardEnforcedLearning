"""Tests for the one agent whose action is a number rather than a choice.

`TestItFindsANumberItCanBeToldExactly` is the load bearing one. The pendulum
has no known answer, so a run of it says nothing about whether the arithmetic
is right, and `Aim` below is a problem whose answer is a number written into
the environment. An agent that cannot find that number cannot be trusted on
anything harder.
"""

from __future__ import annotations

import math

import pytest

from rel.agents import AGENTS
from rel.agents.base import Transition
from rel.agents.features import encoder_for
from rel.agents.gaussian import GaussianActorCritic, standardised
from rel.core import Env, EnvSpec, Step
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.spaces import Box
from rel.training import train

Point = tuple[float, ...]


class Aim(Env[Point, Point]):
    """One step, and the reward is minus how far the action missed a target.

    The whole environment is a number to hit. There is nothing to explore, no
    credit to assign over time and no state to tell apart, so an agent that
    does not end up aiming at the target has something wrong with its
    arithmetic rather than with its exploration.
    """

    def __init__(self, rng: Rng, target: float = 0.7) -> None:
        super().__init__(rng)
        self.target = target
        self.observation_space = Box([-1.0], [1.0])
        self.action_space = Box([-2.0], [2.0])
        self.spec = EnvSpec(
            name="aim",
            summary="Hit a number that the environment knows and does not say.",
            max_episode_steps=1,
        )

    def _reset(self) -> Point:
        return (0.0,)

    def _step(self, action: Point) -> Step[Point]:
        missed = action[0] - self.target
        return Step((0.0,), -(missed**2), terminated=True, truncated=False)


def an_agent(
    env: Env[Point, Point], seed: int = 1, **options: float
) -> GaussianActorCritic[Point]:
    encoder, features = encoder_for(env.observation_space)
    box = env.action_space
    assert isinstance(box, Box)
    return GaussianActorCritic(
        Rng(seed).stream("agent"), box, encoder, features, **options
    )


class TestWhatItAimsAt:
    def test_the_mean_starts_in_the_middle_of_the_box(self) -> None:
        # A small last layer, so the tanh starts near zero and the mean starts
        # near the middle. An agent that started at a bound would have to be
        # talked out of it before it learned anything.
        agent = an_agent(Aim(Rng(1).stream("env")))
        assert agent.greedy((0.0,))[0] == pytest.approx(0.0, abs=0.2)

    def test_the_mean_can_never_leave_the_box(self) -> None:
        """Which is what the tanh is for.

        The weights are pushed hard one way by hand, and the mean stops at the
        bound rather than walking out past it. A mean outside the box would
        clip every draw to the same number, and the gradient of a clipped
        action keeps pushing it further out.
        """
        agent = an_agent(Aim(Rng(1).stream("env")))
        for tensor in agent.means.parameters():
            tensor.data[:] = [value + 50.0 for value in tensor.data]
        aimed = agent.greedy((0.0,))[0]
        assert -2.0 <= aimed <= 2.0
        assert aimed == pytest.approx(2.0, abs=1e-6)

    def test_every_action_it_draws_is_inside_the_box(self) -> None:
        agent = an_agent(Aim(Rng(1).stream("env")), spread=5.0)
        for _ in range(200):
            drawn = agent.act((0.0,))
            assert agent.box.contains(drawn)

    def test_a_wider_spread_wanders_further(self) -> None:
        narrow = an_agent(Aim(Rng(1).stream("env")), spread=0.05)
        wide = an_agent(Aim(Rng(1).stream("env")), spread=1.0)
        near = [narrow.act((0.0,))[0] for _ in range(200)]
        far = [wide.act((0.0,))[0] for _ in range(200)]
        assert max(near) - min(near) < max(far) - min(far)

    def test_a_spread_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="never explores"):
            an_agent(Aim(Rng(1).stream("env")), spread=0.0)

    def test_it_ranks_no_actions(self) -> None:
        assert an_agent(Aim(Rng(1).stream("env"))).action_values((0.0,)) is None


class TestTheLogDensity:
    def test_it_is_the_normal_density_without_its_constant(self) -> None:
        agent = an_agent(Aim(Rng(1).stream("env")), spread=0.4)
        features = agent.encoder((0.0,))
        aimed = agent.mean(features).item()

        for action in (-1.0, 0.0, 0.25, 1.5):
            got = agent._log_density(features, (action,)).item()
            want = -0.5 * ((action - aimed) / 0.4) ** 2 - math.log(0.4)
            assert got == pytest.approx(want)

    def test_the_action_it_aims_at_is_the_likeliest(self) -> None:
        agent = an_agent(Aim(Rng(1).stream("env")))
        features = agent.encoder((0.0,))
        aimed = agent.mean(features).item()
        best = agent._log_density(features, (aimed,)).item()
        for away in (-0.5, -0.1, 0.1, 0.5):
            assert agent._log_density(features, (aimed + away,)).item() < best


class TestItFindsANumberItCanBeToldExactly:
    @pytest.mark.parametrize("target", [0.7, -1.4])
    def test_it_aims_at_the_target(self, target: float) -> None:
        env = Aim(Rng(1).stream("env"), target=target)
        agent = an_agent(env, step_size=0.05, spread_step_size=0.02)
        train(env, agent, 1200, discount=1.0)
        assert agent.greedy((0.0,))[0] == pytest.approx(target, abs=0.15)

    def test_it_narrows_as_it_arrives(self) -> None:
        # The spread is the exploration, and a policy that has found the
        # answer has no reason to keep wandering away from it.
        env = Aim(Rng(1).stream("env"))
        agent = an_agent(env, step_size=0.05, spread_step_size=0.02)
        started = agent.spread()[0]
        train(env, agent, 1200, discount=1.0)
        assert agent.spread()[0] < started / 4.0

    def test_the_value_network_learns_what_the_policy_is_worth(self) -> None:
        env = Aim(Rng(1).stream("env"))
        agent = an_agent(env, step_size=0.05, spread_step_size=0.02)
        train(env, agent, 1200, discount=1.0)
        # A policy aiming at the target with a spread of s is worth minus s
        # squared, because the miss is the draw and its square averages to the
        # variance.
        spread = agent.spread()[0]
        assert agent.state_value((0.0,)) == pytest.approx(-(spread**2), abs=0.02)


class TestWhatItLearned:
    def test_it_names_both_networks_and_the_spread(self) -> None:
        agent = an_agent(Aim(Rng(1).stream("env")))
        lines = list(agent.learned())
        assert any(line.startswith("means.") for line in lines)
        assert any(line.startswith("value.") for line in lines)
        assert sum(line.startswith("log spread|") for line in lines) == 1

    def test_learning_moves_it(self) -> None:
        env = Aim(Rng(1).stream("env"))
        agent = an_agent(env, step_size=0.05)
        before = list(agent.learned())
        train(env, agent, 20, discount=1.0)
        assert list(agent.learned()) != before

    def test_it_says_what_it_is_over(self) -> None:
        assert repr(an_agent(Aim(Rng(1).stream("env")))).startswith(
            "GaussianActorCritic(QNetwork("
        )


class TestTheEntropyBonus:
    def test_no_bonus_leaves_the_spread_to_the_gradient(self) -> None:
        env = Aim(Rng(1).stream("env"))
        agent = an_agent(env, step_size=0.05, spread_step_size=0.02, entropy=0.0)
        train(env, agent, 400, discount=1.0)
        assert agent.spread()[0] < 0.5

    def test_a_large_bonus_holds_the_spread_open(self) -> None:
        env = Aim(Rng(1).stream("env"))
        agent = an_agent(env, step_size=0.05, spread_step_size=0.02, entropy=1.0)
        train(env, agent, 400, discount=1.0)
        assert agent.spread()[0] > 0.5


class TestTheRegistry:
    def test_it_builds_on_the_environment_with_a_box_of_actions(self) -> None:
        env = ENVIRONMENTS.make("pendulum", Rng(1).stream("env"))
        agent = AGENTS.make("gaussian-actor-critic", Rng(1).stream("agent"), env)
        assert isinstance(agent, GaussianActorCritic)

    def test_an_environment_with_a_list_of_actions_is_refused(self) -> None:
        env = ENVIRONMENTS.make("pendulum-levels", Rng(1).stream("env"))
        with pytest.raises(TypeError, match="aims at a point in a box"):
            AGENTS.make("gaussian-actor-critic", Rng(1).stream("agent"), env)

    def test_the_message_says_where_to_find_the_list(self) -> None:
        env = ENVIRONMENTS.make("cliff", Rng(1).stream("env"))
        with pytest.raises(TypeError, match="pendulum-levels"):
            AGENTS.make("gaussian-actor-critic", Rng(1).stream("agent"), env)


class TestStandardising:
    def test_the_mean_comes_off_and_the_spread_divides(self) -> None:
        assert standardised([1.0, 2.0, 3.0]) == pytest.approx(
            [-1.224744871, 0.0, 1.224744871]
        )

    def test_values_that_are_all_alike_come_back_as_nothing(self) -> None:
        assert standardised([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]


class TestOneStepAtATime:
    def test_it_learns_nothing_until_the_episode_ends(self) -> None:
        env = Aim(Rng(1).stream("env"))
        agent = an_agent(env, step_size=0.05)
        before = list(agent.learned())
        for _ in range(20):
            agent.observe(Transition((0.0,), (1.0,), -0.1, (0.0,), True, False))
        assert list(agent.learned()) == before
        agent.end_episode()
        assert list(agent.learned()) != before
