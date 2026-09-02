"""Tests for the agents that learn a policy directly.

The gradients themselves are covered in `tests/test_autograd.py`, against the
definition of a derivative. What is checked here is that the pieces are put
together the right way round, and that the whole thing learns something on a
problem whose answer is known.

## What these tests do not claim

Policy gradient methods are sensitive to their settings in a way that the
tabular methods in this project are not. The measurements are in
`docs/algorithms.md`, and the short version is that REINFORCE with a baseline
solves the cart pole and this project's actor critic does not. That is reported
rather than tuned away, and these tests hold only what is stable across seeds.
"""

from __future__ import annotations

import math

import pytest

from rel.agents.features import centred, encoder_for, one_hot
from rel.agents.policy import ActorCritic, ClippedPolicy, Reinforce, standardised
from rel.envs.classic import cliff_walk
from rel.envs.control import CartPole
from rel.rng import Rng
from rel.spaces import Box, Discrete
from rel.training import evaluate, train


def build(cls: type[Reinforce[int]], seed: int = 5, **options: object):  # type: ignore[no-untyped-def]
    rng = Rng(seed)
    env = cliff_walk(rng.stream("env"))
    encoder, features = encoder_for(env.observation_space)
    settings: dict[str, object] = {"discount": 0.99, **options}
    agent = cls(
        rng.stream("agent"),
        env.action_space,
        encoder,
        features,
        **settings,  # type: ignore[arg-type]
    )
    return env, agent


class TestEncoding:
    def test_a_state_is_one_value_out_of_many(self) -> None:
        # The obvious encoding of state 37 is the number 37, and it is wrong:
        # a network reading it learns that 37 is nearly 38, which on a grid is
        # the cell to the right, and nothing like 25, which is directly above.
        encode = one_hot(4)
        assert list(encode(2)) == [0.0, 0.0, 1.0, 0.0]

    def test_a_state_outside_the_space_is_refused(self) -> None:
        with pytest.raises(IndexError):
            one_hot(4)(9)

    def test_a_box_is_centred_on_zero(self) -> None:
        # A network whose inputs are all positive gets a gradient on its first
        # layer that points the same way for every weight of a unit, so those
        # weights can only rise or fall together.
        encode = centred(Box([-1.0, 0.0], [1.0, 10.0]))
        assert list(encode((-1.0, 0.0))) == [-1.0, -1.0]
        assert list(encode((1.0, 10.0))) == [1.0, 1.0]
        assert list(encode((0.0, 5.0))) == [0.0, 0.0]

    def test_a_value_outside_the_box_is_held_at_the_edge(self) -> None:
        encode = centred(Box([-1.0], [1.0]))
        assert list(encode((-99.0,))) == [-1.0]

    def test_the_encoder_matches_the_space(self) -> None:
        encoder, size = encoder_for(Discrete(6))
        assert size == 6
        assert len(encoder(0)) == 6

        encoder, size = encoder_for(Box([0.0, 0.0], [1.0, 1.0]))
        assert size == 2

    def test_a_space_with_no_encoding_says_so(self) -> None:
        with pytest.raises(ValueError, match="starts at zero"):
            encoder_for(Discrete(4, start=1))


class TestStandardising:
    def test_the_mean_comes_out_at_zero(self) -> None:
        values = standardised([1.0, 2.0, 6.0, 3.0])
        assert sum(values) == pytest.approx(0.0)

    def test_the_spread_comes_out_at_one(self) -> None:
        values = standardised([1.0, 2.0, 6.0, 3.0])
        mean = sum(values) / len(values)
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        assert spread == pytest.approx(1.0)

    def test_values_that_are_all_the_same_do_not_divide_by_zero(self) -> None:
        assert standardised([4.0, 4.0, 4.0]) == [0.0, 0.0, 0.0]


class TestThePolicy:
    def test_the_probabilities_add_up_to_one(self) -> None:
        _, agent = build(Reinforce)
        assert sum(agent.probabilities(0)) == pytest.approx(1.0)

    def test_it_starts_near_uniform(self) -> None:
        # A policy that starts confident explores nothing and takes a long
        # time to be talked out of it.
        _, agent = build(Reinforce)
        for state in range(0, 48, 7):
            assert max(agent.probabilities(state)) < 0.4

    def test_acting_uses_the_probabilities(self) -> None:
        _, agent = build(Reinforce)
        chosen = {agent.act(0) for _ in range(200)}
        assert chosen == {0, 1, 2, 3}

    def test_greedy_takes_the_most_likely_action(self) -> None:
        _, agent = build(Reinforce)
        shares = agent.probabilities(0)
        best = shares.index(max(shares))
        assert {agent.greedy(0) for _ in range(20)} == {best}


class TestTheReturns:
    def test_the_return_of_a_step_is_the_rest_of_the_episode(self) -> None:
        from rel.agents.base import Transition

        _, agent = build(Reinforce, discount=0.5)
        agent._episode = [
            Transition(0, 0, 1.0, 1, False, False),
            Transition(1, 0, 2.0, 2, False, False),
            Transition(2, 0, 4.0, 3, True, False),
        ]
        # 1 + 0.5 * 2 + 0.25 * 4, then 2 + 0.5 * 4, then 4.
        assert agent._returns() == pytest.approx([3.0, 4.0, 4.0])

    def test_an_ending_stops_the_return_reaching_back(self) -> None:
        from rel.agents.base import Transition

        _, agent = build(Reinforce, discount=1.0)
        agent._episode = [
            Transition(0, 0, 1.0, 1, True, False),
            Transition(2, 0, 5.0, 3, True, False),
        ]
        assert agent._returns() == pytest.approx([1.0, 5.0])

    def test_a_step_limit_does_not_stop_the_return_reaching_back(self) -> None:
        # An episode that the step limit cut off has a future. Reading it as
        # an ending is the fault that stopped REINFORCE reaching the goal on
        # three seeds of six of the cliff walk.
        from rel.agents.base import Transition

        _, agent = build(Reinforce, discount=0.5)
        assert agent.value is not None

        cut_off = Transition(0, 0, 1.0, 7, terminated=False, truncated=True)
        agent._episode = [cut_off]

        ahead = agent.value(agent.encoder(7)).item()
        assert agent._returns() == pytest.approx([1.0 + 0.5 * ahead])
        assert ahead != 0.0

    def test_a_cut_off_tail_without_a_baseline_is_the_mean_reward(self) -> None:
        # There is no network to ask, so the estimate is the reward of the
        # episode so far, paid for ever. On five steps of -1 with a discount
        # of 0.9 that is -10, and every step of the episode is worth -10.
        from rel.agents.base import Transition

        _, agent = build(Reinforce, discount=0.9, baseline=False)
        agent._episode = [
            Transition(step, 0, -1.0, step + 1, terminated=False, truncated=step == 4)
            for step in range(5)
        ]
        assert agent._returns() == pytest.approx([-10.0] * 5)

    def test_an_undiscounted_cut_off_episode_has_no_tail(self) -> None:
        # The undiscounted return of an episode that never finishes is not a
        # number. Zero is not right either, and it is what there is.
        from rel.agents.base import Transition

        _, agent = build(Reinforce, discount=1.0, baseline=False)
        agent._episode = [
            Transition(0, 0, -1.0, 1, terminated=False, truncated=False),
            Transition(1, 0, -1.0, 2, terminated=False, truncated=True),
        ]
        assert agent._returns() == pytest.approx([-2.0, -1.0])

    def test_a_uniformly_bad_episode_does_not_reward_its_own_last_steps(
        self,
    ) -> None:
        # This is the fault itself, written as the property it broke. Every
        # step of a cliff walk episode that never reaches the goal pays -1,
        # so no step of it is better than any other, and the weights must say
        # so. Under a zero tail the last steps drew about +3.2 and the first
        # about -0.8, which told the agent to keep doing whatever the step
        # limit stopped it doing.
        from rel.agents.base import Transition

        _, agent = build(Reinforce, discount=0.99, baseline=False)
        agent._episode = [
            Transition(step, 0, -1.0, step + 1, terminated=False, truncated=step == 499)
            for step in range(500)
        ]
        weights = standardised(agent._returns())
        assert max(weights) - min(weights) < 1e-6


class TestActorCriticTargets:
    def test_a_target_is_one_step_of_reward_plus_the_value_after_it(self) -> None:
        from rel.agents.base import Transition

        _, agent = build(ActorCritic, discount=0.5)
        assert agent.value is not None

        agent._episode = [
            Transition(0, 0, 1.0, 1, False, False),
            Transition(1, 0, 2.0, 2, True, False),
        ]
        features = [agent.encoder(step.observation) for step in agent._episode]
        targets, weights = agent._targets_and_weights(features)

        second = agent.value(features[1]).item()
        assert targets[0] == pytest.approx(1.0 + 0.5 * second)
        # The last step ended, so nothing after it counts.
        assert targets[1] == pytest.approx(2.0)

        # The weight is how much better the step turned out than the value
        # network expected, which is what pushes the action.
        first = agent.value(features[0]).item()
        assert weights[0] == pytest.approx(targets[0] - first)
        assert weights[1] == pytest.approx(targets[1] - second)

    def test_a_cut_off_episode_keeps_the_value_of_where_it_stopped(self) -> None:
        # The state a step limit stopped in has a future, and dropping it is
        # the fault this project warns about everywhere else.
        from rel.agents.base import Transition

        _, agent = build(ActorCritic, discount=0.5)
        assert agent.value is not None

        agent._episode = [Transition(0, 0, 1.0, 9, False, True)]
        features = [agent.encoder(agent._episode[0].observation)]
        targets, _ = agent._targets_and_weights(features)

        ahead = agent.value(agent.encoder(9)).item()
        assert targets[0] == pytest.approx(1.0 + 0.5 * ahead)


class TestItLearns:
    """Both of these find the best policy on the cliff walk.

    That is the claim these tests make and the only one. The cart pole is a
    harder problem for these methods, one of them does not solve it, and both
    of those facts are in `docs/algorithms.md` with the commands behind them
    rather than in a test that would have to be tuned to pass.
    """

    @pytest.mark.parametrize("builder", [Reinforce, ActorCritic])
    def test_it_finds_a_short_path_on_the_cliff_walk(
        self, builder: type[Reinforce[int]]
    ) -> None:
        # A bound rather than a number. The best possible is -13 and both of
        # these reach it on this seed, and pinning that would make the test
        # fail on any change that moved the answer by one step, which is not a
        # fault.
        #
        # This is one seed and it is not the whole story. Over six seeds
        # REINFORCE reaches the goal on four of them and never finishes on the
        # other two. `docs/algorithms.md` has that measurement and the command.
        env, agent = build(builder, seed=5)
        train(env, agent, 400, discount=0.99)

        watched = evaluate(cliff_walk(Rng(6).stream("env")), agent, 5, discount=0.99)
        assert -30.0 < watched.final() < -10.0

    def test_the_baseline_can_be_switched_off(self) -> None:
        env, agent = build(Reinforce, baseline=False)
        assert agent.value is None
        train(env, agent, 200, discount=0.99)
        assert agent.episodes == 200

    def test_an_entropy_bonus_keeps_the_policy_from_going_certain(self) -> None:
        # Without it a policy that has found something good keeps sharpening,
        # and once it is nearly deterministic it stops seeing what the other
        # actions would have paid.
        def sharpest(entropy: float) -> float:
            env, agent = build(Reinforce, entropy=entropy, seed=8)
            train(env, agent, 300, discount=0.99)
            start = env.state_of(3, 0)
            return max(agent.probabilities(start))

        assert sharpest(0.0) > sharpest(0.5)


@pytest.mark.slow
def test_reinforce_keeps_the_cart_pole_up() -> None:
    # Measured over three seeds at a thousand episodes: 500, 500 and 189.
    # This runs one of them.
    rng = Rng(5)
    env = CartPole(rng.stream("env"))
    encoder, features = encoder_for(env.observation_space)
    agent = Reinforce(
        rng.stream("agent"), env.action_space, encoder, features, discount=0.99
    )
    train(env, agent, 1000, discount=0.99)

    watched = evaluate(CartPole(Rng(105).stream("env")), agent, 10, discount=0.99)
    assert watched.final() > 300.0


class TestTheClip:
    """When the clip stops a step moving, and when it leaves it alone.

    The whole method is this rule. Get it backwards and the agent still runs,
    still learns, and pushes hardest exactly where it has least evidence.
    """

    @staticmethod
    def agent(**extra: object) -> ClippedPolicy[int]:
        _, made = build(ClippedPolicy, **extra)  # type: ignore[arg-type]
        return made  # type: ignore[return-value]

    def test_a_good_step_is_stopped_once_the_ratio_is_too_large(self) -> None:
        made = self.agent(clip_range=0.2)
        assert made.binds(1.3, 1.0)
        assert not made.binds(1.1, 1.0)
        assert not made.binds(0.1, 1.0)

    def test_a_bad_step_is_stopped_once_the_ratio_is_too_small(self) -> None:
        made = self.agent(clip_range=0.2)
        assert made.binds(0.7, -1.0)
        assert not made.binds(0.9, -1.0)
        assert not made.binds(9.0, -1.0)

    def test_the_boundary_itself_is_not_stopped(self) -> None:
        made = self.agent(clip_range=0.2)
        assert not made.binds(1.2, 1.0)
        assert not made.binds(0.8, -1.0)

    def test_an_advantage_of_nothing_is_never_stopped(self) -> None:
        # It moves the policy nowhere either way, so there is nothing to stop.
        made = self.agent(clip_range=0.2)
        assert not made.binds(9.0, 0.0)
        assert not made.binds(0.1, 0.0)

    def test_a_wider_range_stops_less(self) -> None:
        assert self.agent(clip_range=0.2).binds(1.3, 1.0)
        assert not self.agent(clip_range=0.5).binds(1.3, 1.0)

    def test_a_run_of_no_passes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least once"):
            self.agent(passes=0)

    def test_a_clip_range_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above nothing"):
            self.agent(clip_range=0.0)


class TestOnePassIsReinforce:
    """One pass over an episode is REINFORCE with a baseline, exactly.

    Before any pass the policy is the one that collected the episode, so every
    ratio is one, and the gradient of `weight * exp(logp - logp)` is `weight`
    where REINFORCE's `weight * logp` is also `weight`. That is not an
    approximation and the digests hold it to the last bit.

    It is the load bearing test of this class. If the ratio were built wrongly
    the agent would still learn, because a wrong constant in front of the
    gradient is a step size, and nothing about a run would say so.
    """

    @staticmethod
    def run(cls: type[Reinforce[int]], **extra: object) -> tuple[str, str]:
        env, agent = build(cls, seed=4, entropy=0.05, **extra)
        record = train(env, agent, 20, discount=0.99)
        return record.digest.hexdigest(), "".join(agent.learned())

    def test_the_two_runs_are_the_same_run(self) -> None:
        assert self.run(ClippedPolicy, passes=1) == self.run(Reinforce, baseline=True)

    def test_two_passes_are_a_different_run(self) -> None:
        assert self.run(ClippedPolicy, passes=2) != self.run(ClippedPolicy, passes=1)


class TestWhatItCounts:
    def test_nothing_considered_is_a_share_of_nothing(self) -> None:
        _, agent = build(ClippedPolicy)
        assert agent.share_clipped == 0.0

    def test_the_first_pass_never_clips(self) -> None:
        # Every ratio is one before the policy has moved, so a run of one pass
        # cannot clip however narrow the range is.
        env, agent = build(ClippedPolicy, passes=1, clip_range=1e-9)
        train(env, agent, 5, discount=0.99)
        assert agent.considered > 0
        assert agent.clipped == 0

    def test_a_narrow_range_clips_more_than_a_wide_one(self) -> None:
        shares = []
        for width in (0.02, 2.0):
            env, agent = build(ClippedPolicy, passes=4, clip_range=width)
            train(env, agent, 12, discount=0.99)
            shares.append(agent.share_clipped)
        assert shares[0] > shares[1]

    def test_it_counts_one_step_of_every_pass(self) -> None:
        env, agent = build(ClippedPolicy, passes=3)
        record = train(env, agent, 4, discount=0.99)
        assert agent.considered == 3 * sum(record.lengths)

    def test_it_says_how_it_is_set(self) -> None:
        _, agent = build(ClippedPolicy, passes=6, clip_range=0.3)
        assert "passes=6" in repr(agent)
        assert "clip_range=0.3" in repr(agent)
