"""Tests for Q-learning with a network in place of the table.

The two settings this agent exists to study are `replay` and `target_refresh`,
and turning either off has to be a real difference rather than a flag that
changes nothing. Most of what is here checks exactly that.
"""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.features import encoder_for, one_hot
from rel.agents.value_network import DeepQ
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

FOUR = Discrete(4)


def an_agent(**extra: object) -> DeepQ[int]:
    settings: dict[str, object] = {
        "step_size": 0.05,
        "discount": 0.9,
        "epsilon": 0.0,
        "replay": 100,
        "batch": 4,
        "target_refresh": 10,
        **extra,
    }
    return DeepQ(Rng(1), FOUR, one_hot(6), 6, **settings)  # type: ignore[arg-type]


def go(
    state: int, action: int, reward: float, landed: int, **flags: bool
) -> Transition[int]:
    return Transition(
        state,
        action,
        reward,
        landed,
        flags.get("terminated", False),
        flags.get("truncated", False),
    )


class TestTheShape:
    def test_it_gives_a_value_for_every_action(self) -> None:
        assert len(an_agent().action_values(0) or []) == 4

    def test_it_chooses_a_legal_action(self) -> None:
        agent = an_agent()
        assert 0 <= agent.act(0) < 4
        assert 0 <= agent.greedy(0) < 4

    def test_exploring_takes_something_other_than_the_best(self) -> None:
        agent = an_agent(epsilon=1.0)
        assert len({agent.act(0) for _ in range(40)}) > 1

    def test_it_has_an_opinion_everywhere(self) -> None:
        # A network is not a table with holes in it. Saying otherwise would
        # draw a value map with gaps the agent does not have.
        assert an_agent().knows(5)

    def test_a_batch_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            an_agent(batch=0)

    def test_a_negative_buffer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no fewer than no steps"):
            an_agent(replay=-1)

    def test_a_negative_refresh_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not negative"):
            an_agent(target_refresh=-1)


class TestTheTargetNetwork:
    def test_it_is_a_second_copy_when_asked_for(self) -> None:
        agent = an_agent(target_refresh=10)
        assert agent.target is not None
        assert agent.target.values([1.0] + [0.0] * 5) == agent.live.values(
            [1.0] + [0.0] * 5
        )

    def test_there_is_none_when_the_refresh_is_off(self) -> None:
        assert an_agent(target_refresh=0).target is None

    def test_it_holds_still_between_refreshes(self) -> None:
        # The point of having one. The thing being fitted must not chase
        # itself, so the target's weights do not move while the live ones do.
        agent = an_agent(target_refresh=1000)
        assert agent.target is not None
        before = agent.target.values([1.0] + [0.0] * 5)

        for _ in range(20):
            agent.observe(go(0, 1, 1.0, 1))

        assert agent.target.values([1.0] + [0.0] * 5) == before
        assert agent.live.values([1.0] + [0.0] * 5) != before

    def test_a_refresh_takes_the_live_weights(self) -> None:
        agent = an_agent(target_refresh=5)
        assert agent.target is not None
        for _ in range(5):
            agent.observe(go(0, 1, 1.0, 1))

        assert agent.refreshes == 1
        assert agent.target.values([1.0] + [0.0] * 5) == agent.live.values(
            [1.0] + [0.0] * 5
        )

    def test_with_no_target_the_live_network_supplies_the_target(self) -> None:
        agent = an_agent(target_refresh=0)
        assert agent.target is None
        agent.observe(go(0, 1, 1.0, 1))
        assert agent.refreshes == 0


class TestTheBuffer:
    def test_it_remembers_when_asked_to(self) -> None:
        agent = an_agent(replay=100)
        assert agent.memory is not None
        agent.observe(go(0, 1, 1.0, 1))
        assert len(agent.memory) == 1

    def test_there_is_none_when_the_buffer_is_off(self) -> None:
        assert an_agent(replay=0).memory is None

    def test_with_no_buffer_it_learns_from_the_step_it_took(self) -> None:
        # And from that step once, not `batch` times. The same gradient added
        # up is a larger step and not a larger batch.
        agent = an_agent(replay=0, batch=8)
        before = agent.live.values(one_hot(6)(0))
        agent.observe(go(0, 1, 1.0, 1))
        assert agent.live.values(one_hot(6)(0)) != before


class TestTheTarget:
    def test_a_terminated_episode_has_no_future(self) -> None:
        agent = an_agent()
        assert agent._target_for(go(0, 1, 7.0, 1, terminated=True)) == pytest.approx(
            7.0
        )

    def test_a_cut_off_episode_keeps_its_future(self) -> None:
        # The step limit is not an ending, and treating it as one is the fault
        # this whole project is careful about.
        agent = an_agent()
        cut = agent._target_for(go(0, 1, 7.0, 1, truncated=True))
        ended = agent._target_for(go(0, 1, 7.0, 1, terminated=True))
        assert cut != ended

    def test_it_is_the_reward_plus_the_best_of_the_next(self) -> None:
        agent = an_agent(discount=0.5)
        ahead = agent.target
        assert ahead is not None
        best = max(ahead.values(one_hot(6)(1)))
        assert agent._target_for(go(0, 1, 3.0, 1)) == pytest.approx(3.0 + 0.5 * best)


class TestItLearns:
    def test_it_reaches_the_end_of_the_walk(self) -> None:
        # The smallest environment here that a network can be pointed at, so
        # this stays a test rather than a measurement.
        root = Rng(1)
        env = ENVIRONMENTS.make("walk", root.stream("env"))
        encoder, features = encoder_for(env.observation_space)
        agent = DeepQ(
            root.stream("agent"),
            env.action_space,
            encoder,
            features,
            step_size=0.02,
            discount=0.99,
            epsilon=0.1,
        )
        record = train(env, agent, 150, discount=0.99)
        assert sum(record.returns[-20:]) / 20 > 0.9

    def test_the_same_seed_gives_the_same_run(self) -> None:
        def digest(seed: int) -> str:
            root = Rng(seed)
            env = ENVIRONMENTS.make("walk", root.stream("env"))
            encoder, features = encoder_for(env.observation_space)
            agent = DeepQ(root.stream("agent"), env.action_space, encoder, features)
            return train(env, agent, 10, discount=0.99).digest.hexdigest()

        assert digest(4) == digest(4)
        assert digest(4) != digest(5)
