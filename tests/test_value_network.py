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
) -> Transition[int, int]:
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


class TestPriorityIsPassedThrough:
    def test_the_buffer_draws_evenly_by_default(self) -> None:
        agent = an_agent()
        assert agent.memory is not None
        assert agent.memory.priority == 0.0
        assert agent.memory.weighting == 0.0

    def test_the_settings_reach_the_buffer(self) -> None:
        agent = an_agent(priority=0.7, weighting=0.4)
        assert agent.memory is not None
        assert agent.memory.priority == 0.7
        assert agent.memory.weighting == 0.4

    def test_a_bad_setting_is_refused_where_it_is_meant(self) -> None:
        with pytest.raises(ValueError, match="priority power"):
            an_agent(priority=2.0)

    def test_the_priorities_are_told_what_the_errors_were(self) -> None:
        agent = an_agent(priority=1.0, replay=10, batch=4)
        assert agent.memory is not None
        # A step worth ten against steps worth nothing. Once the agent has
        # been wrong about it, its priority must stand above the rest.
        agent.observe(go(0, 1, 0.0, 1))
        for _ in range(6):
            agent.observe(go(2, 0, 0.0, 2))
        agent.observe(go(0, 1, 10.0, 1))

        weights = agent.memory.priorities()
        for place, held in enumerate(agent.memory.steps()):
            if held.reward == 10.0:
                assert weights[place] == max(weights)

    def test_a_priority_draw_changes_the_run(self) -> None:
        def learned(**extra: object) -> list[float]:
            agent = an_agent(replay=50, batch=4, **extra)
            for number in range(40):
                agent.observe(go(number % 6, number % 4, float(number % 3), 1))
            return list(agent.live.values(one_hot(6)(0)))

        even = learned()
        assert learned() == even
        assert learned(priority=1.0) != even
        assert learned(priority=1.0, weighting=1.0) != learned(priority=1.0)

    def test_the_even_draw_takes_the_same_step_it_always_did(self) -> None:
        # The weights are all one when the draw is even, and multiplying by
        # one is exact, so nothing recorded before this setting existed moves.
        agent = an_agent(replay=50, batch=4)
        for number in range(20):
            agent.observe(go(number % 6, number % 4, float(number % 3), 1))
        # Taken from the agent as it stood before priority existed, and asked
        # for exactly rather than nearly, because that is the claim.
        assert agent.live.values(one_hot(6)(0)) == [
            2.4250550281318835,
            1.808761350592422,
            1.7114970493054495,
            1.913814023247627,
        ]


class TestDoubleEstimation:
    def test_it_is_off_by_default(self) -> None:
        assert an_agent().double is False

    def test_it_needs_a_target_network_to_value_with(self) -> None:
        # Without one there is no second opinion to split off, so asking for
        # it would be Q-learning with extra steps.
        with pytest.raises(ValueError, match="needs a target network"):
            an_agent(double=True, target_refresh=0)

    def test_the_live_network_names_the_action_and_the_target_values_it(
        self,
    ) -> None:
        agent = an_agent(double=True, discount=1.0)
        assert agent.target is not None
        # Two networks that disagree about which action of state 1 is best.
        # Plain Q-learning would take the largest of the target's row; double
        # takes the entry the live network points at.
        features = one_hot(6)(1)
        naming = agent.live.values(features)
        valuing = agent.target.values(features)
        chosen = naming.index(max(naming))

        target = agent._target_for(go(0, 1, 2.0, 1))
        assert target == pytest.approx(2.0 + valuing[chosen])

    def test_without_it_the_target_takes_the_largest_of_the_row(self) -> None:
        agent = an_agent(double=False, discount=1.0)
        assert agent.target is not None
        valuing = agent.target.values(one_hot(6)(1))
        assert agent._target_for(go(0, 1, 2.0, 1)) == pytest.approx(2.0 + max(valuing))

    def test_the_two_differ_once_the_networks_disagree(self) -> None:
        # If they never differed this setting would be a flag that changes
        # nothing, which is the failure mode worth a test. They agree while
        # the two copies are the same, so the live one is moved first.
        agent = an_agent(double=True, discount=1.0, target_refresh=1000)
        for number in range(30):
            agent.observe(go(number % 6, number % 4, float(number % 5), number % 6))

        differ = 0
        for state in range(6):
            step = go(0, 1, 0.0, state)
            splitting = agent._target_for(step)
            agent.double = False
            plain = agent._target_for(step)
            agent.double = True
            differ += splitting != plain
        assert differ > 0

    def test_a_terminated_step_has_no_future_either_way(self) -> None:
        for double in (False, True):
            agent = an_agent(double=double)
            assert agent._target_for(go(0, 1, 3.0, 1, terminated=True)) == 3.0

    def test_the_choice_does_not_spend_randomness(self) -> None:
        # A tie broken by a draw inside a target would make two runs of one
        # seed differ by how many ties the network happened to have.
        agent = an_agent(double=True)
        before = agent.rng.snapshot()
        agent._target_for(go(0, 1, 1.0, 1))
        assert agent.rng.snapshot() == before

    def test_the_double_target_is_never_above_the_plain_one(self) -> None:
        # Which is the whole point: the plain target takes the largest of the
        # valuing row and the double target takes some entry of it.
        agent = an_agent(double=True, discount=1.0)
        assert agent.target is not None
        for state in range(6):
            step = go(state, 1, 0.0, state)
            splitting = agent._target_for(step)
            agent.double = False
            plain = agent._target_for(step)
            agent.double = True
            assert splitting <= plain


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


class TestBothPiecesAreNeededOnTheCartPole:
    """The ablation, cut down to a test.

    Neither piece alone is distinguishable from neither. That is the finding
    `scripts/measure_value_network.py` measures over ten seeds, and this holds
    three of them to the part of it that is not close: with both, the agent
    keeps the pole up several times longer than a policy that has learned
    nothing.
    """

    EPISODES = 300
    SEEDS = (1, 2, 3)

    def _kept_up(self, replay: int, refresh: int) -> float:
        lasts = []
        for seed in self.SEEDS:
            root = Rng(seed)
            env = ENVIRONMENTS.make("cartpole", root.stream("env"))
            encoder, features = encoder_for(env.observation_space)
            agent = DeepQ(
                root.stream("agent"),
                env.action_space,
                encoder,
                features,
                step_size=0.02,
                discount=0.99,
                epsilon=0.1,
                replay=replay,
                target_refresh=refresh,
            )
            record = train(env, agent, self.EPISODES, discount=0.99)
            lasts.append(sum(record.lengths[-30:]) / 30)
        return sum(lasts) / len(lasts)

    def test_with_both_it_learns_something(self) -> None:
        assert self._kept_up(2000, 200) > 20.0

    def test_with_neither_it_does_not(self) -> None:
        # A pole that is not being balanced falls in about nine steps.
        assert self._kept_up(0, 0) < 15.0

    def test_replay_alone_is_not_enough(self) -> None:
        assert self._kept_up(2000, 0) < 20.0

    def test_a_target_network_alone_is_not_enough(self) -> None:
        assert self._kept_up(0, 200) < 15.0
