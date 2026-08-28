"""Tests for learning about one policy while following another.

The correction is a ratio, and the two estimators divide the same product by
different things. What is checked here is the ratio itself, the rule that a
zero cuts the episode, and the property that separates the two estimators: one
is unbiased and the other is steadier, and neither of those is a preference.
"""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.dp import evaluate_policy
from rel.agents.off_policy import ESTIMATORS, OffPolicyMonteCarlo, ratio
from rel.envs.classic import cliff_walk, frozen_lake
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train


class TestTheRatio:
    def test_it_is_one_when_the_policies_agree(self) -> None:
        same = [0.25, 0.25, 0.25, 0.25]
        assert ratio(same, same, 2) == 1.0

    def test_it_is_the_quotient_of_the_two(self) -> None:
        assert ratio([1.0, 0.0], [0.5, 0.5], 0) == 2.0
        assert ratio([0.6, 0.4], [0.3, 0.7], 1) == pytest.approx(0.4 / 0.7)

    def test_it_is_zero_where_the_target_policy_never_goes(self) -> None:
        # This is what cuts an episode. A greedy target policy puts zero on
        # every action but one, so the ratio is zero the moment the behaviour
        # policy explored.
        assert ratio([1.0, 0.0], [0.9, 0.1], 1) == 0.0

    def test_a_behaviour_policy_with_no_coverage_is_refused(self) -> None:
        # Dividing by zero here would give an infinity that spreads through
        # the whole table and never says where it came from.
        with pytest.raises(ValueError, match="coverage"):
            ratio([0.5, 0.5], [1.0, 0.0], 1)


class TestTheTargetPolicy:
    def test_all_of_the_weight_is_on_the_best_action(self) -> None:
        agent: OffPolicyMonteCarlo[int] = OffPolicyMonteCarlo(
            Rng(1).stream("a"), Discrete(3), epsilon=0.1
        )
        agent.values(0)[:] = [1.0, 5.0, 2.0]
        assert agent.target_probabilities(0) == [0.0, 1.0, 0.0]

    def test_the_behaviour_policy_still_explores(self) -> None:
        # The two have to differ, or there is nothing to correct.
        agent: OffPolicyMonteCarlo[int] = OffPolicyMonteCarlo(
            Rng(1).stream("a"), Discrete(3), epsilon=0.3
        )
        agent.values(0)[:] = [1.0, 5.0, 2.0]
        behaviour = agent.policy_probabilities(0)
        assert min(behaviour) > 0.0
        assert behaviour != agent.target_probabilities(0)

    def test_an_estimator_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ValueError, match="estimator is one of"):
            OffPolicyMonteCarlo(Rng(1).stream("a"), Discrete(2), estimator="clever")  # type: ignore[arg-type]


class TestTheEpisodeIsCutAtAnExploringStep:
    def test_a_step_before_an_exploring_one_is_not_credited(self) -> None:
        agent: OffPolicyMonteCarlo[int] = OffPolicyMonteCarlo(
            Rng(1).stream("a"), Discrete(2), epsilon=0.2, discount=1.0
        )
        # Action 1 is greedy at state 5, so taking action 0 there explores.
        agent.values(5)[:] = [0.0, 9.0]

        agent.start_episode()
        agent.observe(Transition(0, 0, 1.0, 5, terminated=False, truncated=False))
        agent.observe(Transition(5, 0, 1.0, 6, terminated=True, truncated=False))
        agent.end_episode()

        # State 5 is the step that explored, so it is credited. State 0 comes
        # before it and is multiplied by a ratio of zero.
        assert agent.knows(5)
        assert agent.peek(0)[0] == 0.0

    def test_a_greedy_episode_credits_all_of_it(self) -> None:
        agent: OffPolicyMonteCarlo[int] = OffPolicyMonteCarlo(
            Rng(1).stream("a"), Discrete(2), epsilon=0.2, discount=1.0
        )
        agent.values(0)[:] = [9.0, 0.0]
        agent.values(5)[:] = [9.0, 0.0]

        agent.start_episode()
        agent.observe(Transition(0, 0, 1.0, 5, terminated=False, truncated=False))
        agent.observe(Transition(5, 0, 1.0, 6, terminated=True, truncated=False))
        agent.end_episode()

        assert agent.peek(0)[0] != 9.0


class TestTheTwoEstimators:
    def test_both_reach_the_true_value_of_a_known_answer(self) -> None:
        # One state, one step, a reward of exactly 1. Whatever the correction
        # does along the way, both estimators have to land on 1.
        for estimator in ESTIMATORS:
            agent: OffPolicyMonteCarlo[int] = OffPolicyMonteCarlo(
                Rng(1).stream("a"),
                Discrete(2),
                epsilon=0.0,
                discount=1.0,
                estimator=estimator,
            )
            for _ in range(60):
                agent.start_episode()
                agent.observe(
                    Transition(0, 0, 1.0, 1, terminated=True, truncated=False)
                )
                agent.end_episode()
            assert agent.peek(0)[0] == pytest.approx(1.0, abs=0.05), estimator

    def test_the_ordinary_estimator_moves_about_more(self) -> None:
        # This is the whole lesson of the module. The ordinary estimator
        # divides by a count and the weighted one divides by the ratios it
        # really saw, so a large ratio moves the ordinary one a long way and
        # the weighted one hardly at all.
        def spread(estimator: str) -> float:
            seen: list[float] = []
            for seed in range(1, 7):
                rng = Rng(seed)
                env = frozen_lake(rng.stream("env"))
                agent = OffPolicyMonteCarlo(
                    rng.stream("agent"),
                    env.action_space,
                    epsilon=0.2,
                    discount=1.0,
                    estimator=estimator,  # type: ignore[arg-type]
                )
                train(env, agent, 400, discount=1.0)
                seen.append(agent.peek(0)[0])
            middle = sum(seen) / len(seen)
            return max(abs(value - middle) for value in seen)

        assert spread("ordinary") > spread("weighted")


class TestItLearns:
    @pytest.mark.parametrize("estimator", list(ESTIMATORS))
    def test_it_reaches_a_good_policy_on_the_frozen_lake(self, estimator: str) -> None:
        rng = Rng(4)
        env = frozen_lake(rng.stream("env"))
        agent = OffPolicyMonteCarlo(
            rng.stream("agent"),
            env.action_space,
            epsilon=0.1,
            discount=1.0,
            estimator=estimator,  # type: ignore[arg-type]
        )
        train(env, agent, 3000, discount=1.0)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=1.0)
        assert report.reaches_end
        assert report.start_value > 0.5


class TestWhereItCannotLearn:
    """The weakness, pinned rather than left for somebody to rediscover.

    The walk backwards stops at the first step the behaviour policy explored,
    so an episode teaches only its tail. Where episodes are long and rarely
    finish, that tail never reaches anything worth knowing.
    """

    def test_it_credits_about_one_cell_per_episode_on_the_cliff_walk(self) -> None:
        rng = Rng(4)
        env = cliff_walk(rng.stream("env"))
        agent: OffPolicyMonteCarlo[int] = OffPolicyMonteCarlo(
            rng.stream("agent"), env.action_space, epsilon=0.1, discount=1.0
        )

        moved: list[int] = []
        learn = agent._learn

        def watched() -> None:
            before = {key: list(row) for key, row in agent.q.items()}
            learn()
            moved.append(
                sum(1 for key, row in agent.q.items() if before.get(key) != row)
            )

        agent._learn = watched  # type: ignore[method-assign]
        train(env, agent, 200, discount=1.0)

        # A cliff walk episode runs to its five hundred step limit far more
        # often than it reaches the goal, and each one moves about one cell.
        assert sum(moved) / len(moved) < 2.0

    def test_it_does_not_reach_the_goal_there(self) -> None:
        rng = Rng(4)
        env = cliff_walk(rng.stream("env"))
        agent: OffPolicyMonteCarlo[int] = OffPolicyMonteCarlo(
            rng.stream("agent"), env.action_space, epsilon=0.1, discount=1.0
        )
        train(env, agent, 600, discount=1.0)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        assert not evaluate_policy(env, policy, discount=1.0).reaches_end
