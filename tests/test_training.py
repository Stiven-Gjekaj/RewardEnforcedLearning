"""Tests for the loop and for what it writes down.

The digest is what the whole reproducibility claim rests on. If two runs that
took different paths could produce the same digest, every comparison made with
one would be worthless, and nothing else in the suite would notice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from rel.agents.base import Agent, RandomAgent, Transition
from rel.agents.td import QLearning
from rel.core import Env, EnvSpec, Outcome, Step, TabularEnv
from rel.envs.classic import cliff_walk
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import Digest, Episode, Record, evaluate, run_episode, train


class Corridor(TabularEnv):
    """Three cells in a line. Right reaches the end, left goes back."""

    def __init__(self, rng: Rng, steps: int | None = None) -> None:
        super().__init__(rng)
        self.observation_space = Discrete(3)
        self.action_space = Discrete(2)
        self.spec = EnvSpec(
            name="corridor",
            summary="Three cells in a line.",
            max_episode_steps=steps,
        )
        self.at = 0
        self.wandered = 0

    def _reset(self) -> int:
        self.at = 0
        self.wandered = 0
        return 0

    def _step(self, action: int) -> Step[int]:
        branch = self.transitions(self.at, action)[0]
        if branch.observation < self.at:
            self.wandered += 1
        self.at = branch.observation
        return Step(branch.observation, branch.reward, branch.terminated, False)

    def audit(self) -> Mapping[str, float]:
        return {"wandered": float(self.wandered)}

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if state == 2:
            return [Outcome(1.0, 2, 0.0, terminated=True)]
        landed = min(state + 1, 2) if action == 1 else max(state - 1, 0)
        return [Outcome(1.0, landed, -1.0, landed == 2)]

    def start_states(self) -> Sequence[tuple[float, int]]:
        return [(1.0, 0)]


class AlwaysRight(Agent[int]):
    def act(self, observation: int) -> int:
        return 1

    def greedy(self, observation: int) -> int:
        return 1


class TestOneEpisode:
    def test_it_reports_what_happened(self) -> None:
        episode = run_episode(Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)))
        assert episode.length == 2
        assert episode.total_reward == -2.0
        assert episode.terminated

    def test_the_discounted_return_uses_the_discount(self) -> None:
        episode = run_episode(
            Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)), discount=0.5
        )
        # -1, then half of -1.
        assert episode.discounted_reward == pytest.approx(-1.5)

    def test_the_audit_comes_from_the_environment(self) -> None:
        env = Corridor(Rng(1))
        episode = run_episode(env, AlwaysRight(Rng(1), Discrete(2)))
        assert episode.audit == {"wandered": 0.0}

    def test_a_truncated_episode_says_it_did_not_end(self) -> None:
        env = Corridor(Rng(1), steps=1)
        episode = run_episode(env, AlwaysRight(Rng(1), Discrete(2)))
        assert episode.length == 1
        assert not episode.terminated

    def test_learning_can_be_switched_off(self) -> None:
        env = Corridor(Rng(1))
        agent: QLearning[int] = QLearning(Rng(1), env.action_space)
        run_episode(env, agent, learn=False)
        assert agent.q == {}
        assert agent.episodes == 0

    def test_the_greedy_policy_can_be_watched_while_it_still_learns(self) -> None:
        # Two separate choices. An agent can be watched greedily while it
        # learns, and the loop has to let them be set apart.
        env = Corridor(Rng(1))
        agent: QLearning[int] = QLearning(Rng(1), env.action_space, epsilon=1.0)
        run_episode(env, agent, learn=True, greedy=True)
        assert agent.q != {}

    def test_the_steps_are_kept_when_they_are_asked_for(self) -> None:
        episode = run_episode(
            Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)), keep_steps=True
        )
        assert episode.rewards == (-1.0, -1.0)

    def test_the_steps_are_not_kept_otherwise(self) -> None:
        episode = run_episode(Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)))
        assert episode.rewards == ()


class TestTheRecord:
    def test_it_holds_one_entry_for_each_episode(self) -> None:
        record = train(Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)), 5)
        assert len(record) == 5
        assert record.returns == [-2.0] * 5
        assert record.steps == 10

    def test_the_final_mean_is_the_tail(self) -> None:
        record = Record()
        for number in range(200):
            record.add(
                Episode(
                    number=number,
                    total_reward=0.0 if number < 100 else 10.0,
                    discounted_reward=0.0,
                    length=1,
                    terminated=True,
                    audit={},
                )
            )
        assert record.final(100) == 10.0

    def test_the_audits_are_collected_by_name(self) -> None:
        record = train(Corridor(Rng(1)), RandomAgent(Rng(2), Discrete(2)), 6)
        assert "wandered" in record.audits
        assert len(record.audits["wandered"]) == 6

    def test_the_mean_reward_at_each_step_needs_the_steps(self) -> None:
        # A bandit run wants this: the interesting curve there is the reward at
        # each pull averaged over problems, and the total of a whole problem
        # hides the shape entirely.
        without = train(Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)), 3)
        assert without.reward_by_step() == []

        with_steps = train(
            Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)), 3, keep_steps=True
        )
        assert with_steps.reward_by_step() == [-1.0, -1.0]

    def test_a_step_that_only_some_episodes_reached_averages_over_those(self) -> None:
        record = Record()
        record.step_rewards = [(1.0, 2.0, 3.0), (5.0,)]
        assert record.reward_by_step() == [3.0, 2.0, 3.0]

    def test_it_says_which_episodes_ended_inside_the_rules(self) -> None:
        record = train(Corridor(Rng(1), steps=1), AlwaysRight(Rng(1), Discrete(2)), 3)
        assert record.terminated == [False, False, False]


class TestTheDigest:
    def test_two_runs_of_one_command_agree(self) -> None:
        def digest() -> str:
            env = cliff_walk(Rng(4).stream("env"))
            agent: QLearning[int] = QLearning(Rng(4).stream("agent"), env.action_space)
            return train(env, agent, 20).digest.hexdigest()

        assert digest() == digest()

    def test_two_seeds_do_not_agree(self) -> None:
        def digest(seed: int) -> str:
            env = cliff_walk(Rng(seed).stream("env"))
            agent: QLearning[int] = QLearning(
                Rng(seed).stream("agent"), env.action_space
            )
            return train(env, agent, 20).digest.hexdigest()

        assert digest(4) != digest(5)

    def test_a_different_action_changes_it(self) -> None:
        first = Digest()
        second = Digest()
        first.add(Transition(0, 0, -1.0, 1, False, False))
        second.add(Transition(0, 1, -1.0, 1, False, False))
        assert first.hexdigest() != second.hexdigest()

    def test_a_different_reward_changes_it(self) -> None:
        first = Digest()
        second = Digest()
        first.add(Transition(0, 0, -1.0, 1, False, False))
        second.add(Transition(0, 0, -1.5, 1, False, False))
        assert first.hexdigest() != second.hexdigest()

    def test_a_different_ending_changes_it(self) -> None:
        # A step limit and a real ending are different things, and a digest
        # that could not tell them apart would call two different runs the
        # same run.
        first = Digest()
        second = Digest()
        first.add(Transition(0, 0, -1.0, 1, True, False))
        second.add(Transition(0, 0, -1.0, 1, False, True))
        assert first.hexdigest() != second.hexdigest()

    def test_the_order_of_the_steps_matters(self) -> None:
        forward = Digest()
        backward = Digest()
        steps = [
            Transition(0, 0, -1.0, 1, False, False),
            Transition(1, 1, -2.0, 2, False, False),
        ]
        for step in steps:
            forward.add(step)
        for step in reversed(steps):
            backward.add(step)
        assert forward.hexdigest() != backward.hexdigest()

    def test_it_counts_the_steps_it_saw(self) -> None:
        digest = Digest()
        for _ in range(7):
            digest.add(Transition(0, 0, -1.0, 1, False, False))
        assert digest.steps == 7

    def test_an_observation_of_several_numbers_is_covered(self) -> None:
        first = Digest()
        second = Digest()
        first.add(Transition((0.5, 1.5), 0, 0.0, (0.0, 0.0), False, False))
        second.add(Transition((0.5, 1.6), 0, 0.0, (0.0, 0.0), False, False))
        assert first.hexdigest() != second.hexdigest()

    def test_a_change_far_below_the_rounding_is_not_covered(self) -> None:
        # Deliberate. The digest keeps twelve significant figures, because a C
        # library may return `cos` one bit different from another one and a
        # digest that changed with the C library would report a fault that does
        # not exist. That is a real limit and it is stated rather than hidden.
        first = Digest()
        second = Digest()
        first.add(Transition((1.0,), 0, 0.0, (0.0,), False, False))
        second.add(Transition((1.0 + 1e-15,), 0, 0.0, (0.0,), False, False))
        assert first.hexdigest() == second.hexdigest()


class TestEvaluate:
    def test_it_does_not_teach_the_agent_anything(self) -> None:
        env = cliff_walk(Rng(1).stream("env"))
        agent: QLearning[int] = QLearning(Rng(1).stream("agent"), env.action_space)
        train(env, agent, 30)
        before = {state: list(row) for state, row in agent.q.items()}

        evaluate(cliff_walk(Rng(2).stream("env")), agent, 3)
        assert {state: list(row) for state, row in agent.q.items()} == before

    def test_it_runs_the_policy_with_no_exploration(self) -> None:
        env = Corridor(Rng(1))
        agent: QLearning[int] = QLearning(Rng(1), env.action_space, epsilon=1.0)
        agent.q[0] = [-99.0, 0.0]
        agent.q[1] = [-99.0, 0.0]

        record = evaluate(env, agent, 3)
        assert record.returns == [-2.0, -2.0, -2.0]

    def test_the_episode_count_is_what_was_asked_for(self) -> None:
        record = evaluate(Corridor(Rng(1)), AlwaysRight(Rng(1), Discrete(2)), 4)
        assert len(record) == 4


class TestTheLoopCallsTheHooks:
    def test_the_start_of_an_episode_is_announced(self) -> None:
        # A bandit agent needs this: every episode of that environment is a new
        # set of levers, and an agent that carried its estimates across would
        # be answering a question about the last problem.
        class Counting(AlwaysRight):
            starts = 0
            ends = 0

            def start_episode(self) -> None:
                Counting.starts += 1

            def end_episode(self) -> None:
                Counting.ends += 1
                super().end_episode()

        Counting.starts = 0
        Counting.ends = 0
        train(Corridor(Rng(1)), Counting(Rng(1), Discrete(2)), 5)
        assert Counting.starts == 5
        assert Counting.ends == 5

    def test_the_caller_is_told_after_each_episode(self) -> None:
        seen: list[int] = []

        def watch(episode: Episode, record: Record) -> None:
            seen.append(len(record))

        train(
            Corridor(Rng(1)),
            AlwaysRight(Rng(1), Discrete(2)),
            4,
            on_episode=watch,
        )
        assert seen == [1, 2, 3, 4]


def test_the_loop_hands_the_agent_the_action_it_returned() -> None:
    """Both on-policy agents depend on it, and one of them broke over it.

    An n step agent that wrote down the action it expected rather than the one
    that was taken updated the wrong cell. That was fixed in the agents, and
    this holds the other side of the promise.
    """
    taken: list[int] = []

    class Recording(Agent[int]):
        def act(self, observation: int) -> int:
            chosen = 1 if observation < 2 else 0
            taken.append(chosen)
            return chosen

        def greedy(self, observation: int) -> int:
            return self.act(observation)

        def observe(self, transition: Transition[int]) -> None:
            assert transition.action == taken[-1]

    env: Env[int] = Corridor(Rng(1))
    run_episode(env, Recording(Rng(1), Discrete(2)))
    assert taken == [1, 1]
