"""Tests for the loop and for what it writes down.

The digest is what the whole reproducibility claim rests on. If two runs that
took different paths could produce the same digest, every comparison made with
one would be worthless, and nothing else in the suite would notice.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from functools import cache

import pytest

from rel.agents import AGENTS
from rel.agents.base import Agent, RandomAgent, Transition
from rel.agents.td import DoubleQ, ExpectedSarsa, QLearning
from rel.core import Env, EnvSpec, Outcome, Step, TabularEnv
from rel.envs import ENVIRONMENTS
from rel.envs.classic import cliff_walk
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import (
    Digest,
    Episode,
    Record,
    digest_line,
    digest_of,
    evaluate,
    run_episode,
    train,
)


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


class AlwaysRight(Agent[int, int]):
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

    class Recording(Agent[int, int]):
        def act(self, observation: int) -> int:
            chosen = 1 if observation < 2 else 0
            taken.append(chosen)
            return chosen

        def greedy(self, observation: int) -> int:
            return self.act(observation)

        def observe(self, transition: Transition[int, int]) -> None:
            assert transition.action == taken[-1]

    env: Env[int, int] = Corridor(Rng(1))
    run_episode(env, Recording(Rng(1), Discrete(2)))
    assert taken == [1, 1]


class TestOneWayToSpellATransition:
    """The line a digest hashes is built in one place.

    `rel.recording` writes a file whose step lines come from the same pieces,
    and reads them back without the transition they came from. Two ways of
    spelling one transition would be two digests, and a recording would be
    checkable against only one of them.
    """

    def test_a_line_holds_the_four_things_a_digest_covers(self) -> None:
        line = digest_line(Transition(3, 1, -1.0, 4, terminated=True, truncated=False))
        assert line == "3|1|-1|10\n"

    def test_a_tuple_observation_is_written_out_in_order(self) -> None:
        line = digest_line(
            Transition(
                (0.5, -1.25), 0, 2.0, (0.0, 0.0), terminated=False, truncated=True
            )
        )
        assert line == "0.5,-1.25|0|2|01\n"

    def test_adding_a_line_and_adding_a_transition_agree(self) -> None:
        step = Transition(7, 2, 0.25, 8, terminated=False, truncated=False)

        one = Digest()
        one.add(step)
        other = Digest()
        other.add_line(digest_line(step))

        assert other.hexdigest() == one.hexdigest()
        assert other.steps == one.steps


class TestTheDigestOfWhatAnAgentLearned:
    """The second digest, kept beside the first rather than merged into it.

    The path digest hashes the transitions, so two agents that happen to walk
    the same path get the same one. That answers "did these two runs do the
    same thing" and not "did these two agents come to the same conclusion".
    """

    @staticmethod
    def _fed(agent: Agent[int, int]) -> Agent[int, int]:
        for step in (
            Transition(0, 0, -1.0, 1, terminated=False, truncated=False),
            Transition(1, 1, -1.0, 2, terminated=False, truncated=False),
            Transition(2, 0, 10.0, 3, terminated=True, truncated=False),
        ):
            agent.observe(step)
        agent.end_episode()
        return agent

    def test_two_agents_on_one_path_give_different_digests(self) -> None:
        # The whole reason for having it. Q-learning takes the largest value
        # at the next state and expected SARSA takes the average over its
        # policy, so one step is enough to separate them, and the path they
        # walked is the same path.
        step = Transition(0, 0, -1.0, 1, terminated=False, truncated=False)

        one: QLearning[int] = QLearning(
            Rng(1), Discrete(2), step_size=0.5, discount=0.9, epsilon=0.1
        )
        other: ExpectedSarsa[int] = ExpectedSarsa(
            Rng(1), Discrete(2), step_size=0.5, discount=0.9, epsilon=0.1
        )
        for agent in (one, other):
            agent.values(1)[:] = [0.0, 10.0]
            agent.observe(step)

        walk = Digest()
        walk.add(step)
        before = walk.hexdigest()
        walk_again = Digest()
        walk_again.add(step)

        assert digest_of(one) != digest_of(other)
        # Both walked the same path, and the path digest cannot say so.
        assert walk_again.hexdigest() == before

    def test_two_agents_that_learned_the_same_thing_agree(self) -> None:
        one = self._fed(QLearning(Rng(1), Discrete(2), step_size=0.5, discount=0.9))
        other = self._fed(QLearning(Rng(9), Discrete(2), step_size=0.5, discount=0.9))
        assert digest_of(one) == digest_of(other)

    def test_an_agent_that_learns_nothing_says_nothing(self) -> None:
        # Not the hash of nothing. That is the same for every agent that keeps
        # nothing and would read as a fact about the agent.
        assert digest_of(RandomAgent(Rng(1), Discrete(2))) is None

    def test_the_order_the_states_were_seen_in_does_not_change_it(self) -> None:
        one: QLearning[int] = QLearning(Rng(1), Discrete(2), step_size=1.0)
        one.values(9)[0] = 1.0
        one.values(2)[1] = 2.0

        other: QLearning[int] = QLearning(Rng(1), Discrete(2), step_size=1.0)
        other.values(2)[1] = 2.0
        other.values(9)[0] = 1.0

        assert digest_of(one) == digest_of(other)

    def test_it_covers_both_of_a_double_q_agent_s_tables(self) -> None:
        # Either one alone would call two agents that split their coin
        # differently the same agent.
        one: DoubleQ[int] = DoubleQ(Rng(1), Discrete(2))
        one.values(0)[0] = 1.0
        other: DoubleQ[int] = DoubleQ(Rng(1), Discrete(2))
        other.other_values(0)[0] = 1.0
        assert digest_of(one) != digest_of(other)

    def test_the_path_digest_is_unchanged_by_any_of_this(self) -> None:
        # Every number in the documentation was compared against this one. It
        # is pinned to a value written down before the second digest existed.
        rng = Rng(1)
        env = cliff_walk(rng.stream("env"))
        agent: QLearning[int] = QLearning(
            rng.stream("agent"), env.action_space, step_size=0.5, epsilon=0.1
        )
        record = train(env, agent, 50)
        assert record.digest.hexdigest() == "0de6831e401c7ddd"


@cache
def digests(name: str, grid: str, episodes: int) -> tuple[str, str | None]:
    """One run's two digests, kept so that two tests cost one run."""
    root = Rng(1)
    env = ENVIRONMENTS.make(grid, root.stream("env"))
    agent = AGENTS.make(name, root.stream("agent"), env)
    record = train(env, agent, episodes, discount=env.spec.suggested_discount)
    return record.digest.hexdigest(), digest_of(agent)


class TestWhichDigestsMoveBetweenPythons:
    """The learned digest is not the same number on every interpreter.

    CPython 3.12 gave `sum()` compensated summation over floats. That is more
    accurate than adding them up one at a time and it is therefore a different
    answer, so an agent whose value is a long sum can learn a table that
    differs in its last bits from one interpreter to the next.

    What checks this is the version matrix in CI rather than anything the
    tests below do. The same numbers run on 3.11, 3.12 and 3.13, so an agent
    written here as settled that starts to move fails on one of the three,
    and an agent written here as moving that stops fails on the others.

    The docstring of `digest_of` said four agents were settled and one of them
    was `reinforce`, which moves after fifty episodes. Nothing held the claim,
    so nothing said it had stopped being true.
    """

    #: The agent, the grid, the episodes, the path digest, and every learned
    #: digest any interpreter gives. One where they agree, two where 3.11
    #: disagrees with 3.12 and above.
    RUNS = (
        (
            "tile-sarsa",
            "cartpole",
            20,
            "979877443e6c00a3",
            ("54f776207a734d90", "5c5e9c0f6fd6d467"),
        ),
        (
            "reinforce",
            "cliff",
            50,
            "2193ddacf7ade68a",
            ("767e0edcd040a76a", "8347fbc29e4eb199"),
        ),
        ("expected-sarsa", "cliff", 100, "c3eb8cdd36d792b0", ("d069bcadd897b1c5",)),
        ("tree-backup", "cliff", 100, "4e770664ec3b44c2", ("de9e528a3f24e728",)),
        ("off-policy-mc", "cliff", 100, "e093b9be677f4db3", ("53c21c59bb55d97f",)),
        ("q-learning", "cliff", 100, "6a1dfba7a7a9193d", ("4b61dd694663d8df",)),
    )

    @pytest.mark.parametrize(("name", "grid", "episodes", "path", "learned"), RUNS)
    def test_the_path_digest_is_the_same_on_every_interpreter(
        self, name: str, grid: str, episodes: int, path: str, learned: tuple[str, ...]
    ) -> None:
        """One value, not a pair, for every agent here.

        A transition is an observation, an action and a reward written to a
        fixed number of figures, and those survive a difference in the last
        bits of a value, because a policy is a comparison between values
        rather than a value. That is the reason the two digests are kept
        apart, and it is checked rather than argued.
        """
        assert digests(name, grid, episodes)[0] == path

    @pytest.mark.parametrize(("name", "grid", "episodes", "path", "learned"), RUNS)
    def test_the_learned_digest_is_the_one_for_this_interpreter(
        self, name: str, grid: str, episodes: int, path: str, learned: tuple[str, ...]
    ) -> None:
        # The exact number rather than either of two, so that an agent which
        # started moving fails here and not only on another job of the
        # matrix. A settled agent has the same number written twice.
        wanted = learned[0] if sys.version_info < (3, 12) else learned[-1]
        assert digests(name, grid, episodes)[1] == wanted

    def test_something_moves_and_something_does_not(self) -> None:
        # Otherwise a table that had quietly become all one thing or all the
        # other would still pass every row above, and the two claims the
        # class is written to make would have nothing behind them.
        sizes = {len(row[4]) for row in self.RUNS}
        assert sizes == {1, 2}
        for row in self.RUNS:
            assert len(set(row[4])) == len(row[4]), row[0]
