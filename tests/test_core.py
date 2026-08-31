"""Tests for the environment contract.

Every environment in the project inherits this behaviour, so these tests use a
counter environment written here rather than a real one. A test that drove the
cliff walk to check truncation would fail the day somebody changed the size of
the cliff.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from rel.core import DiscreteEnv, Env, EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete


class Counter(DiscreteEnv[int]):
    """Counts up. Action 1 ends the episode, action 0 does not."""

    def __init__(self, rng: Rng, limit: int | None = None) -> None:
        super().__init__(rng)
        self.observation_space = Discrete(1000)
        self.action_space = Discrete(2)
        self.spec = EnvSpec(
            name="counter",
            summary="Counts up until told to stop.",
            max_episode_steps=limit,
        )
        self.at = 0
        self.stops = 0

    def _reset(self) -> int:
        self.at = 0
        return 0

    def _step(self, action: int) -> Step[int]:
        self.at += 1
        if action == 1:
            self.stops += 1
            return Step(self.at, 10.0, terminated=True, truncated=False)
        return Step(self.at, 1.0, terminated=False, truncated=False)

    def audit(self) -> Mapping[str, float]:
        return {"stops": float(self.stops)}


class TestStep:
    def test_done_covers_both_endings(self) -> None:
        assert Step(0, 0.0, terminated=True, truncated=False).done
        assert Step(0, 0.0, terminated=False, truncated=True).done
        assert not Step(0, 0.0, terminated=False, truncated=False).done

    def test_info_is_empty_by_default(self) -> None:
        assert Step(0, 0.0, terminated=False, truncated=False).info == {}


class TestTheBaseClass:
    def test_stepping_before_reset_is_refused(self) -> None:
        env = Counter(Rng(1))
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_stepping_after_the_episode_ended_is_refused(self) -> None:
        env = Counter(Rng(1))
        env.reset()
        env.step(1)
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_an_action_outside_the_space_is_refused(self) -> None:
        env = Counter(Rng(1))
        env.reset()
        for bad in (-1, 2, 99):
            with pytest.raises(ValueError):
                env.step(bad)

    def test_a_bool_is_refused_as_an_action(self) -> None:
        # `True == 1` in Python, and action 1 ends this episode. An unguarded
        # environment would take the flag, end the episode and report nothing.
        env = Counter(Rng(1))
        env.reset()
        with pytest.raises(ValueError):
            env.step(True)  # type: ignore[arg-type]

    def test_elapsed_counts_the_steps_of_this_episode(self) -> None:
        env = Counter(Rng(1))
        env.reset()
        assert env.elapsed == 0
        env.step(0)
        env.step(0)
        assert env.elapsed == 2
        env.reset()
        assert env.elapsed == 0


class TestTruncation:
    def test_the_limit_truncates_rather_than_terminates(self) -> None:
        # The difference decides whether an agent bootstraps through the end of
        # the episode. Reporting a limit as a termination teaches the agent
        # that the state it was stopped in is worthless.
        env = Counter(Rng(1), limit=3)
        env.reset()
        assert not env.step(0).done
        assert not env.step(0).done

        last = env.step(0)
        assert last.truncated
        assert not last.terminated

    def test_a_real_ending_is_not_reported_as_a_limit(self) -> None:
        env = Counter(Rng(1), limit=3)
        env.reset()
        env.step(0)
        ending = env.step(1)
        assert ending.terminated
        assert not ending.truncated

    def test_a_reward_survives_truncation(self) -> None:
        # The step that hits the limit still did something, and its reward has
        # to be paid. Rebuilding the step to set the flag is the place where
        # that reward is easy to drop.
        env = Counter(Rng(1), limit=1)
        env.reset()
        assert env.step(0).reward == 1.0

    def test_no_limit_means_no_truncation(self) -> None:
        env = Counter(Rng(1))
        env.reset()
        for _ in range(500):
            assert not env.step(0).done


class TestAudit:
    def test_the_default_reports_nothing(self) -> None:
        class Bare(Counter):
            pass

        env = Bare(Rng(1))
        assert Env.audit(env) == {}

    def test_an_environment_can_report_what_the_reward_did_not_pay_for(
        self,
    ) -> None:
        env = Counter(Rng(1))
        env.reset()
        env.step(1)
        assert env.audit() == {"stops": 1.0}


def test_rendering_is_optional() -> None:
    env = Counter(Rng(1))
    with pytest.raises(NotImplementedError):
        env.render()


class Chain(TabularEnv):
    """Three states in a line. State 2 is terminal."""

    def __init__(self, rng: Rng) -> None:
        super().__init__(rng)
        self.observation_space = Discrete(3)
        self.action_space = Discrete(2)
        self.spec = EnvSpec(name="chain", summary="Three states in a line.")
        self.at = 0

    def _reset(self) -> int:
        self.at = 0
        return 0

    def _step(self, action: int) -> Step[int]:
        outcome = self.transitions(self.at, action)[0]
        self.at = outcome.observation
        return Step(outcome.observation, outcome.reward, outcome.terminated, False)

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if state == 2:
            return [Outcome(1.0, 2, 0.0, terminated=True)]
        target = min(state + 1, 2) if action == 1 else max(state - 1, 0)
        return [Outcome(1.0, target, 1.0 if target == 2 else 0.0, target == 2)]

    def start_states(self) -> Sequence[tuple[float, int]]:
        return [(1.0, 0)]


class TestTabularEnv:
    def test_terminal_states_come_out_of_the_model(self) -> None:
        assert Chain(Rng(1)).terminal_states() == frozenset({2})

    def test_the_probabilities_of_a_transition_add_up_to_one(self) -> None:
        env = Chain(Rng(1))
        for state in env.observation_space:
            for action in env.action_space:
                branches = env.transitions(state, action)
                total = sum(branch.probability for branch in branches)
                assert abs(total - 1.0) < 1e-12
