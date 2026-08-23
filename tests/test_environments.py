"""One suite that every registered environment has to pass.

A test written against one environment covers one environment. These ask the
same questions of all of them, so an environment added next month is covered
the moment its entry lands in the table.

The questions are the ones an agent depends on and a docstring cannot promise:
that the observation is really inside the space that was declared, that every
action is accepted, that a seed replays the run, and that the model, where
there is one, adds up.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rel.core import Env, TabularEnv
from rel.envs import ENVIRONMENTS
from rel.rng import Rng
from rel.spaces import Box, Discrete

NAMES = ENVIRONMENTS.names()
KNOWN_TAGS = frozenset({"bandit", "continuous", "gaming", "grid", "tabular"})


def build(name: str, seed: int = 1) -> Env[Any]:
    return ENVIRONMENTS.make(name, Rng(seed).stream("env"))


def walk(env: Env[Any], policy: Rng, steps: int) -> list[tuple[Any, float]]:
    """Drive the environment with a random policy and record what came back."""
    path: list[tuple[Any, float]] = []
    env.reset()
    for _ in range(steps):
        outcome = env.step(policy.below(env.action_space.n))
        path.append((outcome.observation, outcome.reward))
        if outcome.done:
            env.reset()
    return path


class TestTheTable:
    def test_every_name_is_lower_case_with_hyphens(self) -> None:
        for name in NAMES:
            assert name == name.lower()
            assert " " not in name
            assert "_" not in name

    def test_every_summary_is_a_sentence(self) -> None:
        # The summary is printed by --list and read straight into the
        # documentation. A fragment reads badly in both places.
        for entry in ENVIRONMENTS:
            assert entry.summary.endswith("."), entry.name
            assert entry.summary[0].isupper(), entry.name

    def test_every_tag_is_one_this_project_uses(self) -> None:
        for entry in ENVIRONMENTS:
            assert set(entry.tags) <= KNOWN_TAGS, entry.name

    def test_every_environment_carries_a_tag(self) -> None:
        for entry in ENVIRONMENTS:
            assert entry.tags, entry.name

    def test_the_table_is_not_empty(self) -> None:
        assert len(ENVIRONMENTS) >= 9


@pytest.mark.parametrize("name", NAMES)
class TestEveryEnvironment:
    def test_it_builds_from_a_seeded_stream(self, name: str) -> None:
        env = build(name)
        assert env.spec.name
        assert env.action_space.n >= 2

    def test_the_first_observation_is_inside_the_space(self, name: str) -> None:
        env = build(name)
        assert env.observation_space.contains(env.reset())

    def test_every_observation_is_inside_the_space(self, name: str) -> None:
        env = build(name)
        policy = Rng(1).stream("policy")
        for observation, _ in walk(env, policy, 2000):
            assert env.observation_space.contains(observation), (
                f"{name} returned {observation!r}, "
                f"which is outside {env.observation_space!r}"
            )

    def test_every_reward_is_a_finite_number(self, name: str) -> None:
        env = build(name)
        policy = Rng(2).stream("policy")
        for _, reward in walk(env, policy, 2000):
            assert isinstance(reward, float)
            assert math.isfinite(reward)

    def test_every_action_is_accepted(self, name: str) -> None:
        env = build(name)
        for action in env.action_space:
            env.reset()
            env.step(action)

    def test_the_summary_on_the_spec_matches_the_table(self, name: str) -> None:
        # Two places say what an environment is. They are allowed to differ in
        # wording, and they are not allowed to describe different things, so
        # this checks that both were written rather than one copied and left.
        env = build(name)
        assert env.spec.summary.endswith(".")

    def test_it_renders_without_a_terminal(self, name: str) -> None:
        env = build(name)
        env.reset()
        drawn = env.render()
        assert isinstance(drawn, str)
        assert drawn.strip()

    def test_the_audit_reports_numbers_only(self, name: str) -> None:
        # Everything on this channel is averaged over episodes by the recorder.
        # A value that is not a number cannot be averaged, and it would fail in
        # the report rather than here.
        env = build(name)
        policy = Rng(3).stream("policy")
        walk(env, policy, 500)
        for key, value in env.audit().items():
            assert isinstance(key, str)
            assert isinstance(value, float), f"{name}: {key} is {type(value)}"
            assert math.isfinite(value)

    def test_a_seed_replays_the_run(self, name: str) -> None:
        assert walk(build(name, 7), Rng(7).stream("policy"), 500) == walk(
            build(name, 7), Rng(7).stream("policy"), 500
        )

    def test_two_seeds_do_not_give_the_same_run(self, name: str) -> None:
        assert walk(build(name, 7), Rng(7).stream("policy"), 500) != walk(
            build(name, 8), Rng(8).stream("policy"), 500
        )

    def test_an_episode_can_be_started_again(self, name: str) -> None:
        env = build(name)
        env.reset()
        env.step(0)
        env.reset()
        env.step(0)


@pytest.mark.parametrize(
    "name", [entry.name for entry in ENVIRONMENTS.tagged("tabular")]
)
class TestEveryTabularEnvironment:
    def test_it_really_is_tabular(self, name: str) -> None:
        env = build(name)
        assert isinstance(env, TabularEnv)
        assert isinstance(env.observation_space, Discrete)

    def test_the_model_adds_up_for_every_pair(self, name: str) -> None:
        env = build(name)
        assert isinstance(env, TabularEnv)
        for state in env.observation_space:
            for action in env.action_space:
                total = sum(
                    outcome.probability for outcome in env.transitions(state, action)
                )
                assert abs(total - 1.0) < 1e-9, f"{name} {state} {action}"

    def test_no_branch_has_a_negative_probability(self, name: str) -> None:
        env = build(name)
        assert isinstance(env, TabularEnv)
        for state in env.observation_space:
            for action in env.action_space:
                for outcome in env.transitions(state, action):
                    assert outcome.probability >= 0.0

    def test_the_start_distribution_adds_up(self, name: str) -> None:
        env = build(name)
        assert isinstance(env, TabularEnv)
        starts = env.start_states()
        assert starts
        assert abs(sum(share for share, _ in starts) - 1.0) < 1e-9
        for _, state in starts:
            assert env.observation_space.contains(state)

    def test_a_reset_lands_on_a_start_state(self, name: str) -> None:
        env = build(name)
        assert isinstance(env, TabularEnv)
        allowed = {state for _, state in env.start_states()}
        for _ in range(50):
            assert env.reset() in allowed


@pytest.mark.parametrize(
    "name", [entry.name for entry in ENVIRONMENTS.tagged("continuous")]
)
class TestEveryContinuousEnvironment:
    def test_the_observation_space_is_a_box(self, name: str) -> None:
        assert isinstance(build(name).observation_space, Box)

    def test_it_offers_a_tiling_box_inside_the_observation_box(self, name: str) -> None:
        # A tile coder needs a range to divide, and the range that makes good
        # tiles is not always the range the environment can reach. Naming both
        # keeps a divisor out of the agent.
        env = build(name)
        tiling = env.tiling_space  # type: ignore[attr-defined]
        assert isinstance(tiling, Box)
        assert isinstance(env.observation_space, Box)
        assert tiling.dimensions == env.observation_space.dimensions
        for index in range(tiling.dimensions):
            assert env.observation_space.low[index] <= tiling.low[index]
            assert tiling.high[index] <= env.observation_space.high[index]
