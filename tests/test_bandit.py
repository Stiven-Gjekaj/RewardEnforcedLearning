"""Tests for the row of levers."""

from __future__ import annotations

import pytest

from rel.envs.bandit import KArmedBandit, drifting_bandit, stationary_bandit
from rel.rng import Rng


class TestSetup:
    def test_one_lever_is_not_a_bandit(self) -> None:
        with pytest.raises(ValueError, match="at least two levers"):
            KArmedBandit(Rng(1), arms=1)

    def test_negative_noise_is_refused(self) -> None:
        with pytest.raises(ValueError, match="noise"):
            KArmedBandit(Rng(1), noise=-1.0)

    def test_there_is_one_state(self) -> None:
        bandit = KArmedBandit(Rng(1))
        assert bandit.observation_space.n == 1
        assert bandit.reset() == 0

    def test_the_action_space_is_the_levers(self) -> None:
        assert KArmedBandit(Rng(1), arms=7).action_space.n == 7


class TestPayouts:
    def test_a_lever_pays_around_its_true_value(self) -> None:
        bandit = KArmedBandit(Rng(1), arms=3, noise=1.0, pulls=100_000)
        bandit.reset()
        pulls = 20_000
        total = sum(bandit.step(0).reward for _ in range(pulls))
        assert abs(total / pulls - bandit.values[0]) < 0.03

    def test_no_noise_pays_the_true_value_exactly(self) -> None:
        bandit = KArmedBandit(Rng(1), arms=3, noise=0.0)
        bandit.reset()
        assert bandit.step(1).reward == bandit.values[1]

    def test_a_new_episode_is_a_new_problem(self) -> None:
        # A run of this environment averages over problems. Keeping the same
        # true values from one episode to the next would turn a hundred
        # episodes into one problem seen a hundred times, and the curve would
        # say far more than the data supports.
        bandit = KArmedBandit(Rng(2))
        bandit.reset()
        first = list(bandit.values)
        bandit.reset()
        assert bandit.values != first

    def test_the_values_stay_put_inside_an_episode(self) -> None:
        bandit = KArmedBandit(Rng(3), walk=0.0)
        bandit.reset()
        before = list(bandit.values)
        for arm in range(bandit.arms):
            bandit.step(arm)
        assert bandit.values == before


class TestDrift:
    def test_every_lever_starts_equal(self) -> None:
        # Exercise 2.5 starts them level so that the only thing separating two
        # agents is how each one follows a target that moves.
        bandit = drifting_bandit(Rng(4))
        bandit.reset()
        assert bandit.values == [0.0] * bandit.arms

    def test_the_levers_wander(self) -> None:
        bandit = drifting_bandit(Rng(5))
        bandit.reset()
        for _ in range(50):
            bandit.step(0)
        assert all(value != 0.0 for value in bandit.values)

    def test_the_walk_happens_after_the_pull(self) -> None:
        # The reward paid belongs to the values the agent was choosing between.
        # Walking first pays for a lever that did not exist when the choice was
        # made, which makes an agent look worse than it is by exactly the size
        # of one step of the walk.
        bandit = KArmedBandit(Rng(6), arms=2, noise=0.0, walk=0.5)
        bandit.reset()
        expected = bandit.values[0]
        assert bandit.step(0).reward == expected


class TestTheAudit:
    def test_choosing_the_best_lever_every_time_reads_as_one(self) -> None:
        bandit = KArmedBandit(Rng(7), arms=5, noise=1.0)
        bandit.reset()
        for _ in range(100):
            bandit.step(bandit.best_arm())
        assert bandit.audit()["best_arm_share"] == 1.0
        assert bandit.audit()["regret"] == 0.0

    def test_choosing_the_worst_lever_costs_the_gap_each_time(self) -> None:
        bandit = KArmedBandit(Rng(8), arms=5, noise=0.0)
        bandit.reset()
        worst = min(range(bandit.arms), key=lambda arm: bandit.values[arm])
        gap = bandit.best_value() - bandit.values[worst]
        for _ in range(10):
            bandit.step(worst)
        assert abs(bandit.audit()["regret"] - 10 * gap) < 1e-9
        assert bandit.audit()["best_arm_share"] == 0.0

    def test_the_share_is_of_the_pulls_so_far(self) -> None:
        bandit = KArmedBandit(Rng(9), arms=4, noise=0.0)
        bandit.reset()
        best = bandit.best_arm()
        worst = min(range(bandit.arms), key=lambda arm: bandit.values[arm])
        bandit.step(best)
        bandit.step(worst)
        assert bandit.audit()["best_arm_share"] == 0.5

    def test_it_reports_zero_before_anything_is_pulled(self) -> None:
        bandit = KArmedBandit(Rng(10))
        bandit.reset()
        assert bandit.audit() == {"best_arm_share": 0.0, "regret": 0.0}

    def test_the_audit_starts_again_with_the_episode(self) -> None:
        bandit = KArmedBandit(Rng(11), arms=3, noise=0.0)
        bandit.reset()
        bandit.step(min(range(3), key=lambda arm: bandit.values[arm]))
        bandit.reset()
        assert bandit.audit()["regret"] == 0.0


class TestTheEpisode:
    def test_the_episode_lasts_for_the_number_of_pulls_asked_for(self) -> None:
        bandit = KArmedBandit(Rng(12), pulls=25)
        bandit.reset()
        for _ in range(24):
            assert not bandit.step(0).done
        last = bandit.step(0)
        assert last.truncated
        assert not last.terminated

    def test_nothing_ends_the_episode_from_inside(self) -> None:
        bandit = KArmedBandit(Rng(13), pulls=1000)
        bandit.reset()
        assert not any(bandit.step(0).terminated for _ in range(999))


class TestRendering:
    def test_the_best_lever_is_marked(self) -> None:
        bandit = stationary_bandit(Rng(14))
        bandit.reset()
        lines = bandit.render().splitlines()
        assert len(lines) == bandit.arms
        assert sum(line.startswith("*") for line in lines) == 1
        assert lines[bandit.best_arm()].startswith("*")


def test_the_same_seed_gives_the_same_problem() -> None:
    def values(seed: int) -> list[float]:
        bandit = stationary_bandit(Rng(seed).stream("env"))
        bandit.reset()
        return list(bandit.values)

    assert values(15) == values(15)
    assert values(15) != values(16)
