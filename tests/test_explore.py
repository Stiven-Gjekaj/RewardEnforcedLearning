"""The rules an agent can explore by, on their own.

The agents that use them are tested elsewhere. What is held here is what a rule
promises whatever uses it: that `probabilities` is what `choose` really does,
and that `EpsilonGreedy` spends the draws the old code spent.
"""

from __future__ import annotations

import pytest

from rel.agents.explore import (
    EpsilonGreedy,
    argmax,
    greedy_probabilities,
)
from rel.rng import Rng
from rel.schedules import linear


class TestArgmax:
    def test_it_finds_the_best(self) -> None:
        assert argmax(Rng(1), [0.0, 3.0, 1.0]) == 1

    def test_a_lone_best_costs_no_draw(self) -> None:
        """The tie-break is the only reason to draw, and there is no tie.

        This is why the agents can read a policy without disturbing the run
        that produced it.
        """
        rng = Rng(1)
        before = rng.snapshot()
        argmax(rng, [0.0, 3.0, 1.0])
        assert rng.snapshot() == before

    def test_a_tie_is_broken_at_random(self) -> None:
        rng = Rng(4)
        seen = {argmax(rng, [1.0, 1.0, 1.0]) for _ in range(200)}
        assert seen == {0, 1, 2}

    def test_a_tie_is_broken_evenly(self) -> None:
        rng = Rng(4)
        picks = [argmax(rng, [1.0, 1.0]) for _ in range(2000)]
        assert 900 < picks.count(0) < 1100


class TestGreedyProbabilities:
    def test_a_lone_best_takes_all_of_it(self) -> None:
        assert greedy_probabilities([0.0, 3.0, 1.0]) == [0.0, 1.0, 0.0]

    def test_ties_share_it(self) -> None:
        assert greedy_probabilities([2.0, 2.0, 1.0]) == [0.5, 0.5, 0.0]

    def test_it_adds_up_to_one(self) -> None:
        assert sum(greedy_probabilities([1.0, 1.0, 1.0])) == pytest.approx(1.0)


class TestEpsilonGreedy:
    def test_it_adds_up_to_one(self) -> None:
        rule = EpsilonGreedy(0.1)
        shares = rule.probabilities([1.0, 2.0, 3.0, 4.0], None, 0, 0)
        assert sum(shares) == pytest.approx(1.0)

    def test_the_best_action_takes_most_of_it(self) -> None:
        rule = EpsilonGreedy(0.2)
        shares = rule.probabilities([1.0, 2.0], None, 0, 0)
        assert shares == pytest.approx([0.1, 0.9])

    def test_at_zero_it_is_greedy(self) -> None:
        rule = EpsilonGreedy(0.0)
        assert rule.probabilities([1.0, 2.0], None, 0, 0) == pytest.approx([0.0, 1.0])

    def test_at_one_it_is_uniform(self) -> None:
        rule = EpsilonGreedy(1.0)
        assert rule.probabilities([1.0, 2.0], None, 0, 0) == pytest.approx([0.5, 0.5])

    def test_the_schedule_is_read_at_the_episode(self) -> None:
        rule = EpsilonGreedy(linear(1.0, 0.0, 10))
        assert rule.probabilities([1.0, 2.0], None, 0, 0) == pytest.approx([0.5, 0.5])
        assert rule.probabilities([1.0, 2.0], None, 10, 0) == pytest.approx([0.0, 1.0])

    def test_choosing_matches_the_probabilities(self) -> None:
        """The two questions have to have the same answer.

        `probabilities` is what expected SARSA averages over and what an
        off-policy correction divides by. A rule whose two answers disagreed
        would put a bias in both, and neither would report it.
        """
        rule = EpsilonGreedy(0.3)
        rng = Rng(7)
        scores = [0.0, 1.0, 0.0, 0.0]

        picks = [rule.choose(rng, scores, None, 0, 0) for _ in range(20000)]
        wanted = rule.probabilities(scores, None, 0, 0)
        for index, share in enumerate(wanted):
            assert picks.count(index) / 20000 == pytest.approx(share, abs=0.02)

    def test_it_spends_the_draws_the_old_rule_spent(self) -> None:
        """One `chance`, then one `below` only if it explores or ties.

        The seed reaches every part of a run in this project, so a rule that
        spent a draw the old one did not would move every measured number
        while computing the same policy. This holds it to the old count.
        """
        rule = EpsilonGreedy(0.0)

        rng = Rng(3)
        watch = Rng(3)
        rule.choose(rng, [0.0, 1.0], None, 0, 0)
        watch.chance(0.0)
        assert rng.snapshot() == watch.snapshot()

        rng = Rng(3)
        watch = Rng(3)
        rule.choose(rng, [1.0, 1.0], None, 0, 0)
        watch.chance(0.0)
        watch.below(2)
        assert rng.snapshot() == watch.snapshot()

    def test_it_says_what_it_is(self) -> None:
        assert "0.1" in repr(EpsilonGreedy(0.1))
