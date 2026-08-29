"""The rules an agent can explore by.

Two things are held here. A rule on its own has to keep its two answers in
agreement: `probabilities` must be what `choose` really does, or expected SARSA
averages over one policy while the agent follows another.

An agent given the default rule has to run exactly as it did before there was a
rule at all. That is not a nicety. The numbers in the documentation were
measured with the old code, and a refactor that moved any of them would have
made the whole page a claim about a version that no longer exists.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rel.agents.base import TabularAgent
from rel.agents.dyna import DynaQ
from rel.agents.explore import (
    Counts,
    EpsilonGreedy,
    Rule,
    Softmax,
    argmax,
    greedy_probabilities,
)
from rel.agents.monte_carlo import MonteCarloControl
from rel.agents.td import DoubleQ, ExpectedSarsa, QLearning, Sarsa
from rel.agents.traces import SarsaLambda, WatkinsQLambda
from rel.envs.classic import cliff_walk
from rel.rng import Rng
from rel.schedules import linear
from rel.spaces import Discrete
from rel.training import digest_of, train


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


class TestSoftmax:
    def test_it_adds_up_to_one(self) -> None:
        rule = Softmax(1.0)
        assert sum(rule.probabilities([1.0, 2.0, 3.0], None, 0, 0)) == pytest.approx(
            1.0
        )

    def test_a_better_action_is_more_likely(self) -> None:
        shares = Softmax(1.0).probabilities([0.0, 1.0, 2.0], None, 0, 0)
        assert shares[0] < shares[1] < shares[2]

    def test_equal_actions_are_equally_likely(self) -> None:
        shares = Softmax(1.0).probabilities([4.0, 4.0], None, 0, 0)
        assert shares == pytest.approx([0.5, 0.5])

    def test_a_hot_temperature_is_near_uniform(self) -> None:
        shares = Softmax(1000.0).probabilities([0.0, 1.0, 2.0], None, 0, 0)
        assert shares == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=0.01)

    def test_a_cold_temperature_is_near_greedy(self) -> None:
        shares = Softmax(0.01).probabilities([0.0, 1.0, 2.0], None, 0, 0)
        assert shares == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)

    def test_zero_is_greedy_rather_than_a_division_by_zero(self) -> None:
        """A schedule that cools to nothing has to survive its last episode."""
        shares = Softmax(0.0).probabilities([0.0, 1.0, 2.0], None, 0, 0)
        assert shares == [0.0, 0.0, 1.0]

    def test_below_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="zero or above"):
            Softmax(-1.0).probabilities([0.0, 1.0], None, 0, 0)

    def test_far_apart_values_do_not_overflow(self) -> None:
        """The exponents are all zero or below, so nothing can run away."""
        shares = Softmax(0.5).probabilities([-1000.0, 1000.0], None, 0, 0)
        assert shares == pytest.approx([0.0, 1.0])

    def test_the_schedule_is_read_at_the_episode(self) -> None:
        rule = Softmax(linear(100.0, 0.01, 10))
        hot = rule.probabilities([0.0, 1.0], None, 0, 0)
        cold = rule.probabilities([0.0, 1.0], None, 10, 0)
        assert hot[1] < cold[1]

    def test_choosing_matches_the_probabilities(self) -> None:
        rule = Softmax(1.0)
        rng = Rng(11)
        scores = [0.0, 1.0, 2.0]

        picks = [rule.choose(rng, scores, None, 0, 0) for _ in range(20000)]
        for index, share in enumerate(rule.probabilities(scores, None, 0, 0)):
            assert picks.count(index) / 20000 == pytest.approx(share, abs=0.02)

    def test_it_ranks_where_epsilon_greedy_does_not(self) -> None:
        """The whole point, as a comparison.

        Epsilon-greedy gives the second best action and the worst one the same
        share. This gives the second best more.
        """
        scores = [0.0, 1.0, 5.0]
        flat = EpsilonGreedy(0.3).probabilities(scores, None, 0, 0)
        ranked = Softmax(1.0).probabilities(scores, None, 0, 0)

        assert flat[0] == pytest.approx(flat[1])
        assert ranked[1] > ranked[0]

    def test_it_says_what_it_is(self) -> None:
        assert "2" in repr(Softmax(2.0))


class Counted(Rule):
    """A rule that asks for the counts and otherwise takes the first action.

    Written here rather than imported, so that what the agent does with a rule
    that wants counting is tested apart from any particular way of using them.
    """

    needs_counts = True

    def __init__(self) -> None:
        self.seen: list[Counts] = []

    def choose(
        self,
        rng: Rng,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> int:
        self.seen.append(None if counts is None else list(counts))
        return 0

    def probabilities(
        self,
        scores: Sequence[float],
        counts: Counts,
        episodes: int,
        steps: int,
    ) -> list[float]:
        return [1.0] + [0.0] * (len(scores) - 1)


class TestTheDefaultRuleChangesNothing:
    """Every tabular agent explores as it did before there was a rule object.

    The refactor that put the rule behind an object was checked by running
    twelve agents on the cliff walk and comparing both digests against the
    commit before it. All twenty four matched. What is held here is the claim
    that survives a later change to the default: naming the rule and leaving
    it out have to be the same run, step for step and cell for cell.
    """

    AGENTS = (
        QLearning,
        Sarsa,
        ExpectedSarsa,
        DoubleQ,
        MonteCarloControl,
        DynaQ,
        SarsaLambda,
        WatkinsQLambda,
    )

    def _run(self, cls: type[TabularAgent[int]], epsilon: float) -> tuple[str, str]:
        root = Rng(7)
        env = cliff_walk(root.stream("env"))
        agent = cls(root.stream("agent"), env.action_space, epsilon=epsilon)
        record = train(env, agent, 30)
        return record.digest.hexdigest(), digest_of(agent)

    def _named(self, cls: type[TabularAgent[int]], rate: float) -> tuple[str, str]:
        root = Rng(7)
        env = cliff_walk(root.stream("env"))
        agent = cls(root.stream("agent"), env.action_space, explore=EpsilonGreedy(rate))
        record = train(env, agent, 30)
        return record.digest.hexdigest(), digest_of(agent)

    @pytest.mark.parametrize("cls", AGENTS)
    def test_naming_the_default_is_the_same_run(
        self, cls: type[TabularAgent[int]]
    ) -> None:
        assert self._run(cls, 0.1) == self._named(cls, 0.1)

    @pytest.mark.parametrize("cls", AGENTS)
    def test_a_different_rate_is_a_different_run(
        self, cls: type[TabularAgent[int]]
    ) -> None:
        """The check above would pass if `explore` were being ignored."""
        assert self._run(cls, 0.1) != self._named(cls, 0.4)


class TestTheCountsTable:
    def test_it_is_not_kept_for_a_rule_that_does_not_ask(self) -> None:
        agent = QLearning(Rng(1), Discrete(4))
        assert agent.counts is None
        assert agent.action_counts(0) is None

    def test_it_is_kept_for_a_rule_that_does(self) -> None:
        agent = QLearning(Rng(1), Discrete(4), explore=Counted())
        assert agent.counts == {}

    def test_reading_it_makes_no_row(self) -> None:
        """The same rule `peek` follows.

        A count read for a state the agent has never been in must not put that
        state in the table, or the value map reports it has stood there.
        """
        agent = QLearning(Rng(1), Discrete(4), explore=Counted())
        assert list(agent.action_counts(3) or []) == [0, 0, 0, 0]
        assert agent.counts == {}

    def test_acting_counts_the_action_it_took(self) -> None:
        agent = QLearning(Rng(1), Discrete(4), explore=Counted())
        agent.act(3)
        agent.act(3)
        assert agent.counts == {3: [2, 0, 0, 0]}

    def test_the_rule_sees_the_count_before_the_action_it_is_choosing(self) -> None:
        """A rule choosing its nth visit is told n-1, not n.

        A bonus that shrinks with the count would be shrunk by the visit it is
        deciding, which is the visit that has not happened yet.
        """
        rule = Counted()
        agent = QLearning(Rng(1), Discrete(2), explore=rule)
        agent.act(0)
        agent.act(0)
        assert rule.seen == [[0, 0], [1, 0]]
