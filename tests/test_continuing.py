"""Tests for the task with no ending and two ways to be paid.

The one that matters is the last class. The crossover this environment computes
by bisection has to be the discount at which `value_iteration` really changes
its mind, because the whole point of the environment is that claim. Two ways of
arriving at the same number, and if they disagreed the arithmetic in the
docstring would be a story rather than a fact.
"""

from __future__ import annotations

import pytest

from rel.agents.dp import value_iteration
from rel.envs.continuing import LONG, SHORT, TwoLoops, two_loops
from rel.rng import Rng


class TestTheSettingsAreChecked:
    def test_the_long_loop_takes_at_least_two_steps(self) -> None:
        with pytest.raises(ValueError, match="at least two steps"):
            TwoLoops(Rng(1), length=1)

    def test_both_loops_pay_something(self) -> None:
        with pytest.raises(ValueError, match="above zero"):
            TwoLoops(Rng(1), short_pay=0.0)
        with pytest.raises(ValueError, match="above zero"):
            TwoLoops(Rng(1), long_pay=-1.0)


class TestWhatEachLoopPays:
    def test_the_short_loop_pays_every_step(self) -> None:
        env = two_loops(Rng(1))
        env.reset()
        for _ in range(5):
            assert env.step(SHORT).reward == pytest.approx(1.0)

    def test_the_long_loop_pays_only_on_the_way_back_in(self) -> None:
        env = two_loops(Rng(1))
        env.reset()
        paid = [env.step(LONG).reward for _ in range(5)]
        assert paid == pytest.approx([0.0, 0.0, 0.0, 0.0, 10.0])

    def test_the_long_loop_comes_back_to_the_junction(self) -> None:
        env = two_loops(Rng(1))
        env.reset()
        seen = [env.step(LONG).observation for _ in range(5)]
        assert seen == [1, 2, 3, 4, 0]

    def test_the_action_only_matters_at_the_junction(self) -> None:
        """Once on the long way round there is nothing to decide."""
        env = two_loops(Rng(1))
        for state in range(1, 5):
            assert env.transitions(state, SHORT) == env.transitions(state, LONG)

    def test_per_step_is_the_pay_divided_by_the_length(self) -> None:
        env = two_loops(Rng(1))
        assert env.per_step(SHORT) == pytest.approx(1.0)
        assert env.per_step(LONG) == pytest.approx(2.0)

    def test_the_long_loop_is_twice_as_good_by_that_measure(self) -> None:
        """The claim the environment exists to make, with no discount in it."""
        env = two_loops(Rng(1))
        assert env.per_step(LONG) == pytest.approx(2 * env.per_step(SHORT))


class TestItNeverEnds:
    def test_no_branch_of_the_model_terminates(self) -> None:
        env = two_loops(Rng(1))
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                for outcome in env.transitions(state, action):
                    assert not outcome.terminated

    def test_the_spec_says_so(self) -> None:
        assert two_loops(Rng(1)).spec.ends is False

    def test_it_has_no_terminal_states(self) -> None:
        assert two_loops(Rng(1)).terminal_states() == frozenset()

    def test_a_run_stops_at_the_step_limit_and_not_before(self) -> None:
        env = TwoLoops(Rng(1), steps=20)
        env.reset()
        for step in range(1, 21):
            outcome = env.step(SHORT)
            assert not outcome.terminated
            assert outcome.truncated == (step == 20)


class TestTheCrossover:
    """Two ways of finding the discount where the answer changes.

    The environment computes it by bisecting the closed form of the two loop
    values. Value iteration finds it by solving the model. They have to agree,
    or the arithmetic in the docstring is a story.
    """

    def test_it_is_about_three_quarters(self) -> None:
        assert two_loops(Rng(1)).crossover() == pytest.approx(0.7394, abs=0.001)

    def test_just_below_it_the_short_loop_wins(self) -> None:
        env = two_loops(Rng(1))
        solved = value_iteration(env, discount=env.crossover() - 0.005)
        assert solved.policy[0] == SHORT

    def test_just_above_it_the_long_loop_wins(self) -> None:
        env = two_loops(Rng(1))
        solved = value_iteration(env, discount=env.crossover() + 0.005)
        assert solved.policy[0] == LONG

    @pytest.mark.parametrize("discount", [0.5, 0.7, 0.735])
    def test_a_small_discount_prefers_the_worse_loop(self, discount: float) -> None:
        """The finding. The agent is not going wrong: it is answering the
        question it was asked, and the question had a discount in it."""
        env = two_loops(Rng(1))
        assert value_iteration(env, discount=discount).policy[0] == SHORT
        assert env.per_step(SHORT) < env.per_step(LONG)

    @pytest.mark.parametrize("discount", [0.745, 0.8, 0.9, 0.99])
    def test_a_large_discount_prefers_the_better_one(self, discount: float) -> None:
        env = two_loops(Rng(1))
        assert value_iteration(env, discount=env.spec.suggested_discount).policy[0] == (
            LONG
        )
        assert value_iteration(env, discount=discount).policy[0] == LONG

    def test_a_longer_loop_needs_a_larger_discount(self) -> None:
        """The trade, as a shape rather than one number.

        The further away the pay is, the more patient an agent has to be to
        prefer it, whatever it is worth per step.
        """
        crossovers = [TwoLoops(Rng(1), length=n).crossover() for n in (3, 5, 8)]
        assert crossovers == sorted(crossovers)

    @pytest.mark.parametrize("length", [2, 3, 5, 8])
    def test_the_suggested_discount_works_at_every_length(self, length: int) -> None:
        """A caller who gives no discount has to get one that works.

        A fixed 0.9 works at the default length and takes the worse loop at a
        length of eight, which is why the suggestion is computed rather than
        typed. This environment can do that because it knows its own answer.
        A real task that never ends has the same threshold and no way to find
        it, and that is the point rather than an aside.
        """
        env = TwoLoops(Rng(1), length=length)
        assert env.per_step(LONG) > env.per_step(SHORT)
        assert env.spec.suggested_discount > env.crossover()
        assert (
            value_iteration(env, discount=env.spec.suggested_discount).policy[0] == LONG
        )

    def test_a_fixed_nine_tenths_would_not_have(self) -> None:
        """The fault the computed suggestion removes, kept as a test.

        At a length of eight the long loop still pays a quarter more per step
        and a discount of 0.9 takes the short one.
        """
        env = TwoLoops(Rng(1), length=8)
        assert env.per_step(LONG) > env.per_step(SHORT)
        assert value_iteration(env, discount=0.9).policy[0] == SHORT

    def test_a_loop_that_is_no_better_gets_no_special_treatment(self) -> None:
        """At ten steps the two pay the same, so preferring neither is right."""
        env = TwoLoops(Rng(1), length=10)
        assert env.per_step(LONG) == pytest.approx(env.per_step(SHORT))
        assert env.spec.suggested_discount <= 0.99


class TestWhatItReports:
    def test_it_counts_the_loops_it_went_round(self) -> None:
        env = two_loops(Rng(1))
        env.reset()
        for _ in range(3):
            env.step(SHORT)
        for _ in range(5):
            env.step(LONG)
        assert env.audit() == {"short_loops": 3.0, "long_loops": 1.0}

    def test_resetting_forgets_them(self) -> None:
        env = two_loops(Rng(1))
        env.reset()
        env.step(SHORT)
        env.reset()
        assert env.audit() == {"short_loops": 0.0, "long_loops": 0.0}

    def test_it_draws_where_it_is(self) -> None:
        env = two_loops(Rng(1))
        env.reset()
        assert env.render() == "@oooo"
        env.step(LONG)
        assert env.render() == "o@ooo"
