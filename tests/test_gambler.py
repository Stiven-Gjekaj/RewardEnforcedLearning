"""Tests for the bet on a biased coin.

The load bearing tests are `TestTheClosedForm` and
`TestStakingNothingIsNotAnAction`. The first is the reason the environment is
here: a fair coin makes this a fair game, so every capital is worth that
capital over the goal in arithmetic, and a sweep that disagrees is wrong
rather than differently right. The second holds a design decision that was
made by measuring: with a null stake in the action space, value iteration's
own policy never reaches an ending.
"""

from __future__ import annotations

import pytest

from rel.agents.dp import evaluate_policy, value_iteration
from rel.envs.gambler import Gambler
from rel.rng import Rng


def a_gambler(goal: int = 100, heads: float = 0.4) -> Gambler:
    return Gambler(Rng(1).stream("env"), goal=goal, heads=heads)


class TestTheShape:
    def test_a_state_for_every_capital_including_both_endings(self) -> None:
        assert a_gambler(goal=20).observation_space.n == 21

    def test_an_action_for_every_stake_up_to_half_the_goal(self) -> None:
        # Half is the largest stake that is ever legal. A larger one would win
        # past the goal, and winning past the goal is winning.
        assert a_gambler(goal=20).action_space.n == 10

    def test_the_smallest_goal_has_one_capital_and_one_stake(self) -> None:
        env = a_gambler(goal=2)
        assert env.observation_space.n == 3
        assert env.action_space.n == 1

    def test_a_goal_below_two_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            a_gambler(goal=1)

    def test_a_coin_that_never_lands_heads_is_refused(self) -> None:
        for heads in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError, match="lands heads"):
                a_gambler(heads=heads)

    def test_it_starts_at_half_the_goal(self) -> None:
        assert a_gambler(goal=100).reset() == 50


class TestTheStake:
    def test_an_action_stakes_one_more_than_its_number(self) -> None:
        assert a_gambler().stake(50, 0) == 1
        assert a_gambler().stake(50, 7) == 8

    def test_it_is_never_more_than_the_capital(self) -> None:
        assert a_gambler().stake(3, 40) == 3

    def test_it_is_never_more_than_what_is_left_to_win(self) -> None:
        assert a_gambler(goal=100).stake(97, 40) == 3

    def test_at_an_ending_there_is_nothing_to_stake(self) -> None:
        env = a_gambler(goal=100)
        assert env.stake(0, 10) == 0
        assert env.stake(100, 10) == 0

    def test_no_action_stakes_nothing_from_a_capital_that_can_be_held(
        self,
    ) -> None:
        # The whole reason the null stake was left out. An action that stakes
        # nothing is a loop of one state, and a loop of one state at a
        # discount of one ties with the best real stake at every capital.
        env = a_gambler(goal=40)
        for capital in range(1, env.goal):
            for action in range(env.action_space.n):
                assert env.stake(capital, action) >= 1


class TestTheModel:
    def test_a_bet_has_two_branches_that_add_up_to_one(self) -> None:
        env = a_gambler(heads=0.4)
        branches = env.transitions(50, 9)
        assert len(branches) == 2
        assert sum(branch.probability for branch in branches) == pytest.approx(1.0)

    def test_heads_adds_the_stake_and_tails_takes_it(self) -> None:
        env = a_gambler(heads=0.4)
        won, lost = env.transitions(50, 9)
        assert (won.probability, won.observation) == (0.4, 60)
        assert (lost.probability, lost.observation) == (pytest.approx(0.6), 40)

    def test_only_reaching_the_goal_pays(self) -> None:
        env = a_gambler(goal=100, heads=0.4)
        won, lost = env.transitions(90, 9)
        assert (won.observation, won.reward, won.terminated) == (100, 1.0, True)
        assert (lost.observation, lost.reward, lost.terminated) == (80, 0.0, False)

    def test_losing_everything_ends_the_episode_and_pays_nothing(self) -> None:
        env = a_gambler(goal=100, heads=0.4)
        _, lost = env.transitions(10, 9)
        assert (lost.observation, lost.reward, lost.terminated) == (0, 0.0, True)

    def test_an_ending_stays_where_it_is(self) -> None:
        env = a_gambler(goal=100)
        for state in (0, 100):
            (only,) = env.transitions(state, 3)
            assert (only.probability, only.observation, only.terminated) == (
                1.0,
                state,
                True,
            )

    def test_the_two_endings_are_named_rather_than_read_off_the_model(
        self,
    ) -> None:
        # At a goal of two there is one capital and both of its branches reach
        # an ending, so a model reader would call the capital an ending too.
        # `RandomWalk` has the same trap at its smallest size.
        assert a_gambler(goal=2).terminal_states() == frozenset({0, 2})
        assert value_iteration(a_gambler(goal=2, heads=0.4)).values[1] > 0.0


class TestStepping:
    def test_a_win_adds_the_stake(self) -> None:
        env = a_gambler(goal=100, heads=1.0 - 1e-12)
        env.reset()
        step = env.step(9)
        assert step.observation == 60

    def test_a_loss_takes_the_stake(self) -> None:
        env = a_gambler(goal=100, heads=1e-12)
        env.reset()
        step = env.step(9)
        assert step.observation == 40

    def test_reaching_the_goal_pays_one_and_ends_it(self) -> None:
        env = a_gambler(goal=100, heads=1.0 - 1e-12)
        env.reset()
        for _ in range(10):
            step = env.step(env.action_space.n - 1)
            if step.terminated:
                break
        assert (step.observation, step.reward, step.terminated) == (100, 1.0, True)

    def test_losing_everything_ends_it_and_pays_nothing(self) -> None:
        env = a_gambler(goal=100, heads=1e-12)
        env.reset()
        for _ in range(10):
            step = env.step(env.action_space.n - 1)
            if step.terminated:
                break
        assert (step.observation, step.reward, step.terminated) == (0, 0.0, True)

    def test_a_seed_replays_the_coin(self) -> None:
        first = Gambler(Rng(3).stream("env"))
        second = Gambler(Rng(3).stream("env"))
        first.reset()
        second.reset()
        for _ in range(20):
            assert first.step(2).observation == second.step(2).observation


class TestTheClosedForm:
    """A fair game stopped at either end is worth the same however it is played.

    So the chance of reaching the goal from a capital is that capital over the
    goal. This is one of two environments here whose values are known without
    a sweep, and a sweep that disagrees with it is wrong rather than merely
    different.
    """

    def test_a_fair_coin_is_the_capital_over_the_goal(self) -> None:
        env = a_gambler(goal=20, heads=0.5)
        assert env.true_values() == pytest.approx(
            [capital / 20 for capital in range(21)]
        )

    def test_the_sweep_agrees_with_it(self) -> None:
        env = a_gambler(goal=100, heads=0.5)
        solved = value_iteration(env, discount=1.0, tolerance=1e-12)
        truth = env.true_values()
        worst = max(
            abs(solved.values[capital] - truth[capital]) for capital in range(1, 100)
        )
        assert worst < 1e-9

    def test_a_biased_coin_is_refused_rather_than_answered_wrongly(self) -> None:
        with pytest.raises(ValueError, match="fair coin"):
            a_gambler(heads=0.4).true_values()

    def test_the_endings_are_left_out_of_the_scoring(self) -> None:
        env = a_gambler(goal=20, heads=0.5)
        assert set(env.values_to_score()) == set(range(1, 20))


class TestStakingNothingIsNotAnAction:
    """Held by what happens when it is one, rather than by an assertion.

    A null stake ties with the best real stake at every capital, so a solver
    that breaks ties towards the lower action takes it, and the policy it
    reports never reaches an ending from anywhere.
    """

    def test_the_optimal_policy_reaches_an_ending(self) -> None:
        for heads in (0.25, 0.4, 0.5, 0.55):
            env = a_gambler(goal=100, heads=heads)
            solved = value_iteration(env, discount=1.0)
            report = evaluate_policy(env, list(solved.policy), discount=1.0)
            assert report.reaches_end, heads

    def test_the_optimal_policy_is_worth_what_the_sweep_said(self) -> None:
        env = a_gambler(goal=100, heads=0.4)
        solved = value_iteration(env, discount=1.0)
        report = evaluate_policy(env, list(solved.policy), discount=1.0)
        assert report.start_value == pytest.approx(solved.start_value, abs=1e-6)

    def test_a_null_stake_would_have_broken_both(self) -> None:
        """The measurement the design decision came from.

        A gambler with the null stake put back, solved the same way, whose
        greedy policy stakes nothing at a third of its capitals and therefore
        never ends.
        """

        class WithNothing(Gambler):
            """The action space as the problem states it, zero included."""

            def stake(self, state: int, action: int) -> int:
                if self.is_ending(state):
                    return 0
                return min(action, state, self.goal - state)

        env = WithNothing(Rng(1).stream("env"), goal=100, heads=0.4)
        solved = value_iteration(env, discount=1.0)
        stalls = [
            capital
            for capital in range(1, 100)
            if env.stake(capital, solved.policy[capital]) == 0
        ]
        assert len(stalls) > 20
        assert not evaluate_policy(env, list(solved.policy), discount=1.0).reaches_end


class TestWhatTheRunReallyDid:
    def test_it_reports_nothing_before_a_flip(self) -> None:
        env = a_gambler()
        env.reset()
        assert env.audit() == {}

    def test_it_counts_the_flips_and_the_heads(self) -> None:
        env = a_gambler(goal=100, heads=1.0 - 1e-12)
        env.reset()
        env.step(0)
        env.step(0)
        assert env.audit() == {"flips": 2.0, "heads_share": 1.0}

    def test_a_run_of_tails_shares_nothing(self) -> None:
        env = a_gambler(goal=100, heads=1e-12)
        env.reset()
        env.step(0)
        assert env.audit()["heads_share"] == 0.0


class TestTheRegistryEntries:
    def test_both_are_registered(self) -> None:
        from rel.envs import ENVIRONMENTS

        for name in ("gambler", "fair-gambler"):
            assert "tabular" in ENVIRONMENTS[name].tags

    def test_the_fair_one_really_is_fair(self) -> None:
        from rel.envs import ENVIRONMENTS

        env = ENVIRONMENTS.make("fair-gambler", Rng(1).stream("env"))
        assert isinstance(env, Gambler)
        assert env.heads == 0.5

    def test_the_default_one_is_the_coin_the_literature_draws(self) -> None:
        from rel.envs import ENVIRONMENTS

        env = ENVIRONMENTS.make("gambler", Rng(1).stream("env"))
        assert isinstance(env, Gambler)
        assert (env.heads, env.goal) == (0.4, 100)


class TestItDraws:
    def test_the_bar_fills_with_the_capital(self) -> None:
        env = a_gambler(goal=100)
        env.reset()
        assert "50 of 100" in env.render()
        assert env.render().count("#") == 20

    def test_an_empty_purse_is_an_empty_bar(self) -> None:
        env = a_gambler(goal=100)
        env.reset()
        env.at = 0
        assert env.render().count("#") == 0
