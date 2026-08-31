"""Tests for the counterexample.

Most of what makes this environment a counterexample is not in its transitions.
It is in the table of features and the pair of policies it carries, so most of
these tests are about those, and the ones about stepping are there to hold the
model and the code together.

The divergence itself is not tested here. It belongs to the agent that
diverges, and `tests/test_linear_prediction.py` measures it.
"""

from __future__ import annotations

import pytest

from rel.agents.lookup import Lookup
from rel.envs import ENVIRONMENTS
from rel.envs.baird import BEHAVIOUR, DASHED, SOLID, STARTING_WEIGHTS, TARGET, Baird
from rel.rng import Rng


def build(upper: int = 6, seed: int = 1) -> Baird:
    return Baird(Rng(seed).stream("env"), upper=upper)


class TestTheShape:
    def test_it_is_in_the_registry(self) -> None:
        env = ENVIRONMENTS.make("baird", Rng(1).stream("env"))
        assert isinstance(env, Baird)

    def test_seven_states_by_default(self) -> None:
        assert build().observation_space.n == 7

    def test_two_actions(self) -> None:
        assert build().action_space.n == 2

    def test_one_upper_state_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two upper states"):
            build(upper=1)

    def test_it_never_ends(self) -> None:
        assert not build().spec.ends
        assert build().terminal_states() == frozenset()

    def test_it_suggests_a_discount_below_one(self) -> None:
        # A task with no ending and no discount is worth an unbounded amount.
        assert build().spec.suggested_discount == 0.99


class TestWhereTheActionsGo:
    def test_solid_goes_to_the_lower_state_from_anywhere(self) -> None:
        env = build()
        for state in range(7):
            assert env.transitions(state, SOLID) == (
                *env.transitions(0, SOLID),
            ), state
        assert env.transitions(3, SOLID)[0].observation == 6

    def test_dashed_spreads_over_the_upper_states_only(self) -> None:
        env = build()
        landed = {outcome.observation for outcome in env.transitions(6, DASHED)}
        assert landed == {0, 1, 2, 3, 4, 5}

    def test_dashed_is_even(self) -> None:
        shares = {outcome.probability for outcome in build().transitions(0, DASHED)}
        assert shares == {1.0 / 6.0}

    def test_nothing_pays_anything(self) -> None:
        env = build()
        for state in range(7):
            for action in (DASHED, SOLID):
                for outcome in env.transitions(state, action):
                    assert outcome.reward == 0.0

    def test_stepping_solid_really_lands_below(self) -> None:
        env = build()
        env.reset()
        assert env.step(SOLID).observation == 6

    def test_stepping_dashed_never_lands_below(self) -> None:
        env = build()
        env.reset()
        for _ in range(200):
            assert env.step(DASHED).observation != 6


class TestTheFeatures:
    def test_there_is_one_row_for_each_state(self) -> None:
        assert len(build().feature_rows) == 7

    def test_there_are_two_more_weights_than_upper_states(self) -> None:
        assert len(build().feature_rows[0]) == 8
        assert len(build(upper=3).feature_rows[0]) == 5

    def test_an_upper_state_leans_twice_on_its_own_and_once_on_the_shared(
        self,
    ) -> None:
        rows = build().feature_rows
        assert rows[2] == (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def test_the_lower_state_leans_once_on_its_own_and_twice_on_the_shared(
        self,
    ) -> None:
        assert build().feature_rows[6] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0)

    def test_every_state_touches_the_shared_weight(self) -> None:
        # This is the whole mechanism. Take the shared weight away and every
        # row is one state's own feature, which is a table with extra steps.
        for row in build().feature_rows:
            assert row[-1] > 0.0

    def test_every_row_adds_up_to_three(self) -> None:
        assert {sum(row) for row in build().feature_rows} == {3.0}

    def test_the_rows_are_a_table_a_coder_accepts(self) -> None:
        coder = Lookup(build().feature_rows)
        assert coder.features == 8
        assert coder.states == 7
        assert coder.encode(0) == ((0, 7), (2.0, 1.0))

    def test_zero_weights_make_every_state_worth_zero(self) -> None:
        # The true value under either policy is zero everywhere, and the
        # approximation can say so exactly. Nothing about the shape of this
        # problem is what a diverging method is failing at.
        coder = Lookup(build().feature_rows)
        weights = [0.0] * coder.features
        for state in range(coder.states):
            indices, values = coder.encode(state)
            worth = sum(
                weights[index] * value
                for index, value in zip(indices, values, strict=True)
            )
            assert worth == 0.0


class TestThePolicies:
    def test_the_behaviour_policy_mostly_dashes(self) -> None:
        assert build().behaviour_shares == BEHAVIOUR
        assert BEHAVIOUR[DASHED] == pytest.approx(6.0 / 7.0)

    def test_the_target_policy_never_dashes(self) -> None:
        assert build().target_shares == TARGET
        assert TARGET[DASHED] == 0.0

    def test_both_add_up_to_one(self) -> None:
        assert sum(BEHAVIOUR) == pytest.approx(1.0)
        assert sum(TARGET) == pytest.approx(1.0)

    def test_the_behaviour_policy_covers_the_target_one(self) -> None:
        # Coverage is the one assumption importance sampling cannot do
        # without. An action the target policy might take has to be one the
        # behaviour policy sometimes takes.
        for share, wanted in zip(BEHAVIOUR, TARGET, strict=True):
            assert share > 0.0 or wanted == 0.0

    def test_the_ratio_on_a_solid_step_is_seven(self) -> None:
        assert TARGET[SOLID] / BEHAVIOUR[SOLID] == pytest.approx(7.0)

    def test_the_ratio_on_a_dashed_step_is_zero(self) -> None:
        assert TARGET[DASHED] / BEHAVIOUR[DASHED] == 0.0


class TestTheStartingWeights:
    def test_there_is_one_for_each_feature(self) -> None:
        assert len(STARTING_WEIGHTS) == len(build().feature_rows[0])

    def test_only_the_lower_states_own_weight_is_large(self) -> None:
        assert STARTING_WEIGHTS == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0)

    def test_nothing_in_the_package_reads_them(self) -> None:
        """They belong to the demonstration, not to the environment.

        A constant that the environment handed to an agent would be the
        environment choosing how the agent starts, which is the line this file
        already bends far enough.
        """
        env = build()
        assert not hasattr(env, "starting_weights")


class TestTheSizeIsASetting:
    def test_three_upper_states_still_share_one_weight(self) -> None:
        rows = build(upper=3).feature_rows
        assert len(rows) == 4
        assert rows[3] == (0.0, 0.0, 0.0, 1.0, 2.0)
        assert {sum(row) for row in rows} == {3.0}

    def test_dashed_spreads_over_however_many_there_are(self) -> None:
        shares = {outcome.probability for outcome in build(upper=3).transitions(0, 0)}
        assert shares == {1.0 / 3.0}

    def test_the_start_is_even_over_all_of_them(self) -> None:
        starts = build(upper=3).start_states()
        assert len(starts) == 4
        assert {share for share, _ in starts} == {0.25}


class TestTheDrawing:
    def test_the_upper_row_and_the_lower_state_are_both_drawn(self) -> None:
        env = build()
        env.at = 6
        upper, lower = env.render().splitlines()
        assert upper == "o o o o o o"
        assert lower.strip() == "@"

    def test_the_state_it_is_in_is_marked(self) -> None:
        env = build()
        env.at = 2
        assert env.render().splitlines()[0] == "o o @ o o o"
