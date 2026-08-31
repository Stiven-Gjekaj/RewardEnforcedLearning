"""Tests for prediction over features, and for the divergence it can have.

The load bearing test is `TestItIsTheTabularUpdateWithTheTableRemoved`. Over a
table of one hot rows a linear predictor is a tabular one, exactly, and if the
two ever disagree then the semi-gradient update is wrong and every number
measured from it is wrong with it.

`TestTheTriad` is the point of the module. It runs the agent on Baird's
counterexample and reads what happens to its weights, above the discount that
makes the problem hard and below it.
"""

from __future__ import annotations

from functools import cache

import pytest

from rel.agents.base import Transition
from rel.agents.linear_prediction import LinearPredictor, SemiGradientTD, fixed
from rel.agents.lookup import Lookup
from rel.agents.prediction import TemporalDifference
from rel.envs.baird import STARTING_WEIGHTS, Baird
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

TWO = Discrete(2)
EVEN = fixed([0.5, 0.5])

#: One weight for each of three states and nothing shared between them, which
#: is a table written as features.
ONE_HOT: list[list[float]] = [
    [1.0 if column == state else 0.0 for column in range(3)] for state in range(3)
]


def step(here: int, action: int, reward: float, there: int) -> Transition[int]:
    return Transition(here, action, reward, there, False, False)


def ends(here: int, action: int, reward: float, there: int) -> Transition[int]:
    return Transition(here, action, reward, there, True, False)


def a_predictor(
    cls: type[LinearPredictor[int]] = SemiGradientTD,
    rows: list[list[float]] | None = None,
    **options: float,
) -> LinearPredictor[int]:
    return cls(
        Rng(1).stream("agent"),
        TWO,
        Lookup(ONE_HOT if rows is None else rows),
        EVEN,
        **options,  # type: ignore[arg-type]
    )


class TestTheFixedPolicy:
    def test_it_gives_the_same_shares_wherever_it_is(self) -> None:
        policy = fixed([0.25, 0.75])
        assert policy(0) == policy(9) == (0.25, 0.75)

    def test_a_policy_with_no_actions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="a share for each action"):
            fixed([])

    def test_a_negative_share_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not negative"):
            fixed([1.5, -0.5])

    def test_shares_that_do_not_add_up_are_refused(self) -> None:
        with pytest.raises(ValueError, match="add up to one"):
            fixed([0.5, 0.4])

    def test_the_shares_are_copied_rather_than_held(self) -> None:
        shares = [0.5, 0.5]
        policy = fixed(shares)
        shares[0] = 99.0
        assert policy(0) == (0.5, 0.5)


class TestWhatItBelieves:
    def test_it_starts_at_nothing(self) -> None:
        agent = a_predictor()
        assert agent.weights == [0.0, 0.0, 0.0]
        assert agent.value(1) == 0.0

    def test_an_optimistic_start_reaches_the_value_asked_for(self) -> None:
        agent = a_predictor(start_value=4.0)
        assert agent.value(2) == 4.0

    def test_the_value_is_the_features_dotted_with_the_weights(self) -> None:
        agent = a_predictor(rows=[[2.0, 1.0], [0.0, 3.0]])
        agent.weights[:] = [5.0, 7.0]
        assert agent.value(0) == 17.0
        assert agent.value(1) == 21.0

    def test_the_state_value_is_the_value(self) -> None:
        agent = a_predictor()
        agent.weights[:] = [1.0, 2.0, 3.0]
        assert agent.state_value(2) == agent.value(2)

    def test_it_ranks_no_actions(self) -> None:
        # A predictor is not choosing, so a value map of its actions would be
        # a picture of something it does not keep.
        assert a_predictor().action_values(0) is None

    def test_the_largest_weight_ignores_the_sign(self) -> None:
        agent = a_predictor()
        agent.weights[:] = [1.0, -9.0, 3.0]
        assert agent.largest_weight() == 9.0

    def test_the_error_against_the_answer_is_a_root_mean_square(self) -> None:
        agent = a_predictor()
        agent.weights[:] = [3.0, 0.0, 4.0]
        assert agent.error_against({0: 0.0, 2: 0.0}) == pytest.approx(12.5**0.5)

    def test_the_error_against_nothing_is_nothing(self) -> None:
        assert a_predictor().error_against({}) == 0.0

    def test_the_error_of_a_run_away_is_a_number_rather_than_a_raise(self) -> None:
        # The one case this method exists for. A power would raise here.
        agent = a_predictor()
        agent.weights[:] = [1e300, 0.0, 0.0]
        assert agent.error_against({0: 0.0}) == float("inf")

    def test_what_it_learned_is_the_weights(self) -> None:
        agent = a_predictor()
        agent.weights[:] = [1.0, 2.0, 3.0]
        assert list(agent.learned()) == ["weights|1,2,3"]

    def test_it_says_which_coder_it_is_over(self) -> None:
        assert repr(a_predictor()) == "SemiGradientTD(Lookup(states=3, features=3))"

    def test_a_discount_above_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            a_predictor(discount=1.5)


class TestWhichPolicyItFollows:
    def test_it_acts_from_the_behaviour_policy(self) -> None:
        agent = a_predictor()
        agent.behaviour = fixed([0.0, 1.0])
        assert [agent.act(0) for _ in range(20)] == [1] * 20

    def test_it_evaluates_the_target_policy(self) -> None:
        # `greedy` is what an evaluation run uses, and the policy a predictor
        # is asked about is the target one.
        agent = a_predictor()
        agent.behaviour = fixed([1.0, 0.0])
        agent.target = fixed([0.0, 1.0])
        assert [agent.greedy(0) for _ in range(20)] == [1] * 20

    def test_with_no_target_the_two_are_the_same_object(self) -> None:
        agent = a_predictor()
        assert agent.target is agent.behaviour

    def test_the_action_offset_is_taken_off_before_the_ratio(self) -> None:
        # An action space that does not start at zero indexes the shares from
        # its own start, and a ratio read at the raw action would be the wrong
        # share or an index error.
        agent = SemiGradientTD(
            Rng(1).stream("agent"),
            Discrete(2, start=5),
            Lookup(ONE_HOT),
            fixed([0.5, 0.5]),
            fixed([0.0, 1.0]),
        )
        assert agent.correction(step(0, 6, 0.0, 1)) == 2.0
        assert agent.correction(step(0, 5, 0.0, 1)) == 0.0


class TestItIsTheTabularUpdateWithTheTableRemoved:
    """One hot features are a table, so the two agents have to agree exactly.

    Both are fed the same transitions rather than left to act, because the two
    draw their actions from the same stream in different ways and would
    otherwise walk different paths.
    """

    @staticmethod
    def _walk() -> list[Transition[int]]:
        return [
            step(0, 0, 1.0, 1),
            step(1, 1, -2.0, 2),
            step(2, 0, 0.5, 0),
            step(0, 1, 3.0, 2),
            ends(2, 0, -1.0, 0),
            step(1, 0, 0.25, 0),
        ]

    @pytest.mark.parametrize("discount", [1.0, 0.9, 0.5])
    def test_the_values_match_to_the_last_bit(self, discount: float) -> None:
        table = TemporalDifference(
            Rng(1).stream("agent"),
            TWO,
            step_size=0.3,
            discount=discount,
        )
        features = SemiGradientTD(
            Rng(1).stream("agent"),
            TWO,
            Lookup(ONE_HOT),
            EVEN,
            step_size=0.3,
            discount=discount,
        )

        for transition in self._walk() * 4:
            table.observe(transition)
            features.observe(transition)

        for state in range(3):
            assert features.value(state) == table.value(state), state

    def test_a_terminated_step_bootstraps_from_nothing_in_both(self) -> None:
        table = TemporalDifference(Rng(1).stream("agent"), TWO, step_size=0.5)
        features = a_predictor(step_size=0.5)

        table.observe(step(0, 0, 0.0, 1))
        features.observe(step(0, 0, 0.0, 1))
        table.v[1] = 10.0
        features.weights[1] = 10.0

        table.observe(ends(0, 0, 2.0, 1))
        features.observe(ends(0, 0, 2.0, 1))
        assert features.value(0) == table.value(0) == 1.0


class TestTheSemiGradientUpdate:
    def test_one_step_moves_the_active_weights_by_their_share(self) -> None:
        agent = a_predictor(rows=[[2.0, 1.0], [0.0, 0.0]], step_size=0.5, discount=0.0)
        # The features of state 0 dot themselves to five, so the step size is
        # a tenth. The error is the reward, so each weight moves by its value
        # times a tenth times one.
        agent.observe(step(0, 0, 4.0, 1))
        assert agent.weights == pytest.approx([0.1 * 4.0 * 2.0, 0.1 * 4.0 * 1.0])

    def test_an_action_the_target_policy_never_takes_moves_nothing(self) -> None:
        agent = SemiGradientTD(
            Rng(1).stream("agent"),
            TWO,
            Lookup(ONE_HOT),
            fixed([0.5, 0.5]),
            fixed([0.0, 1.0]),
            step_size=0.5,
        )
        agent.observe(step(0, 0, 100.0, 1))
        assert agent.weights == [0.0, 0.0, 0.0]

    def test_a_step_the_target_policy_prefers_counts_for_more(self) -> None:
        alone = a_predictor(step_size=0.5)
        alone.observe(step(0, 1, 1.0, 1))

        corrected = SemiGradientTD(
            Rng(1).stream("agent"),
            TWO,
            Lookup(ONE_HOT),
            fixed([0.5, 0.5]),
            fixed([0.0, 1.0]),
            step_size=0.5,
        )
        corrected.observe(step(0, 1, 1.0, 1))
        assert corrected.value(0) == pytest.approx(2.0 * alone.value(0))


@cache
def on_baird(
    which: str, episodes: int, discount: float = 0.99, seed: int = 1
) -> tuple[float, float]:
    """The largest weight and the value error after a run of the counterexample.

    Cached, because several tests read the same run and each one is thousands
    of steps. The episode limit of the environment is a thousand steps, so an
    episode here is a thousand steps of learning.
    """
    env = Baird(Rng(seed).stream("env"))
    coder = Lookup(env.feature_rows)
    cls = {"semi-gradient": SemiGradientTD}[which]
    agent: LinearPredictor[int] = cls(
        Rng(seed).stream("agent"),
        env.action_space,
        coder,
        fixed(env.behaviour_shares),
        fixed(env.target_shares),
        step_size=0.05,
        discount=discount,
    )
    agent.weights[:] = list(STARTING_WEIGHTS)

    train(env, agent, episodes, discount=discount)
    truth = dict.fromkeys(range(env.observation_space.n), 0.0)
    return agent.largest_weight(), agent.error_against(truth)


class TestTheTriad:
    def test_the_semi_gradient_weights_run_away(self) -> None:
        assert on_baird("semi-gradient", 20)[0] > 1e15

    def test_they_keep_running_away_rather_than_settling(self) -> None:
        early = on_baird("semi-gradient", 5)[0]
        late = on_baird("semi-gradient", 20)[0]
        assert late > early * 1e6

    def test_the_two_start_from_the_same_place(self) -> None:
        # Nothing about the start explains the difference between them.
        assert max(STARTING_WEIGHTS) == 10.0

    def test_the_semi_gradient_agent_settles_below_the_threshold(self) -> None:
        # The same three ingredients, the same features, the same start. Only
        # the discount is different, and it decides everything.
        assert on_baird("semi-gradient", 20, discount=0.85)[1] < 0.01

    def test_it_keeps_falling_rather_than_settling_short(self) -> None:
        assert (
            on_baird("semi-gradient", 20, discount=0.85)[1]
            < on_baird("semi-gradient", 5, discount=0.85)[1] / 100.0
        )

    def test_it_runs_away_above_the_threshold(self) -> None:
        # Five hundredths of a discount either side of the crossing, and the
        # difference between the two runs is nine orders of magnitude.
        assert on_baird("semi-gradient", 20, discount=0.9)[1] > 1e3

    def test_a_settled_run_keeps_a_weight_nothing_can_see(self) -> None:
        """Below the threshold the weights stop moving and are not zero.

        What is left is the part of the start that no update can reach: a
        direction in which every state's features are worth nothing, so it
        changes no value and no error can find it. The largest weight is the
        wrong thing to read there, and the value error is the right one.
        """
        largest, error = on_baird("semi-gradient", 20, discount=0.85)
        assert error < 0.01
        assert largest > 5.0

    def test_the_divergence_is_not_one_seed(self) -> None:
        for seed in (1, 2, 3):
            assert on_baird("semi-gradient", 10, seed=seed)[0] > 1e8, seed
