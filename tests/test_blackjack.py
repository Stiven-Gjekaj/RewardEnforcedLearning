"""Tests for the card game with the model written out.

The load bearing test is `TestItSolvesToTheBooksPolicy`. Every other check
here says the model is self consistent, which a wrong model can also be. The
book prints the optimal policy of this game as a picture, and an exact sweep of
this model has to reproduce it square for square.

`TestTheAceCountsWhicheverWayFits` is the other one. A soft hand that survives
a card a hard hand would lose to is the whole reason the third part of the
state exists, and a model that got it wrong would still add up to one
everywhere.
"""

from __future__ import annotations

import pytest

from rel.agents.dp import value_iteration
from rel.envs.blackjack import (
    BUST,
    CARDS,
    HIT,
    LOWEST,
    STICK,
    Blackjack,
    dealer_totals,
    drawn,
    player_start,
)
from rel.rng import Rng


def a_game() -> Blackjack:
    return Blackjack(Rng(1).stream("env"))


#: The book's figure 5.2, as the lowest hand to stick on against each card the
#: dealer can show, from an ace through to a ten.
HARD_STICKS_AT = (17, 13, 13, 12, 12, 12, 17, 17, 17, 17)
SOFT_STICKS_AT = (19, 18, 18, 18, 18, 18, 18, 18, 19, 19)


def sticks_at(env: Blackjack, policy: tuple[int, ...], soft: bool) -> tuple[int, ...]:
    """The lowest hand this policy sticks on, for each card the dealer shows."""
    lowest = []
    for showing in range(1, 11):
        found = 22
        for total in range(LOWEST, 22):
            if policy[env.fold(total, showing, soft)] == STICK:
                found = total
                break
        lowest.append(found)
    return tuple(lowest)


class TestTheCards:
    def test_a_ten_comes_up_four_times_in_thirteen(self) -> None:
        # Ten, jack, queen and king are all worth ten.
        shares = dict(CARDS)
        assert shares[10] == pytest.approx(4.0 / 13.0)
        assert shares[1] == pytest.approx(1.0 / 13.0)

    def test_the_shares_add_up_to_one(self) -> None:
        assert sum(share for _, share in CARDS) == pytest.approx(1.0)

    def test_there_are_ten_of_them(self) -> None:
        assert [card for card, _ in CARDS] == list(range(1, 11))


class TestTheAceCountsWhicheverWayFits:
    def test_an_ace_is_eleven_where_that_fits(self) -> None:
        assert drawn(0, False, 1) == (11, True)

    def test_an_ace_is_one_where_eleven_would_go_past(self) -> None:
        assert drawn(15, False, 1) == (16, False)

    def test_a_soft_hand_that_goes_past_counts_its_ace_as_one(self) -> None:
        # The whole reason the third part of the state exists. Seventeen with
        # a soft ace, drawing a ten, is sixteen rather than a loss.
        assert drawn(17, True, 10) == (17, False)

    def test_a_hard_hand_that_goes_past_is_a_loss(self) -> None:
        after, soft = drawn(17, False, 10)
        assert after == 27
        assert not soft

    def test_a_second_ace_is_counted_as_one(self) -> None:
        # Thirteen soft is an ace and a two. Another ace as eleven would be
        # twenty four, so it is one, and the first ace stays soft.
        assert drawn(13, True, 1) == (14, True)

    def test_a_soft_hand_never_falls_below_the_lowest_asked_about(self) -> None:
        for total in range(LOWEST, 22):
            for card, _ in CARDS:
                after, _ = drawn(total, True, card)
                assert after >= LOWEST or after > 21


class TestTheDealer:
    def test_he_stops_at_seventeen(self) -> None:
        assert dealer_totals(17, False) == ((17, 1.0),)
        assert dealer_totals(21, False) == ((21, 1.0),)

    def test_he_stops_on_a_soft_seventeen_too(self) -> None:
        # The book's rule. A dealer who drew on soft seventeen would be a
        # different game with a different answer.
        assert dealer_totals(17, True) == ((17, 1.0),)

    def test_a_hand_past_twenty_one_is_reported_as_bust(self) -> None:
        assert dealer_totals(25, False) == ((BUST, 1.0),)

    def test_his_endings_add_up_to_one(self) -> None:
        for card, _ in CARDS:
            total, soft = drawn(0, False, card)
            assert sum(share for _, share in dealer_totals(total, soft)) == (
                pytest.approx(1.0)
            )

    def test_he_only_ever_ends_at_seventeen_or_more(self) -> None:
        for card, _ in CARDS:
            total, soft = drawn(0, False, card)
            for end, _ in dealer_totals(total, soft):
                assert end >= 17

    def test_showing_a_six_busts_him_most_often(self) -> None:
        # Which is why the player sticks lowest against it. A model that had
        # the dealer's rule wrong would not have this shape.
        busts = {}
        for card, _ in CARDS:
            total, soft = drawn(0, False, card)
            busts[card] = dict(dealer_totals(total, soft)).get(BUST, 0.0)
        assert max(busts, key=lambda card: busts[card]) == 6


class TestTheDeal:
    def test_it_never_goes_past_twenty_one(self) -> None:
        # The largest hand below twelve is eleven, and the largest card is an
        # ace counted as eleven, which would be twenty two and is one instead.
        for (total, _), _ in player_start():
            assert LOWEST <= total <= 21

    def test_the_hands_add_up_to_one(self) -> None:
        assert sum(share for _, share in player_start()) == pytest.approx(1.0)

    def test_a_soft_hand_is_dealt_sometimes_and_not_always(self) -> None:
        soft = sum(share for (_, is_soft), share in player_start() if is_soft)
        assert 0.0 < soft < 1.0

    def test_the_start_states_add_up_to_one(self) -> None:
        assert sum(share for share, _ in a_game().start_states()) == pytest.approx(1.0)

    def test_every_start_is_a_hand_that_can_be_held(self) -> None:
        env = a_game()
        for _, state in env.start_states():
            total, showing, _ = env.unfold(state)
            assert LOWEST <= total <= 21
            assert 1 <= showing <= 10


class TestFoldingAState:
    def test_folding_and_unfolding_come_back_to_the_same_place(self) -> None:
        env = a_game()
        for total in range(LOWEST, 22):
            for showing in range(1, 11):
                for soft in (False, True):
                    state = env.fold(total, showing, soft)
                    assert env.unfold(state) == (total, showing, soft)

    def test_there_is_a_state_for_every_hand_and_one_more(self) -> None:
        # Two hundred hands and the hand that is over.
        assert a_game().observation_space.n == 201

    def test_the_hand_that_is_over_is_the_last_state(self) -> None:
        env = a_game()
        assert env.over == 200
        assert env.terminal_states() == frozenset({200})


class TestTheModel:
    def test_every_pair_of_branches_adds_up(self) -> None:
        env = a_game()
        for state in range(env.observation_space.n):
            for action in (STICK, HIT):
                total = sum(
                    branch.probability for branch in env.transitions(state, action)
                )
                assert total == pytest.approx(1.0), (state, action)

    def test_a_stick_always_ends_the_hand(self) -> None:
        env = a_game()
        for state in range(200):
            for branch in env.transitions(state, STICK):
                assert branch.terminated
                assert branch.observation == env.over

    def test_a_stick_pays_one_of_three_numbers(self) -> None:
        env = a_game()
        paid = {
            branch.reward
            for state in range(200)
            for branch in env.transitions(state, STICK)
        }
        assert paid <= {1.0, 0.0, -1.0}

    def test_sticking_on_twenty_one_never_loses(self) -> None:
        env = a_game()
        for showing in range(1, 11):
            for branch in env.transitions(env.fold(21, showing, False), STICK):
                assert branch.reward >= 0.0

    def test_sticking_on_twelve_against_a_ten_usually_loses(self) -> None:
        env = a_game()
        losing = sum(
            branch.probability
            for branch in env.transitions(env.fold(12, 10, False), STICK)
            if branch.reward < 0.0
        )
        assert losing > 0.5

    def test_a_hit_that_goes_past_pays_minus_one_and_ends_it(self) -> None:
        env = a_game()
        lost = [
            branch
            for branch in env.transitions(env.fold(21, 5, False), HIT)
            if branch.terminated
        ]
        assert len(lost) == 1
        assert lost[0].reward == -1.0
        assert lost[0].observation == env.over

    def test_a_hit_that_does_not_go_past_pays_nothing_yet(self) -> None:
        env = a_game()
        for branch in env.transitions(env.fold(12, 5, False), HIT):
            if not branch.terminated:
                assert branch.reward == 0.0

    def test_a_hit_leaves_the_dealer_showing_the_same_card(self) -> None:
        env = a_game()
        for branch in env.transitions(env.fold(13, 7, False), HIT):
            if not branch.terminated:
                assert env.unfold(branch.observation)[1] == 7

    def test_hitting_a_soft_hand_never_ends_it(self) -> None:
        # A soft hand counts its ace as one instead of losing.
        env = a_game()
        for total in range(LOWEST, 22):
            for branch in env.transitions(env.fold(total, 5, True), HIT):
                assert not branch.terminated, total

    def test_the_hand_that_is_over_stays_over(self) -> None:
        env = a_game()
        for action in (STICK, HIT):
            (only,) = env.transitions(env.over, action)
            assert (only.probability, only.observation, only.terminated) == (
                1.0,
                env.over,
                True,
            )


class TestItSolvesToTheBooksPolicy:
    """The check that the model is right rather than merely consistent.

    A wrong model can add up to one everywhere. What it cannot do is solve to
    the picture the book prints, square for square, on both halves of the
    board.
    """

    @staticmethod
    def _solved() -> tuple[Blackjack, tuple[int, ...]]:
        env = a_game()
        return env, value_iteration(env, discount=1.0).policy

    def test_the_hard_hands_match(self) -> None:
        env, policy = self._solved()
        assert sticks_at(env, policy, soft=False) == HARD_STICKS_AT

    def test_the_soft_hands_match(self) -> None:
        env, policy = self._solved()
        assert sticks_at(env, policy, soft=True) == SOFT_STICKS_AT

    def test_the_deal_is_worth_less_than_nothing(self) -> None:
        # Played perfectly and without the natural bonus, the house still has
        # it. The number is stated rather than approved of.
        env = a_game()
        assert value_iteration(env, discount=1.0).start_value == pytest.approx(
            -0.04656, abs=1e-4
        )

    def test_a_soft_hand_is_never_worth_less_than_the_same_hard_one(self) -> None:
        # An ace that can be counted two ways cannot hurt.
        env = a_game()
        values = value_iteration(env, discount=1.0).values
        for total in range(LOWEST, 22):
            for showing in range(1, 11):
                soft = values[env.fold(total, showing, True)]
                hard = values[env.fold(total, showing, False)]
                assert soft >= hard - 1e-9, (total, showing)


class TestStepping:
    def test_a_deal_lands_on_a_hand_the_model_lists(self) -> None:
        env = a_game()
        allowed = {state for _, state in env.start_states()}
        for _ in range(200):
            assert env.reset() in allowed

    def test_a_stick_ends_the_hand_at_once(self) -> None:
        env = a_game()
        env.reset()
        step = env.step(STICK)
        assert step.terminated
        assert step.observation == env.over
        assert step.reward in (1.0, 0.0, -1.0)

    def test_every_landing_is_a_branch_the_model_named(self) -> None:
        env = a_game()
        for _ in range(300):
            state = env.reset()
            while True:
                action = env.rng.below(2)
                allowed = {
                    branch.observation
                    for branch in env.transitions(state, action)
                    if branch.probability > 0.0
                }
                step = env.step(action)
                assert step.observation in allowed, (state, action)
                if step.terminated or step.truncated:
                    break
                state = step.observation

    def test_a_seed_replays_the_cards(self) -> None:
        first, second = a_game(), a_game()
        for _ in range(50):
            assert first.reset() == second.reset()
            assert first.step(HIT).observation == second.step(HIT).observation


class TestWhatTheRunReallyDid:
    def test_it_reports_nothing_before_a_hand(self) -> None:
        env = a_game()
        env.reset()
        assert env.audit() == {}

    def test_it_counts_the_hands_and_what_they_paid(self) -> None:
        env = a_game()
        for _ in range(20):
            env.reset()
            env.step(STICK)
        audit = env.audit()
        assert audit["hands"] == 20.0
        assert -1.0 <= audit["won_a_hand"] <= 1.0


class TestItDraws:
    def test_it_says_the_hand_and_the_card(self) -> None:
        env = a_game()
        env.reset()
        drawn_out = env.render()
        assert "player" in drawn_out
        assert "dealer shows" in drawn_out

    def test_a_finished_hand_says_so(self) -> None:
        env = a_game()
        env.reset()
        env.step(STICK)
        assert env.render() == "the hand is over"

    def test_a_soft_ace_is_named(self) -> None:
        env = a_game()
        env.reset()
        env.total, env.soft = 18, True
        assert "soft ace" in env.render()


class TestTheRegistryEntry:
    def test_it_is_registered_as_tabular(self) -> None:
        from rel.envs import ENVIRONMENTS

        assert "tabular" in ENVIRONMENTS["blackjack"].tags

    def test_it_ends_and_asks_for_no_discount(self) -> None:
        from rel.envs import ENVIRONMENTS

        env = ENVIRONMENTS.make("blackjack", Rng(1).stream("env"))
        assert env.spec.ends
        assert env.spec.suggested_discount == 1.0
