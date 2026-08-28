"""Tests for an action that lasts several steps.

The test that matters most is the collapse: an option whose stopping rule
holds everywhere takes one step and hands control back, so it is a primitive
action wearing this shape. An options agent that could not hold its own
primitive actions would be a different kind of agent rather than a more
general one.
"""

from __future__ import annotations

import pytest

from rel.options import Option, primitive, primitives
from rel.spaces import Discrete


def a_corridor() -> Option:
    """Walk right along states 0 to 3 and stop at 3."""
    return Option(name="to the end", policy={0: 1, 1: 1, 2: 1}, stops=frozenset({3}))


class TestTheThreeParts:
    def test_it_can_start_where_its_policy_has_an_answer(self) -> None:
        option = a_corridor()
        assert option.can_start(0)
        assert option.can_start(2)

    def test_it_cannot_start_where_its_policy_is_silent(self) -> None:
        assert not a_corridor().can_start(9)

    def test_it_says_what_to_do_while_it_runs(self) -> None:
        assert a_corridor().act(1) == 1

    def test_asking_it_about_a_state_it_does_not_cover_is_refused(self) -> None:
        # Silently returning an action chosen for somewhere else is the fault
        # this prevents, and it would show up as an agent that walks into a
        # wall for twenty steps rather than as an error.
        with pytest.raises(ValueError, match="no action for state 9"):
            a_corridor().act(9)

    def test_it_stops_where_its_stopping_rule_says(self) -> None:
        assert a_corridor().stops_at(3)

    def test_it_keeps_going_in_the_middle(self) -> None:
        assert not a_corridor().stops_at(1)

    def test_walking_out_of_its_own_states_stops_it(self) -> None:
        # The rule is total, so an option that leaves the room it was built
        # for ends there rather than carrying on with a stale action.
        assert a_corridor().stops_at(9)

    def test_it_says_how_many_states_it_covers(self) -> None:
        assert len(a_corridor()) == 3


class TestAnOptionThatStopsAfterOneStepIsAnAction:
    def test_it_stops_wherever_it_lands(self) -> None:
        option = primitive(2, range(5))
        assert all(option.stops_at(state) for state in range(5))

    def test_it_takes_the_same_action_everywhere(self) -> None:
        option = primitive(2, range(5))
        assert [option.act(state) for state in range(5)] == [2] * 5

    def test_it_can_still_start_anywhere_it_covers(self) -> None:
        # An option always takes at least one step, so a stopping rule that
        # holds everywhere does not stop it before it has moved. Testing the
        # rule at the state it started in would make this option useless and
        # the collapse to a primitive action impossible.
        option = primitive(2, range(5))
        assert all(option.can_start(state) for state in range(5))

    def test_there_is_one_for_each_action_in_order(self) -> None:
        built = primitives(Discrete(4), range(3))
        assert [option.act(0) for option in built] == [0, 1, 2, 3]

    def test_they_are_named_after_the_action_they_take(self) -> None:
        assert primitives(Discrete(2), [0])[1].name == "action 1"

    def test_an_action_space_that_does_not_start_at_zero_is_followed(self) -> None:
        built = primitives(Discrete(3, start=5), [0])
        assert [option.act(0) for option in built] == [5, 6, 7]
