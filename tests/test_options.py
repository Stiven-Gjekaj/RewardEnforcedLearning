"""Tests for an action that lasts several steps.

The test that matters most is the collapse: an option whose stopping rule
holds everywhere takes one step and hands control back, so it is a primitive
action wearing this shape. An options agent that could not hold its own
primitive actions would be a different kind of agent rather than a more
general one.
"""

from __future__ import annotations

import pytest

from rel.envs.classic import (
    cliff_walk,
    dyna_maze,
    four_rooms,
    frozen_lake,
    windy_grid,
)
from rel.options import (
    Option,
    hallway_options,
    primitive,
    primitives,
    reachable,
    rooms,
    steps_between,
)
from rel.rng import Rng
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


class TestReadingTheRoomsOffAModel:
    def test_a_step_that_lands_where_it_started_is_not_a_way_somewhere(self) -> None:
        # Walking into a wall. The cell is not a neighbour of itself, and
        # counting it as one would join every room to itself.
        env = four_rooms(Rng(1))
        edges = steps_between(env)
        assert all(state not in landings for state, landings in edges.items())

    def test_the_cliff_cells_are_described_and_never_stood_in(self) -> None:
        # Walking into the cliff sends the agent back to the start, so the
        # model has these cells and no run ever holds one. Counting each as a
        # room of its own would be counting places nobody goes.
        env = cliff_walk(Rng(1))
        stood_in = reachable(env)
        assert len(stood_in) == 37
        assert env.state_of(3, 5) not in stood_in

    def test_the_four_rooms_grid_falls_into_four(self) -> None:
        env = four_rooms(Rng(1))
        assert sorted(len(part) for part in rooms(env, env.gaps())) == [20, 25, 25, 29]

    def test_no_room_holds_a_doorway(self) -> None:
        env = four_rooms(Rng(1))
        doors = set(env.gaps())
        assert all(not (part & doors) for part in rooms(env, env.gaps()))

    def test_every_cell_is_in_a_room_or_a_doorway(self) -> None:
        env = four_rooms(Rng(1))
        doors = set(env.gaps())
        held = set().union(*rooms(env, env.gaps())) | doors
        assert held == reachable(env)

    def test_taking_no_doors_out_leaves_one_piece(self) -> None:
        env = four_rooms(Rng(1))
        assert len(rooms(env, [])) == 1

    @pytest.mark.parametrize(
        "builder",
        [cliff_walk, windy_grid, frozen_lake, dyna_maze],
        ids=lambda b: b.__name__,
    )
    def test_a_grid_with_no_doors_between_rooms_is_one_room(self, builder) -> None:  # type: ignore[no-untyped-def]
        # The maze is the interesting one here. It has four gaps and they are
        # passages inside one room rather than doors between two, so taking
        # them out leaves the grid in one piece and it gets no options.
        env = builder(Rng(1))
        assert len(rooms(env, env.gaps())) == 1

    def test_the_rooms_come_back_in_a_stable_order(self) -> None:
        env = four_rooms(Rng(1))
        assert rooms(env, env.gaps()) == rooms(four_rooms(Rng(2)), env.gaps())


class TestTheHallwayOptions:
    def test_the_four_rooms_grid_gets_two_options_per_room(self) -> None:
        # Sutton, Precup and Singh, 1999. Eight options, one for each room and
        # each of the two doorways on its edge.
        env = four_rooms(Rng(1))
        assert len(hallway_options(env, env.gaps())) == 8

    def test_each_option_reaches_its_hallway_from_everywhere_it_can_start(
        self,
    ) -> None:
        # The claim the option makes about itself. Follow it from every state
        # in its start set and it has to arrive at the doorway it names.
        env = four_rooms(Rng(1))
        for option in hallway_options(env, env.gaps()):
            door = next(iter(option.stops))
            for start in option.policy:
                at, steps = start, 0
                while not option.stops_at(at) or steps == 0:
                    at = max(
                        env.transitions(at, option.act(at)),
                        key=lambda outcome: outcome.probability,
                    ).observation
                    steps += 1
                    assert steps < 100, f"{option.name} did not stop from {start}"
                assert at == door, f"{option.name} stopped at {at} from {start}"

    def test_it_cannot_start_where_it_stops(self) -> None:
        # An option that could would take an arbitrary first step and then
        # walk back to where it already was.
        env = four_rooms(Rng(1))
        for option in hallway_options(env, env.gaps()):
            assert not (set(option.policy) & option.stops)

    def test_it_can_start_at_the_room_s_other_doorway(self) -> None:
        # This is what makes them worth having. An agent that has just arrived
        # at one doorway can choose to cross the room to the other.
        env = four_rooms(Rng(1))
        doors = set(env.gaps())
        assert any(set(option.policy) & doors for option in hallway_options(env, doors))

    def test_a_doorway_that_joins_no_two_rooms_gives_no_options(self) -> None:
        # The maze has four gaps and none of them is a way between two rooms.
        env = dyna_maze(Rng(1))
        assert hallway_options(env, env.gaps()) == []

    def test_a_grid_with_no_doorways_gives_no_options(self) -> None:
        env = cliff_walk(Rng(1))
        assert hallway_options(env, env.gaps()) == []

    def test_the_options_come_back_in_a_stable_order(self) -> None:
        env = four_rooms(Rng(1))
        names = [option.name for option in hallway_options(env, env.gaps())]
        again = [
            option.name for option in hallway_options(four_rooms(Rng(9)), env.gaps())
        ]
        assert names == again


class TestAnOptionCanSayWhatShapeItIs:
    def test_one_that_stops_everywhere_is_a_primitive_action(self) -> None:
        assert primitive(1, range(5)).is_primitive

    def test_one_that_keeps_going_is_not(self) -> None:
        assert not a_corridor().is_primitive

    def test_a_hallway_option_is_not(self) -> None:
        env = four_rooms(Rng(1))
        assert not any(
            option.is_primitive for option in hallway_options(env, env.gaps())
        )
