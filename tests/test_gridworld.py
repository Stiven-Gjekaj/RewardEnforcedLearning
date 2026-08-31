"""Tests for the grid.

Most tests here write the grid they need. The presets are covered by the
contract tests at the end, which ask the same questions of all of them and so
keep working when a preset is added.
"""

from __future__ import annotations

import pytest

from rel.core import Env
from rel.envs.classic import (
    cliff_walk,
    dyna_maze,
    four_rooms,
    frozen_lake,
    windy_grid,
)
from rel.envs.gridworld import GridWorld
from rel.rng import Rng

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# A room with a wall down the middle and a gap in it. Small enough to check by
# hand, and it holds every case that a move can meet: open ground, a wall, the
# edge of the grid, the goal and a pit.
ROOM = (
    "#######",
    "#S.#..#",
    "#..X.G#",
    "#######",
)


def a_room(rng: Rng | None = None, **options: object) -> GridWorld:
    return GridWorld(rng or Rng(1), ROOM, **options)  # type: ignore[arg-type]


class TestTheLayout:
    def test_a_ragged_layout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="same width"):
            GridWorld(Rng(1), ["S.G", "S.."[:2]])

    def test_an_unknown_tile_is_refused(self) -> None:
        # A layout is written by hand, so a typed character is the fault it
        # actually has. Left unchecked it becomes a wall, silently, and the
        # grid is not the grid that was drawn.
        with pytest.raises(ValueError, match="not a grid tile"):
            GridWorld(Rng(1), ["S?G"])

    def test_a_grid_with_no_start_is_refused(self) -> None:
        with pytest.raises(ValueError, match="start cell"):
            GridWorld(Rng(1), ["..G"])

    def test_a_grid_with_no_goal_is_refused(self) -> None:
        with pytest.raises(ValueError, match="goal cell"):
            GridWorld(Rng(1), ["S.."])

    def test_an_empty_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one row"):
            GridWorld(Rng(1), [])

    def test_a_space_is_floor(self) -> None:
        grid = GridWorld(Rng(1), ["S G"])
        assert grid.tile(1) == "."

    def test_the_wind_needs_one_value_for_each_column(self) -> None:
        with pytest.raises(ValueError, match="each column"):
            GridWorld(Rng(1), ["S.G"], wind=[1, 1])

    def test_a_slip_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            GridWorld(Rng(1), ["S.G"], slip=1.5)


class TestTheGaps:
    """A gap is a cell you can only pass straight through.

    The four rooms case is the one that matters, because the four cells it has
    to find are named in the paper the grid comes from. The rest of these pin
    the shape rather than the grid.
    """

    def test_it_finds_the_gap_in_a_wall(self) -> None:
        # The room has one wall, at row 1 column 3, and the cell below it at
        # row 2 column 3 is a pit. So the only gap is where the wall stops.
        grid = a_room()
        assert grid.gaps() == [grid.state_of(2, 3)]

    def test_a_corner_is_not_a_gap(self) -> None:
        # Two neighbours, at right angles. A corner is a place you turn, and
        # nothing is on the far side of it.
        grid = GridWorld(Rng(1), ("S.#", "..#", "#G#"))
        assert grid.state_of(0, 0) not in grid.gaps()

    def test_a_cell_in_the_open_is_not_a_gap(self) -> None:
        grid = GridWorld(Rng(1), ("S..", "...", "..G"))
        assert grid.gaps() == []

    def test_a_dead_end_is_not_a_gap(self) -> None:
        # One neighbour rather than two. There is nothing to pass through to.
        grid = GridWorld(Rng(1), ("#S#", "#.#", "#G#", "###"))
        assert grid.state_of(0, 1) not in grid.gaps()

    def test_the_four_rooms_grid_has_four_doorways(self) -> None:
        # Sutton, Precup and Singh, 1999. The four hallways of that paper,
        # found from the layout rather than written down here.
        grid = four_rooms(Rng(1))
        cells = [grid.cell_of(state) for state in grid.gaps()]
        assert cells == [(3, 6), (6, 2), (7, 9), (10, 6)]

    @pytest.mark.parametrize(
        "builder", [cliff_walk, windy_grid, frozen_lake], ids=lambda b: b.__name__
    )
    def test_a_grid_with_no_walls_in_it_has_no_gaps(self, builder) -> None:  # type: ignore[no-untyped-def]
        # None of these has an interior wall, so none of them has a gap in one.
        # An options agent on them has nothing but its primitive actions.
        assert builder(Rng(1)).gaps() == []


class TestMoving:
    def test_it_starts_on_the_start_cell(self) -> None:
        grid = a_room()
        assert grid.reset() == grid.state_of(1, 1)

    def test_each_action_moves_one_cell(self) -> None:
        grid = a_room()
        grid.reset()
        grid.at = grid.state_of(2, 2)

        for action, expected in (
            (UP, (1, 2)),
            (RIGHT, (2, 3)),
            (DOWN, (3, 2)),
            (LEFT, (2, 1)),
        ):
            grid.at = grid.state_of(2, 2)
            grid.step(action)
            assert (
                grid.cell_of(grid.at) == expected
                or grid.tile(grid.state_of(*expected)) in "#X"
            )

    def test_a_wall_stops_the_move(self) -> None:
        grid = a_room()
        grid.reset()
        grid.at = grid.state_of(1, 2)
        grid.step(RIGHT)
        assert grid.cell_of(grid.at) == (1, 2)

    def test_the_edge_of_the_grid_stops_the_move(self) -> None:
        grid = GridWorld(Rng(1), ["S.G"])
        grid.reset()
        grid.step(LEFT)
        assert grid.at == 0

    def test_a_step_pays_the_step_reward(self) -> None:
        grid = a_room(step_reward=-2.0)
        grid.reset()
        assert grid.step(RIGHT).reward == -2.0

    def test_reaching_the_goal_ends_the_episode(self) -> None:
        grid = a_room()
        grid.reset()
        grid.at = grid.state_of(2, 4)
        outcome = grid.step(RIGHT)
        assert outcome.terminated
        assert not outcome.truncated

    def test_the_goal_pays_the_step_reward_when_no_goal_reward_is_set(self) -> None:
        # This is the cliff walk setting. The chapter counts steps, so the step
        # onto the goal costs what every other step costs.
        grid = a_room(step_reward=-1.0, goal_reward=None)
        grid.reset()
        grid.at = grid.state_of(2, 4)
        assert grid.step(RIGHT).reward == -1.0

    def test_a_goal_reward_replaces_the_step_reward(self) -> None:
        grid = a_room(step_reward=-1.0, goal_reward=5.0)
        grid.reset()
        grid.at = grid.state_of(2, 4)
        assert grid.step(RIGHT).reward == 5.0


class TestPits:
    def test_a_pit_pays_its_own_reward_and_not_the_step_reward(self) -> None:
        grid = a_room(step_reward=-1.0, pit_reward=-100.0)
        grid.reset()
        grid.at = grid.state_of(2, 2)
        assert grid.step(RIGHT).reward == -100.0

    def test_a_pit_sends_the_agent_back_to_the_start(self) -> None:
        grid = a_room()
        grid.reset()
        grid.at = grid.state_of(2, 2)
        outcome = grid.step(RIGHT)
        assert outcome.observation == grid.state_of(1, 1)
        assert not outcome.terminated

    def test_a_pit_can_end_the_episode_instead(self) -> None:
        grid = a_room(pit_ends_episode=True)
        grid.reset()
        grid.at = grid.state_of(2, 2)
        outcome = grid.step(RIGHT)
        assert outcome.terminated
        assert grid.cell_of(outcome.observation) == (2, 3)

    def test_the_audit_counts_the_pits_the_reward_hid(self) -> None:
        # On the cliff the agent is put back at the start, so the state it
        # lands in gives nothing away. The count has to be taken where the move
        # is resolved.
        grid = a_room()
        grid.reset()
        for _ in range(3):
            grid.at = grid.state_of(2, 2)
            grid.step(RIGHT)
        assert grid.audit() == {"pit_entries": 3.0}

    def test_a_pit_that_pays_what_a_step_pays_is_still_counted(self) -> None:
        # The frozen lake sets both to zero. Counting pits by looking at the
        # reward would report none of them.
        grid = a_room(step_reward=0.0, pit_reward=0.0, pit_ends_episode=True)
        grid.reset()
        grid.at = grid.state_of(2, 2)
        grid.step(RIGHT)
        assert grid.audit() == {"pit_entries": 1.0}

    def test_the_count_starts_again_with_the_episode(self) -> None:
        grid = a_room()
        grid.reset()
        grid.at = grid.state_of(2, 2)
        grid.step(RIGHT)
        grid.reset()
        assert grid.audit() == {"pit_entries": 0.0}


class TestWind:
    def test_the_wind_lifts_the_agent(self) -> None:
        grid = GridWorld(Rng(1), ["....", "S..G", "....", "...."], wind=[2, 0, 0, 0])
        grid.reset()
        grid.step(RIGHT)
        assert grid.cell_of(grid.at) == (0, 1)

    def test_the_wind_uses_the_column_the_agent_left(self) -> None:
        # Sutton's grid reads the wind of the column the agent is standing in
        # when it moves, not the one it lands in. On a grid where the two
        # differ, using the wrong one moves the whole solution.
        grid = GridWorld(Rng(1), ["....", "S..G", "....", "...."], wind=[0, 2, 0, 0])
        grid.reset()
        grid.step(RIGHT)
        assert grid.cell_of(grid.at) == (1, 1)

    def test_the_wind_cannot_push_the_agent_off_the_grid(self) -> None:
        # The row is held at the top rather than going negative, and the move
        # across still happens. Cancelling the move instead would make the top
        # row of a windy column unreachable in both directions.
        grid = GridWorld(Rng(1), ["S.G"], wind=[5, 5, 5])
        grid.reset()
        grid.step(RIGHT)
        assert grid.cell_of(grid.at) == (0, 1)

    def test_a_column_with_no_wind_lifts_nothing(self) -> None:
        grid = GridWorld(Rng(1), ["....", "S..G", "....", "...."], wind=[0, 0, 0, 0])
        grid.reset()
        grid.step(RIGHT)
        assert grid.cell_of(grid.at) == (1, 1)


class TestSlip:
    def test_a_slip_lands_on_a_move_at_right_angles(self) -> None:
        grid = GridWorld(Rng(2), ["...", "S.G", "..."], slip=1.0)
        landed = set()
        for _ in range(200):
            grid.reset()
            grid.step(RIGHT)
            landed.add(grid.cell_of(grid.at))
        assert landed == {(0, 0), (2, 0)}

    def test_the_slip_share_is_what_was_asked_for(self) -> None:
        grid = GridWorld(Rng(3), ["...", "S..", "..G"], slip=2.0 / 3.0)
        intended = 0
        draws = 6000
        for _ in range(draws):
            grid.reset()
            grid.at = grid.state_of(1, 1)
            grid.step(RIGHT)
            if grid.cell_of(grid.at) == (1, 2):
                intended += 1
        assert abs(intended / draws - 1.0 / 3.0) < 0.03

    def test_no_slip_means_no_slip(self) -> None:
        grid = GridWorld(Rng(4), ["...", "S..", "..G"], slip=0.0)
        for _ in range(50):
            grid.reset()
            grid.at = grid.state_of(1, 1)
            grid.step(RIGHT)
            assert grid.cell_of(grid.at) == (1, 2)


class TestDeterminism:
    def test_the_same_seed_gives_the_same_run(self) -> None:
        def walk(seed: int) -> list[tuple[int, float]]:
            grid = frozen_lake(Rng(seed).stream("env"))
            policy = Rng(seed).stream("policy")
            grid.reset()
            path = []
            for _ in range(300):
                outcome = grid.step(policy.below(4))
                path.append((outcome.observation, outcome.reward))
                if outcome.done:
                    grid.reset()
            return path

        assert walk(5) == walk(5)
        assert walk(5) != walk(6)

    def test_a_grid_with_no_chance_in_it_draws_nothing(self) -> None:
        # The cliff walk has one start, no slip and no wind, so nothing about
        # it is random once the episode has begun. Drawing anyway would be
        # harmless on its own and would break the promise that two agents meet
        # the same environment: the agent that explores more would move the
        # environment stream further.
        grid = cliff_walk(Rng(7).stream("env"))
        grid.reset()
        before = grid.rng.snapshot()
        for action in (0, 1, 2, 3, 1, 1, 0):
            grid.step(action)
        assert grid.rng.snapshot() == before


class TestRendering:
    def test_the_agent_is_drawn_on_the_grid(self) -> None:
        grid = a_room()
        grid.reset()
        assert grid.render().splitlines()[1] == "#@.#..#"

    def test_the_rest_of_the_grid_is_the_layout(self) -> None:
        grid = a_room()
        grid.reset()
        drawn = grid.render().splitlines()
        assert drawn[0] == ROOM[0]
        assert drawn[3] == ROOM[3]


PRESETS = {
    "cliff": cliff_walk,
    "windy": windy_grid,
    "rooms": four_rooms,
    "maze": dyna_maze,
    "lake4": frozen_lake,
}


class TestEveryPreset:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_the_transition_probabilities_add_up_to_one(self, name: str) -> None:
        grid = PRESETS[name](Rng(1))
        for state in grid.observation_space:
            for action in grid.action_space:
                total = sum(
                    outcome.probability for outcome in grid.transitions(state, action)
                )
                assert abs(total - 1.0) < 1e-9, f"{name} state {state} action {action}"

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_it_reaches_the_goal_under_some_policy(self, name: str) -> None:
        # An environment nobody can finish is a bug that no learning curve
        # reports: the return is simply flat, and that looks like an agent that
        # did not learn.
        grid = PRESETS[name](Rng(11))
        policy = Rng(11).stream("policy")
        reached = False
        for _ in range(400):
            grid.reset()
            for _ in range(grid.spec.max_episode_steps or 500):
                outcome = grid.step(policy.below(grid.action_space.n))
                if outcome.terminated and grid.is_goal(outcome.observation):
                    reached = True
                    break
                if outcome.done:
                    break
            if reached:
                break
        assert reached, f"a random policy never finished {name}"

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_the_terminal_states_are_the_ones_that_end_it(self, name: str) -> None:
        grid = PRESETS[name](Rng(1))
        terminal = grid.terminal_states()
        for state in terminal:
            assert (
                grid.is_goal(state)
                or (grid.pit_ends_episode and grid.is_pit(state))
                or grid.is_wall(state)
            )

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_it_renders(self, name: str) -> None:
        grid = PRESETS[name](Rng(1))
        grid.reset()
        drawn = grid.render().splitlines()
        assert len(drawn) == grid.height
        assert all(len(line) == grid.width for line in drawn)
        assert sum(line.count("@") for line in drawn) == 1

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_it_obeys_the_environment_contract(self, name: str) -> None:
        grid: Env[int, int] = PRESETS[name](Rng(1))
        assert grid.spec.name
        assert grid.spec.summary.endswith(".")
        with pytest.raises(RuntimeError):
            grid.step(0)


class TestTheModelAgreesWithTheSteps:
    """The model is a second description of the environment.

    Value iteration reads the model and reports the best possible return, and
    every claim in the documentation of the form "the agent reached this share
    of the best return" rests on the model being what the code does. So the
    code is driven, and what it did is compared with what it said it would do.

    What this test can catch, checked by putting each fault in on purpose:

    - Stepping and the model disagree about the slip.
    - The model overwrites a branch instead of adding to it, so two slips that
      land on the same cell are reported as one.
    - The model lists an outcome that never happens, or misses one that does.

    What it cannot catch, and why:

    - A fault inside `_slip_over`, `_gusts` or `_resolve`. Both stepping and
      the model call those, so a change there moves both answers together and
      they still agree. That is the point of sharing them, and it means this
      test is not the thing guarding them. The movement tests above are.
    - Anything about a terminal state. This test drives episodes, and an
      episode stops when it reaches one, so no step is ever taken from there.
      `test_every_state_it_stops_in_is_listed_as_terminal` covers that.
    """

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_what_happens_is_what_the_model_predicted(self, name: str) -> None:
        grid = PRESETS[name](Rng(21))
        policy = Rng(21).stream("policy")

        seen: dict[tuple[int, int], dict[tuple[int, float, bool], int]] = {}
        grid.reset()
        for _ in range(120_000):
            state = grid.at
            action = policy.below(grid.action_space.n)
            outcome = grid.step(action)

            key = (state, action)
            landed = (outcome.observation, outcome.reward, outcome.terminated)
            seen.setdefault(key, {})
            seen[key][landed] = seen[key].get(landed, 0) + 1

            if outcome.done:
                grid.reset()

        compared = 0
        for (state, action), tally in seen.items():
            draws = sum(tally.values())
            if draws < 200:
                continue

            model = {
                (outcome.observation, outcome.reward, outcome.terminated): (
                    outcome.probability
                )
                for outcome in grid.transitions(state, action)
            }

            for landed, count in tally.items():
                assert landed in model, (
                    f"{name}: state {state} action {action} reached {landed}, "
                    f"which the model does not list"
                )

                # Five standard errors of a share taken from this many draws.
                # A flat tolerance is the wrong shape here: the pairs near the
                # start are visited thousands of times and the ones in a corner
                # a few hundred, so one number is either loose where the data
                # is good or tight where the data is thin. This suite ran with
                # a flat 0.06 and failed on a share that was two and a half
                # standard errors out, which is a normal draw and not a defect.
                share = model[landed]
                error = (share * (1.0 - share) / draws) ** 0.5
                assert abs(count / draws - share) < 5.0 * error + 0.01, (
                    f"{name}: state {state} action {action} reached {landed} "
                    f"{count} times in {draws}, and the model says {share:.3f}"
                )
                compared += 1

            for landed, probability in model.items():
                if probability > 0.15:
                    assert landed in tally, (
                        f"{name}: the model promises {landed} from state "
                        f"{state} action {action}, and it never happened"
                    )

        assert compared > 20, f"{name}: too few pairs were visited to say anything"

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_every_state_it_stops_in_is_listed_as_terminal(self, name: str) -> None:
        # Value iteration takes `terminal_states` as the states it must not
        # back a value up through. A state that the environment stops in and
        # the model does not list gets a value from whatever its transitions
        # say, and the best possible return that the report prints is then
        # wrong by an amount nobody can see.
        grid = PRESETS[name](Rng(31))
        policy = Rng(31).stream("policy")
        listed = grid.terminal_states()

        stopped: set[int] = set()
        grid.reset()
        for _ in range(60_000):
            outcome = grid.step(policy.below(grid.action_space.n))
            if outcome.terminated:
                stopped.add(outcome.observation)
            if outcome.done:
                grid.reset()

        assert stopped, f"{name}: no episode ever ended inside the rules"
        assert stopped <= listed, (
            f"{name}: the environment stopped in {sorted(stopped - listed)}, "
            f"which terminal_states() does not list"
        )
