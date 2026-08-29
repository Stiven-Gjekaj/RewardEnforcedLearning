"""Tests for the drawing.

Everything here returns a string, so everything here can be compared. The one
part that writes, `rel.ui.live`, is given a buffer rather than a terminal.
"""

from __future__ import annotations

import io

import pytest

from rel.agents.base import Agent, RandomAgent
from rel.agents.dp import value_iteration
from rel.agents.td import QLearning
from rel.envs.classic import cliff_walk
from rel.rng import Rng
from rel.spaces import Discrete
from rel.ui.chart import (
    BAND,
    MARK,
    _winner,
    bar,
    histogram,
    line_chart,
    smooth,
    sparkline,
)
from rel.ui.colour import (
    CODES,
    Palette,
    pad,
    palette_for,
    terminal_wants_colour,
    width_of,
)
from rel.ui.grid import best_values, difference_map, policy_map, value_map
from rel.ui.live import CLEAR, HIDE, SHOW, Live
from rel.ui.table import pairs, table


class TestColour:
    def test_a_palette_that_is_off_changes_nothing(self) -> None:
        assert Palette(enabled=False).paint("hello", "red") == "hello"

    def test_a_palette_that_is_on_wraps_the_text(self) -> None:
        painted = Palette(enabled=True).paint("hello", "red")
        assert painted.startswith("\x1b[31m")
        assert painted.endswith("\x1b[0m")
        assert "hello" in painted

    def test_an_unknown_colour_is_refused(self) -> None:
        with pytest.raises(KeyError, match="no colour named"):
            Palette(enabled=True).paint("hello", "puce")

    def test_no_color_switches_it_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The convention at no-color.org is that the variable counts whatever
        # its value is, including an empty one. Checking the value rather than
        # the presence is the usual way to get this wrong.
        monkeypatch.setenv("NO_COLOR", "")
        assert not terminal_wants_colour(io.StringIO())

    def test_a_dumb_terminal_switches_it_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert not terminal_wants_colour(io.StringIO())

    def test_something_that_is_not_a_terminal_switches_it_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        assert not terminal_wants_colour(io.StringIO())

    def test_the_caller_can_say_outright(self) -> None:
        assert palette_for(io.StringIO(), forced=True).enabled
        assert not palette_for(io.StringIO(), forced=False).enabled


class TestWidth:
    def test_the_width_ignores_the_colour_codes(self) -> None:
        # `str.ljust` counts the escape codes, so a coloured cell in a table
        # comes out several columns short and every column after it walks left.
        painted = Palette(enabled=True).paint("abc", "red")
        assert len(painted) > 3
        assert width_of(painted) == 3

    def test_padding_uses_the_printed_width(self) -> None:
        painted = Palette(enabled=True).paint("abc", "red")
        assert width_of(pad(painted, 8)) == 8

    def test_padding_can_go_the_other_way(self) -> None:
        assert pad("ab", 5, "right") == "   ab"
        assert pad("ab", 6, "centre") == "  ab  "

    def test_text_wider_than_the_column_is_left_alone(self) -> None:
        assert pad("abcdef", 3) == "abcdef"


class TestSparkline:
    def test_it_is_as_wide_as_it_was_asked_for(self) -> None:
        assert len(sparkline(list(range(100)), 20)) == 20

    def test_a_rising_series_rises(self) -> None:
        drawn = sparkline([float(value) for value in range(8)], 8)
        assert drawn[0] == " "
        assert drawn[-1] == "█"

    def test_a_flat_series_is_drawn_flat(self) -> None:
        drawn = sparkline([3.0] * 10, 10)
        assert len(set(drawn)) == 1

    def test_nothing_gives_nothing(self) -> None:
        assert sparkline([], 10) == ""


class TestLineChart:
    def test_it_has_the_shape_it_was_asked_for(self) -> None:
        lines = line_chart({"a": list(range(50))}, width=30, height=8)
        # Eight rows of chart and the axis under them.
        assert len(lines) == 9
        assert lines[-1].strip().startswith("+")

    def test_it_draws_something(self) -> None:
        lines = line_chart({"a": [float(v) for v in range(50)]}, width=30, height=8)
        drawn = "".join(lines)
        assert any("⠀" <= character <= "⣿" for character in drawn)

    def test_a_mark_is_drawn_and_named(self) -> None:
        lines = line_chart(
            {"a": [1.0, 2.0, 3.0]}, width=20, height=6, marks={"best": 2.5}
        )
        assert "best = 2.5" in lines[-1]

    def test_nothing_to_draw_says_so(self) -> None:
        assert line_chart({}) == ["(nothing to draw)"]
        assert line_chart({"a": []}) == ["(nothing to draw)"]

    def test_a_flat_series_does_not_divide_by_zero(self) -> None:
        lines = line_chart({"a": [5.0] * 20}, width=20, height=6)
        assert len(lines) == 7

    def test_a_steep_step_is_drawn_as_a_line(self) -> None:
        # Without filling in between, a curve that jumps looks like a scatter
        # of dots, and a reader takes the gaps for missing data.
        lines = line_chart({"a": [0.0, 100.0]}, width=8, height=6)
        filled = sum(
            1 for line in lines for character in line if "⠁" <= character <= "⣿"
        )
        assert filled > 4

    def test_clipping_marks_the_bottom_label(self) -> None:
        values = [-1000.0] + [1.0] * 100
        lines = line_chart({"a": values}, width=20, height=6, clip=0.05)
        bottom = lines[-2] if len(lines) > 6 else lines[5]
        assert "<" in bottom

    def test_without_clipping_the_axis_reaches_the_lowest_value(self) -> None:
        values = [-1000.0] + [1.0] * 100
        lines = line_chart({"a": values}, width=20, height=6, clip=0.0)
        assert not any("<" in line for line in lines)


class TestTheBand:
    """The spread of a run, drawn behind the mean of it.

    A mean line on its own says nothing about how much the runs behind it
    disagreed, and in this project that has been the wrong answer twice.
    """

    MEAN = tuple(float(index) for index in range(40))

    def _drawn(self, width: float, **extra: object) -> list[str]:
        low = [value - width for value in self.MEAN]
        high = [value + width for value in self.MEAN]
        return line_chart(
            {"run": self.MEAN},
            bands={"run": (low, high)},
            width=40,
            height=10,
            **extra,  # type: ignore[arg-type]
        )

    def test_the_band_reaches_the_axis(self) -> None:
        # A band drawn outside the range would be flattened against the top
        # and the bottom, and the picture would understate the spread.
        top = self._drawn(10.0)[0].split("|")[0].strip()
        assert float(top) >= 49.0

    def test_the_band_widens_with_the_spread(self) -> None:
        narrow = float(self._drawn(1.0)[0].split("|")[0].strip())
        wide = float(self._drawn(20.0)[0].split("|")[0].strip())
        assert wide > narrow

    def test_it_is_drawn_with_no_terminal(self) -> None:
        lines = self._drawn(5.0)
        assert len(lines) >= 10
        assert all("\x1b" not in line for line in lines)
        assert any(character != " " for line in lines for character in line)

    def test_a_series_with_no_band_is_drawn_anyway(self) -> None:
        lines = line_chart(
            {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]},
            bands={"a": ([0.0, 1.0, 2.0], [2.0, 3.0, 4.0])},
            width=20,
            height=6,
        )
        assert len(lines) >= 6

    def test_a_band_for_a_series_that_is_not_drawn_is_ignored(self) -> None:
        lines = line_chart(
            {"a": [1.0, 2.0, 3.0]},
            bands={"gone": ([0.0], [100.0])},
            width=20,
            height=6,
        )
        top = float(lines[0].split("|")[0].strip())
        assert top < 100.0

    def test_the_curve_keeps_a_cell_the_band_also_wants(self) -> None:
        # A band fills a column of dots and a curve fills one or two, so most
        # dots wins would put the band over the line it is the spread of.
        assert _winner({0: 1, BAND: 8}) == 0
        assert _winner({MARK: 1, BAND: 8}) == MARK
        assert _winner({BAND: 8}) == BAND

    def test_a_band_is_painted_in_its_own_colour_dimmed(self) -> None:
        painted = Palette(enabled=True).faint_series("x", 0)
        assert painted.startswith(CODES["dim"])
        assert CODES["cyan"] in painted

    def test_an_unknown_colour_is_refused_when_dimmed_too(self) -> None:
        with pytest.raises(KeyError, match="no colour named"):
            Palette(enabled=True).faint("x", "puce")


class TestBarsAndHistograms:
    def test_a_bar_is_as_wide_as_the_share(self) -> None:
        assert bar(0.0, 0.0, 10.0, 10).strip() == ""
        assert bar(10.0, 0.0, 10.0, 10) == "█" * 10
        assert len(bar(5.0, 0.0, 10.0, 10)) == 10

    def test_a_bar_outside_the_range_is_held_inside(self) -> None:
        assert bar(99.0, 0.0, 10.0, 6) == "█" * 6
        assert bar(-99.0, 0.0, 10.0, 6).strip() == ""

    def test_a_histogram_has_one_line_for_each_bucket(self) -> None:
        assert len(histogram([float(v) for v in range(100)], buckets=5)) == 5

    def test_a_histogram_of_nothing_is_nothing(self) -> None:
        assert histogram([]) == []


class TestSmooth:
    def test_the_length_does_not_change(self) -> None:
        assert len(smooth([1.0] * 30, 10)) == 30

    def test_it_averages_the_window(self) -> None:
        assert smooth([0.0, 10.0, 20.0], 2) == [0.0, 5.0, 15.0]

    def test_a_window_of_one_changes_nothing(self) -> None:
        assert smooth([1.0, 5.0, 3.0], 1) == [1.0, 5.0, 3.0]

    def test_a_window_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError):
            smooth([1.0], 0)


class TestTable:
    def test_the_columns_line_up(self) -> None:
        lines = table(["a", "bb"], [["1", "2"], ["333", "4"]])
        # Heading, rule, and then one line for each row.
        assert lines[0].startswith("a  ")
        assert set(lines[1]) == {"-", " "}
        assert lines[3].startswith("333")

    def test_a_coloured_cell_does_not_move_the_next_column(self) -> None:
        palette = Palette(enabled=True)
        plain = table(["a", "b"], [["xx", "y"]])
        coloured = table(["a", "b"], [[palette.paint("xx", "red"), "y"]])
        assert [width_of(line) for line in plain] == [
            width_of(line) for line in coloured
        ]

    def test_a_row_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2 cells"):
            table(["a", "b", "c"], [["1", "2"]])

    def test_an_alignment_for_each_column_is_needed(self) -> None:
        with pytest.raises(ValueError, match="one alignment"):
            table(["a", "b"], [], align=["left"])

    def test_pairs_line_up_their_names(self) -> None:
        lines = pairs([("short", "1"), ("much longer", "2")])
        assert lines[0].index("1") == lines[1].index("2")

    def test_pairs_of_nothing_is_nothing(self) -> None:
        assert pairs([]) == []


class TestGridPictures:
    @staticmethod
    def _trained() -> tuple[object, QLearning[int]]:
        rng = Rng(7)
        env = cliff_walk(rng.stream("env"))
        agent: QLearning[int] = QLearning(
            rng.stream("agent"), env.action_space, step_size=0.5, epsilon=0.1
        )
        from rel.training import train

        train(env, agent, 200)
        return env, agent

    def test_a_policy_map_has_the_shape_of_the_grid(self) -> None:
        env, agent = self._trained()
        lines = policy_map(env, agent).splitlines()  # type: ignore[arg-type]
        assert len(lines) == env.height  # type: ignore[attr-defined]
        assert all(len(line) == env.width for line in lines)  # type: ignore[attr-defined]

    def test_a_map_of_an_agent_that_only_takes_actions_has_no_extra_line(
        self,
    ) -> None:
        env, agent = self._trained()
        lines = policy_map(env, agent).splitlines()  # type: ignore[arg-type]
        assert "runs on" not in lines[-1]

    def test_a_choice_that_runs_on_is_marked_and_counted(self) -> None:
        # The picture of an agent over options is misleading without this. It
        # draws one action per cell and the agent does not follow one action
        # per cell: it commits to an option and runs it.
        from rel.agents.options import OptionsQ
        from rel.envs.classic import four_rooms
        from rel.options import hallway_options, primitives
        from rel.training import train

        rng = Rng(1)
        env = four_rooms(rng.stream("env"))
        agent = OptionsQ(
            rng.stream("agent"),
            env.action_space,
            [
                *primitives(env.action_space, range(env.observation_space.n)),
                *hallway_options(env, env.gaps()),
            ],
            step_size=0.5,
            discount=0.95,
        )
        train(env, agent, 200, discount=0.95)

        lines = policy_map(env, agent).splitlines()
        assert "cells where the choice runs on" in lines[-1]
        assert len(lines) == env.height + 1

    def test_the_walls_and_the_goal_are_drawn_as_themselves(self) -> None:
        env, agent = self._trained()
        drawn = policy_map(env, agent)  # type: ignore[arg-type]
        assert "G" in drawn
        assert "X" in drawn

    def test_drawing_the_policy_does_not_teach_the_agent_anything(self) -> None:
        # Reading has to be free of side effects. With the writing accessor
        # behind the renderer, drawing the picture put a row in the table for
        # every cell, and the value map then reported that the agent had stood
        # in all eleven cliff cells.
        env, agent = self._trained()
        before = dict(agent.q)
        policy_map(env, agent)  # type: ignore[arg-type]
        value_map(env, best_values(env, agent))  # type: ignore[arg-type]
        assert set(agent.q) == set(before)

    def test_a_cell_never_stood_in_is_not_drawn_as_the_best_cell(self) -> None:
        env, agent = self._trained()
        drawn = value_map(env, best_values(env, agent))  # type: ignore[arg-type]
        assert "?" in drawn
        assert "never stood in" in drawn

    def test_a_difference_map_counts_the_cells_that_differ(self) -> None:
        env, agent = self._trained()
        best = value_iteration(env)  # type: ignore[arg-type]
        drawn = difference_map(env, agent, best.policy)  # type: ignore[arg-type]
        assert "differ from the best policy" in drawn

    def test_an_agent_with_no_numbers_cannot_be_drawn_as_a_value_map(self) -> None:
        from rel.agents.base import RandomAgent

        env, _ = self._trained()
        nothing: RandomAgent[int] = RandomAgent(Rng(1), env.action_space)  # type: ignore[attr-defined]
        with pytest.raises(TypeError, match="keeps no value"):
            best_values(env, nothing)  # type: ignore[arg-type]


class TestLive:
    def test_it_writes_nothing_when_it_is_not_a_terminal(self) -> None:
        buffer = io.StringIO()
        display = Live(buffer, enabled=False)
        with display:
            display.update(["one", "two"], force=True)
        assert buffer.getvalue() == ""

    def test_it_writes_the_lines_when_it_is_a_terminal(self) -> None:
        buffer = io.StringIO()
        display = Live(buffer, enabled=True)
        with display:
            display.update(["one", "two"], force=True)
        assert "one\ntwo\n" in buffer.getvalue()

    def test_it_hides_and_shows_the_cursor(self) -> None:
        buffer = io.StringIO()
        with Live(buffer, enabled=True):
            pass
        assert buffer.getvalue() == HIDE + SHOW

    def test_a_second_draw_erases_the_first(self) -> None:
        buffer = io.StringIO()
        display = Live(buffer, enabled=True)
        with display:
            display.update(["one"], force=True)
            display.update(["two"], force=True)
        assert CLEAR in buffer.getvalue()

    def test_it_does_not_redraw_faster_than_it_was_told_to(self) -> None:
        # A learning curve does not need sixty frames a second, and drawing one
        # for every episode of a fast environment costs more than the run.
        buffer = io.StringIO()
        display = Live(buffer, enabled=True, every=1000.0)
        with display:
            display.update(["one"], force=True)
            display.update(["two"])
        assert "two" not in buffer.getvalue()

    def test_finish_writes_even_without_a_terminal(self) -> None:
        buffer = io.StringIO()
        display = Live(buffer, enabled=False)
        display.finish(["done"])
        assert buffer.getvalue() == "done\n"


class TestWhatAStateIsWorth:
    """`Agent.state_value`, which the value map is drawn from."""

    def test_an_agent_that_ranks_actions_gives_the_best_of_them(self) -> None:
        agent: QLearning[int] = QLearning(Rng(1), Discrete(3))
        agent.values(0)[:] = [1.0, 5.0, 2.0]
        assert agent.state_value(0) == 5.0

    def test_an_agent_that_keeps_no_numbers_says_nothing(self) -> None:
        assert RandomAgent(Rng(1), Discrete(2)).state_value(0) is None

    def test_the_value_map_asks_for_it(self) -> None:
        # An agent that predicts rather than controls keeps one number per
        # state and no action values at all. Drawing its map off the best
        # action value would leave the agents whose whole job is the map
        # unable to draw one.
        env, agent = self._map_pair()
        drawn = value_map(env, best_values(env, agent))  # type: ignore[arg-type]
        assert drawn.splitlines()

    @staticmethod
    def _map_pair() -> tuple[object, Agent[int]]:
        rng = Rng(1)
        env = cliff_walk(rng.stream("env"))

        class OneNumberPerState(RandomAgent[int]):
            def state_value(self, observation: int) -> float:
                return float(observation)

            def knows(self, observation: int) -> bool:
                return True

        return env, OneNumberPerState(rng.stream("agent"), env.action_space)
