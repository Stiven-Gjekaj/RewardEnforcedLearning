"""Tests for a grid written in a text file rather than in Python.

A header is written by hand, so the fault a file really has is a typed line.
Most of these are about saying which line, and about refusing a setting the
environment does not take rather than dropping it quietly.
"""

from __future__ import annotations

import inspect

import pytest

from rel.envs.gridfile import SETTINGS, GridFileError, parse, read
from rel.envs.gridworld import GridWorld
from rel.rng import Rng

CLIFF = """\
name: cliff
summary: Twelve by four.
step_reward: -1
pit_reward: -100

............
............
............
SXXXXXXXXXXG
"""


class TestTheShapeOfAFile:
    def test_the_header_stops_at_the_first_blank_line(self) -> None:
        got = parse(CLIFF)
        assert got.settings["name"] == "cliff"
        assert len(got.layout) == 4

    def test_a_file_with_no_blank_line_is_all_layout(self) -> None:
        # A layout that starts with a row of walls would be read as a comment
        # if the two halves were told apart line by line.
        got = parse("#####\n#S.G#\n#####\n")
        assert got.layout == ("#####", "#S.G#", "#####")
        assert got.settings == {}

    def test_a_comment_in_the_header_is_ignored(self) -> None:
        got = parse("# the cliff walk\nname: cliff\n\nSG\n")
        assert got.settings == {"name": "cliff"}

    def test_trailing_blank_lines_are_not_rows(self) -> None:
        got = parse("name: x\n\nSG\n\n\n")
        assert got.layout == ("SG",)

    def test_a_header_with_no_layout_is_refused(self) -> None:
        with pytest.raises(GridFileError, match="no layout under it"):
            parse("name: cliff\n\n\n")

    def test_a_trailing_space_is_kept(self) -> None:
        # A space is a floor tile, so padding a short line would open a wall
        # that an editor had trimmed. The rows have to be written as they are.
        got = parse("name: x\n\n#S #\n#.G#\n")
        assert got.layout == ("#S #", "#.G#")

    def test_a_trimmed_row_is_refused_rather_than_padded(self) -> None:
        got = parse("name: x\n\n#S#\n#G\n")
        with pytest.raises(ValueError, match="same width"):
            got.build(Rng(1))


class TestTheSettings:
    def test_a_number_comes_back_as_a_number(self) -> None:
        assert parse(CLIFF).settings["step_reward"] == -1.0

    def test_a_name_stays_text_even_when_it_looks_like_a_number(self) -> None:
        # A reader that guessed from the text would hand the environment a
        # name it cannot print.
        assert parse("name: 12\n\nSG\n").settings["name"] == "12"

    def test_true_and_false_are_read_as_such(self) -> None:
        assert parse("king_moves: yes\n\nSG\n").settings["king_moves"] is True
        assert parse("king_moves: off\n\nSG\n").settings["king_moves"] is False

    def test_the_wind_is_one_number_for_each_column(self) -> None:
        assert parse("wind: 0 0 1 2\n\nSG\n").settings["wind"] == (0, 0, 1, 2)

    def test_a_setting_whose_default_is_nothing_can_be_set_to_nothing(self) -> None:
        assert parse("goal_reward: none\n\nSG\n").settings["goal_reward"] is None

    def test_an_unknown_setting_is_refused(self) -> None:
        with pytest.raises(GridFileError, match="'gravity' is not a grid setting"):
            parse("gravity: 9.8\n\nSG\n")

    def test_a_setting_written_twice_is_refused(self) -> None:
        with pytest.raises(GridFileError, match="'slip' is set twice"):
            parse("slip: 0.1\nslip: 0.2\n\nSG\n")

    def test_a_header_line_that_is_not_a_setting_is_refused(self) -> None:
        with pytest.raises(GridFileError, match="is neither"):
            parse("just some words\n\nSG\n")

    def test_a_value_of_the_wrong_kind_is_refused(self) -> None:
        with pytest.raises(GridFileError, match="could not convert"):
            parse("slip: sideways\n\nSG\n")

    def test_a_bool_that_is_neither_is_refused(self) -> None:
        with pytest.raises(GridFileError, match="not true or false"):
            parse("king_moves: maybe\n\nSG\n")


class TestNamingWhereItWentWrong:
    def test_the_line_number_is_in_the_message(self) -> None:
        # A header is written by hand, so the fault it has is a typed line.
        # A message that did not say which one leaves the reader counting.
        with pytest.raises(GridFileError, match="line 3"):
            parse("name: x\nslip: 0.1\ngravity: 9.8\n\nSG\n")

    def test_the_file_is_named_too(self) -> None:
        with pytest.raises(GridFileError, match=r"grids/cliff\.txt, line 1"):
            parse("gravity: 9.8\n\nSG\n", where="grids/cliff.txt")

    def test_a_file_that_is_not_there_says_so(self) -> None:
        with pytest.raises(GridFileError, match="cannot be read"):
            read("nowhere/at/all.txt")


class TestBuilding:
    def test_it_builds_the_grid_the_file_describes(self) -> None:
        env = parse(CLIFF).build(Rng(1))
        assert env.spec.name == "cliff"
        assert env.height == 4
        assert env.width == 12
        assert env.pit_reward == -100.0

    def test_an_override_wins_over_the_file(self) -> None:
        # What lets the command line change one setting of a grid somebody
        # else wrote.
        env = parse(CLIFF).build(Rng(1), slip=0.5)
        assert env.slip == 0.5

    def test_a_default_is_left_alone(self) -> None:
        assert parse(CLIFF).build(Rng(1)).slip == 0.0


class TestTheSettingsMatchTheEnvironment:
    def test_every_setting_is_one_the_grid_really_takes(self) -> None:
        taken = set(inspect.signature(GridWorld.__init__).parameters)
        assert set(SETTINGS) <= taken

    def test_every_argument_the_grid_takes_can_be_written_in_a_file(self) -> None:
        # A setting added to the environment and forgotten here would be one
        # a file could not reach, with nothing saying so.
        taken = inspect.signature(GridWorld.__init__).parameters
        keyword = {
            name
            for name, parameter in taken.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert keyword == set(SETTINGS)
