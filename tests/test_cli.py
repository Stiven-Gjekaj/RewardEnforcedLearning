"""Tests for the command line.

Every number in the documentation comes from one of these commands with a seed
next to it, so what is checked here is that a command with a seed gives the
same answer twice, and that a command with a mistake in it says what the
mistake was.
"""

from __future__ import annotations

import pytest

from rel.cli import main, parse_settings, parse_value


class TestParsingASetting:
    def test_a_whole_number_stays_a_whole_number(self) -> None:
        # An agent that receives the string "4" fails a long way from here.
        assert parse_value("4") == 4
        assert isinstance(parse_value("4"), int)

    def test_a_decimal_becomes_a_float(self) -> None:
        assert parse_value("0.1") == 0.1

    def test_a_negative_number_is_read(self) -> None:
        assert parse_value("-20") == -20

    def test_the_words_for_yes_and_no_become_booleans(self) -> None:
        assert parse_value("true") is True
        assert parse_value("False") is False
        assert parse_value("on") is True

    def test_none_becomes_none(self) -> None:
        assert parse_value("none") is None

    def test_anything_else_stays_a_string(self) -> None:
        assert parse_value("ordered") == "ordered"

    def test_settings_are_read_in_pairs(self) -> None:
        assert parse_settings(["a=1", "b=0.5"]) == {"a": 1, "b": 0.5}

    def test_spaces_around_the_equals_sign_are_dropped(self) -> None:
        assert parse_settings([" a = 1 "]) == {"a": 1}

    def test_a_setting_with_no_equals_sign_says_so(self) -> None:
        with pytest.raises(SystemExit, match="no equals sign"):
            parse_settings(["epsilon"])


class TestList:
    def test_it_names_every_environment_and_agent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["list", "--no-colour"]) == 0
        printed = capsys.readouterr().out
        for name in ("cliff", "boatrace", "q-learning", "tile-sarsa"):
            assert name in printed

    def test_the_settings_are_shown_when_asked_for(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["list", "--settings", "--no-colour"]) == 0
        printed = capsys.readouterr().out
        assert "step_size" in printed
        assert "planning_steps" in printed


class TestTrain:
    def test_a_short_run_reports(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "40",
                "--seed",
                "3",
                "--quiet",
                "--no-colour",
            ]
        )
        printed = capsys.readouterr().out
        assert code == 0
        assert "best possible" in printed
        assert "digest" in printed

    def test_the_same_seed_gives_the_same_digest(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def digest(seed: str) -> str:
            main(
                [
                    "train",
                    "sarsa",
                    "--env",
                    "cliff",
                    "--episodes",
                    "50",
                    "--seed",
                    seed,
                    "--quiet",
                    "--print",
                    "digest",
                ]
            )
            return capsys.readouterr().out.strip()

        assert digest("11") == digest("11")
        assert digest("11") != digest("12")

    def test_a_setting_reaches_the_agent(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def digest(epsilon: str) -> str:
            main(
                [
                    "train",
                    "q-learning",
                    "--env",
                    "cliff",
                    "--episodes",
                    "30",
                    "--quiet",
                    "--print",
                    "digest",
                    "--set",
                    f"epsilon={epsilon}",
                ]
            )
            return capsys.readouterr().out.strip()

        assert digest("0.0") != digest("0.5")

    def test_a_setting_for_the_environment_reaches_it(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def digest(slip: str) -> str:
            main(
                [
                    "train",
                    "q-learning",
                    "--env",
                    "cliff",
                    "--episodes",
                    "30",
                    "--quiet",
                    "--print",
                    "digest",
                    "--env-set",
                    f"slip={slip}",
                ]
            )
            return capsys.readouterr().out.strip()

        assert digest("0.0") != digest("0.3")

    def test_it_reports_when_the_greedy_policy_never_finishes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A run that ends with the step limit is not a run that scored the step
        # limit. Printing one number would hide that entirely.
        main(
            [
                "train",
                "sarsa",
                "--env",
                "cliff",
                "--episodes",
                "500",
                "--seed",
                "7",
                "--quiet",
                "--no-colour",
                "--no-maps",
            ]
        )
        printed = capsys.readouterr().out
        assert "greedy policy" in printed


class TestSolve:
    def test_it_gives_the_best_possible_return(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["solve", "--env", "cliff", "--no-colour"]) == 0
        assert "-13.0000" in capsys.readouterr().out

    def test_an_environment_with_no_model_says_so(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["solve", "--env", "cartpole"]) == 2
        assert "keeps no model" in capsys.readouterr().err

    def test_an_environment_that_never_ends_uses_its_own_discount(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The boat race has no finish line, so an undiscounted run of it is
        # worth an unbounded amount. The environment says which discount it
        # wants and the command line takes it.
        assert main(["solve", "--env", "boatrace", "--no-colour"]) == 0
        assert "0.99" in capsys.readouterr().out

    def test_forcing_a_discount_of_one_fails_with_an_explanation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["solve", "--env", "boatrace", "--discount", "1.0"]) == 2
        printed = capsys.readouterr().err
        assert "still moving" in printed
        assert "discount below one" in printed


class TestMistakes:
    def test_an_environment_that_does_not_exist_suggests_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["train", "q-learning", "--env", "clif", "--quiet"]) == 2
        assert "Did you mean 'cliff'" in capsys.readouterr().err

    def test_an_agent_that_does_not_exist_suggests_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["train", "q-learnin", "--env", "cliff", "--quiet"]) == 2
        assert "Did you mean 'q-learning'" in capsys.readouterr().err

    def test_a_setting_that_does_not_exist_is_named(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--quiet",
                "--set",
                "wobble=1",
            ]
        )
        assert code == 2
        assert "no setting named 'wobble'" in capsys.readouterr().err

    def test_an_agent_that_needs_a_model_says_so(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["train", "optimal", "--env", "cartpole", "--quiet"]) == 2
        assert "keeps no model" in capsys.readouterr().err

    def test_a_tile_coder_on_a_table_of_states_says_so(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["train", "tile-sarsa", "--env", "cliff", "--quiet"]) == 2
        assert "tile coder divides a Box" in capsys.readouterr().err


class TestCompare:
    def test_it_puts_the_agents_in_one_table(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            [
                "compare",
                "sarsa",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "60",
                "--runs",
                "2",
                "--no-colour",
            ]
        )
        printed = capsys.readouterr().out
        assert code == 0
        assert "sarsa" in printed
        assert "q-learning" in printed
        assert "last 100" in printed


class TestGaming:
    def test_it_shows_the_gap_for_all_three(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["gaming", "--no-learn", "--no-colour"]) == 0
        printed = capsys.readouterr().out
        assert "THE BOAT RACE" in printed
        assert "THE VASE ROOM" in printed
        assert "THE THERMOSTAT" in printed
        assert "LAPS" in printed
        assert "REALLY COMFORTABLE" in printed


def test_the_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    from rel import __version__

    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert __version__ in capsys.readouterr().out
