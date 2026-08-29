"""Tests for the command line.

Every number in the documentation comes from one of these commands with a seed
next to it, so what is checked here is that a command with a seed gives the
same answer twice, and that a command with a mistake in it says what the
mistake was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rel.cli import main, parse_over, parse_settings, parse_value, sweep_settings


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


class TestAGridFromAFile:
    """`--env-file` in place of `--env`."""

    LAYOUT = (
        "# The cliff walk of Sutton and Barto, example 6.6.\n"
        "name: cliff\n"
        "summary: Twelve by four.\n"
        "step_reward: -1\n"
        "pit_reward: -100\n"
        "max_episode_steps: 500\n"
        "\n"
        "............\n"
        "............\n"
        "............\n"
        "SXXXXXXXXXXG\n"
    )

    def _written(self, tmp_path: Path) -> str:
        path = tmp_path / "cliff.txt"
        path.write_text(self.LAYOUT, encoding="utf-8")
        return str(path)

    def test_it_builds_a_grid_from_a_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(["solve", "--env-file", self._written(tmp_path), "--no-colour"]) == 0
        )
        assert "best possible return  -13.0000" in capsys.readouterr().out

    def test_the_file_and_the_built_in_grid_agree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The same layout and the same rewards, so the same exact answer.
        main(["solve", "--env-file", self._written(tmp_path), "--no-colour"])
        from_file = capsys.readouterr().out
        main(["solve", "--env", "cliff", "--no-colour"])
        built_in = capsys.readouterr().out

        def numbers(text: str) -> list[str]:
            return [
                line.split()[-1]
                for line in text.splitlines()
                if line.startswith(("states", "actions", "sweeps", "best possible"))
            ]

        assert numbers(from_file) == numbers(built_in)

    def test_an_agent_can_be_trained_on_one(self, tmp_path: Path) -> None:
        assert (
            main(
                [
                    "train",
                    "q-learning",
                    "--env-file",
                    self._written(tmp_path),
                    "--episodes",
                    "20",
                    "--quiet",
                ]
            )
            == 0
        )

    def test_a_setting_on_the_command_line_wins_over_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "solve",
                "--env-file",
                self._written(tmp_path),
                "--env-set",
                "step_reward=-2",
                "--no-colour",
            ]
        )
        assert "best possible return  -26.0000" in capsys.readouterr().out

    def test_naming_both_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["solve", "--env", "cliff", "--env-file", self._written(tmp_path)])

    def test_naming_neither_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            main(["solve"])

    def test_a_bad_header_names_the_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "wrong.txt"
        path.write_text("name: x\ngravity: 9.8\n\nSG\n", encoding="utf-8")
        assert main(["solve", "--env-file", str(path), "--no-colour"]) == 2
        assert "line 2" in capsys.readouterr().err

    def test_a_file_that_is_not_there_says_so(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["solve", "--env-file", "nowhere.txt", "--no-colour"]) == 2
        assert "cannot be read" in capsys.readouterr().err


class TestSweeping:
    """`rel sweep`, which varies one or two settings and prints the table."""

    def test_the_values_go_through_the_same_reader_as_set(self) -> None:
        # A sweep that handed an agent the string "0.1" would fail somewhere
        # far away from here.
        assert parse_over(["step_size=0.1,0.5"]) == [("step_size", [0.1, 0.5])]
        assert parse_over(["n=1,2,4"]) == [("n", [1, 2, 4])]
        assert parse_over(["hallways=on,off"]) == [("hallways", [True, False])]

    def test_spaces_around_the_values_are_dropped(self) -> None:
        assert parse_over([" n = 1, 2 "]) == [("n", [1, 2])]

    def test_naming_no_values_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="names no values"):
            parse_over(["step_size="])

    def test_one_setting_gives_one_run_for_each_value(self) -> None:
        assert sweep_settings([("n", [1, 2, 3])]) == [{"n": 1}, {"n": 2}, {"n": 3}]

    def test_two_settings_give_every_pair(self) -> None:
        # The point of sweeping two. The settings in this project trade
        # against each other, and varying them one at a time misses that.
        assert sweep_settings([("n", [1, 2]), ("step_size", [0.1, 0.5])]) == [
            {"n": 1, "step_size": 0.1},
            {"n": 1, "step_size": 0.5},
            {"n": 2, "step_size": 0.1},
            {"n": 2, "step_size": 0.5},
        ]

    def test_the_first_setting_varies_slowest(self) -> None:
        pairs = sweep_settings([("a", [1, 2]), ("b", [1, 2])])
        assert [chosen["a"] for chosen in pairs] == [1, 1, 2, 2]

    def test_no_setting_at_all_gives_one_empty_run(self) -> None:
        assert sweep_settings([]) == [{}]

    def test_it_builds_one_agent_for_each_value(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                [
                    "sweep",
                    "q-learning",
                    "--env",
                    "cliff",
                    "--over",
                    "step_size=0.1,0.5",
                    "--episodes",
                    "20",
                    "--runs",
                    "2",
                    "--no-colour",
                ]
            )
            == 0
        )
        printed = capsys.readouterr().out
        rows = [
            line
            for line in printed.splitlines()
            if line.lstrip().startswith(("0.1 ", "0.5 "))
        ]
        assert len(rows) == 2

    def test_the_table_names_every_setting_it_swept(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "sweep",
                "n-step-sarsa",
                "--env",
                "cliff",
                "--over",
                "n=1,2",
                "--over",
                "step_size=0.5",
                "--episodes",
                "20",
                "--runs",
                "1",
                "--no-colour",
            ]
        )
        printed = capsys.readouterr().out
        heading = next(
            line for line in printed.splitlines() if line.startswith("n  step_size")
        )
        assert "last 100" in heading
        assert "exact value" in heading

    def test_two_settings_give_a_row_for_every_pair(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "sweep",
                "n-step-sarsa",
                "--env",
                "cliff",
                "--over",
                "n=1,2",
                "--over",
                "step_size=0.1,0.5",
                "--episodes",
                "20",
                "--runs",
                "1",
                "--no-colour",
            ]
        )
        printed = capsys.readouterr().out
        rows = [
            line
            for line in printed.splitlines()
            if line.lstrip().startswith(("1  ", "2  "))
        ]
        assert len(rows) == 4

    def test_every_seed_can_be_shown(self, capsys: pytest.CaptureFixture[str]) -> None:
        # A mean over runs that vary a great deal has been the wrong answer
        # twice in this project already.
        main(
            [
                "sweep",
                "q-learning",
                "--env",
                "cliff",
                "--over",
                "step_size=0.5",
                "--episodes",
                "20",
                "--runs",
                "3",
                "--each-seed",
                "--no-colour",
            ]
        )
        printed = capsys.readouterr().out
        assert "every seed" in printed
        row = next(
            line for line in printed.splitlines() if line.lstrip().startswith("0.5 ")
        )
        # The setting, four columns of summary, then one number per seed.
        assert len(row.split()) >= 3 + 3

    def test_a_setting_both_fixed_and_swept_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="both fixed with --set"):
            main(
                [
                    "sweep",
                    "q-learning",
                    "--env",
                    "cliff",
                    "--over",
                    "step_size=0.1",
                    "--set",
                    "step_size=0.5",
                ]
            )

    def test_sweeping_nothing_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="at least one --over"):
            main(["sweep", "q-learning", "--env", "cliff"])

    def test_it_works_on_a_grid_from_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cliff.txt"
        path.write_text(TestAGridFromAFile.LAYOUT, encoding="utf-8")
        assert (
            main(
                [
                    "sweep",
                    "q-learning",
                    "--env-file",
                    str(path),
                    "--over",
                    "step_size=0.5",
                    "--episodes",
                    "20",
                    "--runs",
                    "1",
                    "--no-colour",
                ]
            )
            == 0
        )


class TestTheBandOnTheCompareChart:
    def test_the_band_is_on_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                [
                    "compare",
                    "q-learning",
                    "--env",
                    "cliff",
                    "--episodes",
                    "20",
                    "--runs",
                    "3",
                    "--no-colour",
                ]
            )
            == 0
        )
        assert "best and worst seed" in capsys.readouterr().out

    def test_it_can_be_switched_off(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(
            [
                "compare",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--runs",
                "3",
                "--no-band",
                "--no-colour",
            ]
        )
        assert "best and worst seed" not in capsys.readouterr().out

    def test_one_seed_has_no_spread_to_draw(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A band around a single run would be the run.
        main(
            [
                "compare",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--runs",
                "1",
                "--no-colour",
            ]
        )
        assert "best and worst seed" not in capsys.readouterr().out


class TestRecordingARun:
    """`--out` on `rel train`, and `rel replay` reading it back."""

    def test_a_run_can_be_written_to_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "run.txt"
        assert (
            main(
                [
                    "train",
                    "q-learning",
                    "--env",
                    "cliff",
                    "--episodes",
                    "20",
                    "--quiet",
                    "--out",
                    str(path),
                ]
            )
            == 0
        )
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("rel-run 1")

    def test_recording_does_not_change_the_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The recorder stands in for the digest and nothing else moves, so a
        # recorded run and an unrecorded one on the same seed take the same
        # path through the environment.
        path = tmp_path / "run.txt"
        main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--quiet",
                "--out",
                str(path),
            ]
        )
        capsys.readouterr()

        main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--quiet",
                "--print",
                "digest",
            ]
        )
        unrecorded = capsys.readouterr().out.strip()

        from rel.recording import read_run

        assert read_run(path).header["digest"] == unrecorded

    def test_the_header_says_what_was_run(self, tmp_path: Path) -> None:
        from rel.recording import read_run

        path = tmp_path / "run.txt"
        main(
            [
                "train",
                "sarsa",
                "--env",
                "cliff",
                "--episodes",
                "10",
                "--seed",
                "3",
                "--quiet",
                "--out",
                str(path),
            ]
        )
        header = read_run(path).header
        assert header["env"] == "cliff"
        assert header["agent"] == "sarsa"
        assert header["seed"] == "3"
        assert header["episodes"] == "10"

    def test_a_name_ending_in_gz_is_compressed(self, tmp_path: Path) -> None:
        path = tmp_path / "run.txt.gz"
        main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--quiet",
                "--out",
                str(path),
            ]
        )
        assert path.read_bytes()[:2] == b"\x1f\x8b"

    def test_replay_draws_what_the_run_drew(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "run.txt"
        main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "40",
                "--quiet",
                "--out",
                str(path),
                "--no-colour",
            ]
        )
        capsys.readouterr()

        assert main(["replay", str(path), "--no-colour"]) == 0
        printed = capsys.readouterr().out
        assert "q-learning on cliff" in printed
        assert "episodes             40" in printed

    def test_a_compressed_file_replays(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "run.txt.gz"
        main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--quiet",
                "--out",
                str(path),
            ]
        )
        capsys.readouterr()
        assert main(["replay", str(path), "--no-colour"]) == 0
        assert "episodes             20" in capsys.readouterr().out

    def test_a_changed_file_is_refused_rather_than_drawn(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "run.txt"
        main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--quiet",
                "--out",
                str(path),
            ]
        )

        text = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(text):
            if "|" in line:
                fields = line.split("|")
                fields[2] = "-2"
                text[index] = "|".join(fields)
                break
        path.write_text("\n".join(text) + "\n", encoding="utf-8")

        capsys.readouterr()
        assert main(["replay", str(path), "--no-colour"]) == 2
        assert "changed since it was written" in capsys.readouterr().err

    def test_a_file_that_is_not_there_says_so(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["replay", "nowhere.txt", "--no-colour"]) == 2
        assert "cannot be read" in capsys.readouterr().err


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

    def test_an_environment_with_no_ending_is_not_reported_as_stuck(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Every episode of a bandit runs to its step limit, because there is
        # nothing to reach. Saying "twenty of twenty ran out of steps" about
        # that would be describing the environment rather than the policy.
        main(
            [
                "train",
                "bandit-ucb",
                "--env",
                "bandit",
                "--episodes",
                "5",
                "--quiet",
                "--no-colour",
                "--env-set",
                "pulls=200",
            ]
        )
        assert "ran out of steps" not in capsys.readouterr().out

    def test_a_policy_that_stops_moving_is_reported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The same line on an environment that can end means the policy
        # stopped moving, which is worth saying loudly. This is the seed where
        # SARSA on the cliff walk does it.
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
        assert "ran out of steps" in printed
        assert "never reaches an ending" in printed


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


class TestDemo:
    def test_it_leaves_the_last_frame_behind_without_a_terminal(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Moving a cursor in a pipe writes the escape codes into the pipe, so
        # nothing is drawn as it goes. A run in a log has to say something.
        code = main(
            [
                "demo",
                "q-learning",
                "--env",
                "boatrace",
                "--episodes",
                "20",
                "--steps",
                "12",
                "--delay",
                "0",
                "--no-colour",
            ]
        )
        printed = capsys.readouterr().out
        assert code == 0
        assert "step 12" in printed
        assert "laps" in printed

    def test_it_reports_what_the_reward_did_not_pay_for(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "demo",
                "q-learning",
                "--env",
                "vase",
                "--episodes",
                "20",
                "--steps",
                "8",
                "--delay",
                "0",
                "--no-colour",
            ]
        )
        assert "vase_broken" in capsys.readouterr().out


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

    def test_the_pressure_walk_is_off_unless_it_is_asked_for(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # It costs a third again of the time this command already takes, and
        # this command is the one a reader runs first.
        assert main(["gaming", "--no-learn", "--no-colour"]) == 0
        assert "HOW HARD THE AGENT TRIES" not in capsys.readouterr().out

    def test_the_pressure_walk_reports_every_rung(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                [
                    "gaming",
                    "--no-learn",
                    "--no-colour",
                    "--pressure",
                    "--pressure-episodes",
                    "4",
                ]
            )
            == 0
        )
        printed = capsys.readouterr().out
        assert "HOW HARD THE AGENT TRIES" in printed
        assert "the reward" in printed
        assert "the point" in printed
        # Six rungs on each of three environments, and the ends of the ladder
        # are the two rows a reader checks first.
        assert printed.count("       0.0 ") >= 3
        assert printed.count("       1.0 ") >= 3

    def test_the_pressure_chart_draws_with_no_terminal(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A test runs inside a captured stream rather than a terminal. A chart
        # that only drew when one was attached would pass every other test in
        # this file and print nothing where it matters.
        assert (
            main(
                [
                    "gaming",
                    "--no-learn",
                    "--no-colour",
                    "--pressure",
                    "--pressure-episodes",
                    "4",
                ]
            )
            == 0
        )
        printed = capsys.readouterr().out
        drawn = printed[printed.index("HOW HARD THE AGENT TRIES") :]
        assert any("\u2800" <= mark <= "\u28ff" for mark in drawn)


def test_the_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    from rel import __version__

    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert __version__ in capsys.readouterr().out


class TestBothDigests:
    def test_the_report_gives_the_path_and_what_was_learned(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            ["train", "q-learning", "--env", "cliff", "--episodes", "20", "--no-colour"]
        )
        printed = capsys.readouterr().out
        assert "digest, the path" in printed
        assert "digest, what it learned" in printed

    def test_an_agent_that_learns_nothing_shows_only_the_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(["train", "random", "--env", "cliff", "--episodes", "10", "--no-colour"])
        printed = capsys.readouterr().out
        assert "digest, the path" in printed
        assert "what it learned" not in printed

    def test_print_digest_still_gives_the_path_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Every number in the documentation was compared against this one, so
        # what --print digest means cannot change.
        main(
            [
                "train",
                "q-learning",
                "--env",
                "cliff",
                "--episodes",
                "50",
                "--quiet",
                "--print",
                "digest",
            ]
        )
        assert capsys.readouterr().out.strip() == "0de6831e401c7ddd"


class TestTheErrorAgainstTheTrueValues:
    def test_a_predictor_on_the_walk_reports_it(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(
                [
                    "train",
                    "td",
                    "--env",
                    "walk",
                    "--episodes",
                    "200",
                    "--set",
                    "start_value=0.5",
                    "--quiet",
                    "--no-colour",
                ]
            )
            == 0
        )
        assert "error against the true values" in capsys.readouterr().out

    def test_an_agent_that_controls_does_not(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # It keeps a number for an action rather than for a state, so there
        # is nothing to compare.
        main(
            [
                "train",
                "q-learning",
                "--env",
                "walk",
                "--episodes",
                "50",
                "--quiet",
                "--no-colour",
            ]
        )
        assert "error against the true values" not in capsys.readouterr().out

    def test_an_environment_that_cannot_say_does_not(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "train",
                "td",
                "--env",
                "cliff",
                "--episodes",
                "20",
                "--set",
                "policy=optimal",
                "--quiet",
                "--no-colour",
            ]
        )
        assert "error against the true values" not in capsys.readouterr().out
