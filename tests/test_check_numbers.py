"""The parsing in `scripts/check_numbers.py`, which is where it can go wrong.

The script asks whether a number the documentation states still comes out of
some command on the page. Everything hard about that is in deciding which
characters in a cell are a number and which command accounts for a table, and
both are decided here rather than by running anything.

Which command a table came from is worked out by matching its numbers against
every command's output, not by position. `TestAttributingATableToACommand` is
that decision, and the reason for it is in the script.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_numbers.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The script, loaded by its path and put in `sys.modules` first.

    The other measurement scripts are loaded without that line. This one
    needs it because it declares dataclasses, and `dataclass` looks its own
    module up by name to resolve the annotations on it. A module built from
    a path and never registered is not there to be found, and the failure is
    an `AttributeError` inside `dataclasses` that says nothing about paths.
    """
    spec = importlib.util.spec_from_file_location("check_numbers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "page.md"
    path.write_text(text)
    return path


class TestReadingACommand:
    def test_one_command(self, script: ModuleType) -> None:
        assert script.commands_in(["$ python one.py"]) == ["python one.py"]

    def test_two_commands(self, script: ModuleType) -> None:
        assert script.commands_in(["$ a", "$ b"]) == ["a", "b"]

    def test_a_backslash_joins_the_next_line(self, script: ModuleType) -> None:
        """The documentation wraps long commands. Reading the halves as two
        commands would run `measure_agents.py --env cliff` and then report
        that the settings after the backslash produced nothing."""
        assert script.commands_in(
            ["$ python run.py --env cliff \\", "    --runs 30 --set n=4"]
        ) == ["python run.py --env cliff --runs 30 --set n=4"]

    def test_three_lines_join_into_one(self, script: ModuleType) -> None:
        assert script.commands_in(["$ a \\", "b \\", "c"]) == ["a b c"]

    def test_output_lines_are_not_commands(self, script: ModuleType) -> None:
        # A console block that shows what a command printed as well as the
        # command. Only the lines with a prompt are commands.
        assert script.commands_in(["$ rel list", "cliff", "maze"]) == ["rel list"]

    def test_an_unfinished_continuation_is_still_a_command(
        self, script: ModuleType
    ) -> None:
        assert script.commands_in(["$ a \\"]) == ["a"]


class TestFindingTheCommandsAndTheTables:
    PAGE = """# A page

```console
$ python first.py
```

| agent | mean |
| --- | ---: |
| sarsa | -27.70 |

### A heading between the table and the next one

| agent | mean |
| --- | ---: |
| q-learning | -50.71 |

```console
$ python second.py
$ python third.py
```

| what | value |
| --- | ---: |
| something | 4.5 |
"""

    def test_every_command_is_found(self, script: ModuleType, tmp_path: Path) -> None:
        commands, _ = script.read(write(tmp_path, self.PAGE))
        assert commands == ["python first.py", "python second.py", "python third.py"]

    def test_a_command_written_twice_is_run_once(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        # The page shows `rel compare sarsa q-learning --env cliff --runs 10`
        # in two places. Running it twice would double the slowest part of
        # this for nothing.
        page = "```console\n$ python a.py\n```\n\n```console\n$ python a.py\n```\n"
        commands, _ = script.read(write(tmp_path, page))
        assert commands == ["python a.py"]

    def test_every_table_is_found(self, script: ModuleType, tmp_path: Path) -> None:
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert len(claims) == 3
        assert [claim.numbers for claim in claims] == [["-27.70"], ["-50.71"], ["4.5"]]

    def test_a_heading_does_not_join_two_tables_into_one(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert [claim.line for claim in claims] == [7, 13, 22]

    def test_each_table_remembers_the_command_nearest_above_it(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """A hint rather than an answer. It settles a tie when two commands
        account for a table equally well, and the report names it when the
        command that accounts for a table is a different one."""
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert [claim.nearest for claim in claims] == [
            "python first.py",
            "python first.py",
            "python third.py",
        ]

    def test_a_table_before_any_command_is_still_a_table(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        _, claims = script.read(write(tmp_path, "| a | 1 |\n\n```console\n$ x\n```\n"))
        assert [claim.nearest for claim in claims] == [""]

    def test_a_block_that_is_not_console_is_not_a_command(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = "```python\nx = 1\n```\n\n| a | 1 |\n"
        commands, claims = script.read(write(tmp_path, page))
        assert commands == []
        assert len(claims) == 1

    def test_the_table_of_commands_at_the_end_is_not_a_result(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """`algorithms.md` closes with a table of what produces every other
        table on the page. Every number in it belongs to a command line, so
        checking them would ask whether `--runs 10` is still printed."""
        page = (
            "```console\n$ python x.py\n```\n\n"
            "| Table | Command |\n| --- | --- |\n"
            "| Every agent | `python scripts/measure_agents.py --runs 10` |\n"
        )
        assert script.read(write(tmp_path, page))[1] == []


class TestWhichCharactersAreANumber:
    def claim(self, script: ModuleType, *rows: str) -> list[str]:
        made = script.Claim(1)
        made.rows = list(rows)
        return made.numbers

    def test_a_decimal_and_its_sign(self, script: ModuleType) -> None:
        assert self.claim(script, "| sarsa | -27.70 | 4 |") == ["-27.70", "4"]

    def test_the_rule_under_the_heading_carries_no_claim(
        self, script: ModuleType
    ) -> None:
        assert self.claim(script, "| --- | ---: |") == []

    def test_bold_is_not_part_of_the_number(self, script: ModuleType) -> None:
        # A best value is written **-13.00** on this page, and the stars would
        # otherwise sit between the minus sign and the digits.
        assert self.claim(script, "| q-learning | **-13.00** |") == ["-13.00"]

    def test_a_thousands_separator_is_not_two_numbers(self, script: ModuleType) -> None:
        assert self.claim(script, "| features | 52,488 |") == ["52488"]

    def test_a_count_written_in_words_around_it_is_still_found(
        self, script: ModuleType
    ) -> None:
        assert self.claim(script, "| random | 10 of 10 |") == ["10", "10"]


class TestWhatIsMissing:
    def test_a_number_the_output_holds_is_not_missing(self, script: ModuleType) -> None:
        assert script.missing(["-27.70"], ["-27.70", "1.64"]) == []

    def test_a_number_that_moved_is_reported(self, script: ModuleType) -> None:
        assert script.missing(["-27.70"], ["-28.10"]) == ["-27.70"]

    def test_the_same_number_twice_needs_it_twice(self, script: ModuleType) -> None:
        """Counted with multiplicity on purpose. A table that states -13.00
        four times against an output that prints it twice has changed, and
        matching on membership alone would call it unchanged."""
        assert script.missing(["-13.00", "-13.00"], ["-13.00"]) == ["-13.00"]
        assert script.missing(["-13.00", "-13.00"], ["-13.00", "-13.00"]) == []


class TestAttributingATableToACommand:
    """Which command a table came from, decided by its numbers.

    The first version decided by position and was wrong about two thirds of
    the page, because `## The same table, four more environments` writes one
    command and then four tables, and the other three commands are nowhere on
    the page at all.
    """

    def claim(self, script: ModuleType, row: str, nearest: str = "") -> object:
        made = script.Claim(1, nearest)
        made.rows = [row]
        return made

    def test_the_command_that_accounts_for_it_wins(self, script: ModuleType) -> None:
        claim = self.claim(script, "| a | -20.20 | -22.73 |", nearest="wrong one")
        command, absent = script.attribute(
            claim,
            {"wrong one": ["-15.00", "-16.17"], "right one": ["-20.20", "-22.73"]},
        )
        assert command == "right one"
        assert absent == []

    def test_a_tie_goes_to_the_command_above_it(self, script: ModuleType) -> None:
        """A page that does say where a table came from is taken at its word
        whenever the numbers do not disagree with it. Two commands printing
        the same numbers is the ordinary case for a sweep."""
        claim = self.claim(script, "| a | 1.5 |", nearest="the one above")
        both = {"the one above": ["1.5"], "another": ["1.5"]}
        assert script.attribute(claim, both)[0] == "the one above"

    def test_a_table_no_command_accounts_for_comes_back_empty(
        self, script: ModuleType
    ) -> None:
        # Which is the state the page is in for a table whose command it never
        # names. Reporting it as stale would be wrong and reporting it as
        # clean would be worse.
        claim = self.claim(script, "| a | 9.99 |", nearest="something")
        command, absent = script.attribute(claim, {"something": ["1.0"]})
        assert command == ""
        assert absent == ["9.99"]

    def test_the_best_of_several_partial_matches_is_taken(
        self, script: ModuleType
    ) -> None:
        claim = self.claim(script, "| a | 1 | 2 | 3 | 4 |")
        command, absent = script.attribute(
            claim,
            {"two": ["1", "2"], "three": ["1", "2", "3"], "none": []},
        )
        assert command == "three"
        assert absent == ["4"]

    def test_a_command_that_accounts_for_too_little_is_a_coincidence(
        self, script: ModuleType
    ) -> None:
        """The second thing the first version got wrong.

        It named whichever of forty eight commands matched most, and one
        number of seventy is enough to win that when every other command
        matches none. Small integers turn up in every output, so a table with
        no command on the page is otherwise attributed to a coincidence.
        """
        claim = self.claim(script, "| a | 1 | 2 | 3 | 4 |", nearest="above it")
        assert script.attribute(claim, {"above it": ["1"]}) == (
            "",
            ["1", "2", "3", "4"],
        )

    def test_exactly_half_is_enough(self, script: ModuleType) -> None:
        # The boundary, written down because `ENOUGH` is a judgement and the
        # comparison around it is the kind that is easy to get wrong by one.
        claim = self.claim(script, "| a | 1 | 2 | 3 | 4 |")
        assert script.attribute(claim, {"half": ["1", "2"]})[0] == "half"

    def test_a_command_that_accounts_for_nothing_is_not_named(
        self, script: ModuleType
    ) -> None:
        # Even when it is the command written above the table, and even when
        # no other command does better. Naming it would read as an
        # attribution, and the truth is that nothing on the page explains it.
        claim = self.claim(script, "| a | 9.99 |", nearest="above it")
        assert script.attribute(claim, {"above it": ["1.0"]}) == ("", ["9.99"])


class TestRunningTheCommand:
    def test_python_becomes_this_interpreter(self, script: ModuleType) -> None:
        # `python` on the path may be another version or absent. The
        # documentation means the one the project runs under.
        printed, _, trouble = script.run("python -c print(7)", 60.0)
        assert trouble == ""
        assert "7" in printed

    def test_the_installed_entry_point_becomes_the_module(
        self, script: ModuleType
    ) -> None:
        """The page writes `rel list`, which needs the package installed.
        Nothing else in this project does, so the script runs the module."""
        printed, _, trouble = script.run("rel list", 120.0)
        assert trouble == ""
        assert "q-learning" in printed

    def test_a_command_that_fails_says_so(self, script: ModuleType) -> None:
        _, _, trouble = script.run("rel train no-such-agent --env cliff", 120.0)
        assert trouble.startswith("exit ")

    def test_a_command_that_does_not_exist_says_so(self, script: ModuleType) -> None:
        _, _, trouble = script.run("definitely-not-a-command", 60.0)
        assert "could not run it" in trouble

    def test_a_command_that_never_finishes_is_cut_off(self, script: ModuleType) -> None:
        _, taken, trouble = script.run('python -c "import time; time.sleep(30)"', 0.5)
        assert trouble.startswith(script.TIMED_OUT)
        assert taken < 5.0

    def test_giving_up_reads_differently_from_failing(self, script: ModuleType) -> None:
        """The first version reported both as a command that would not run.

        Three of the page's commands take longer than fifteen minutes, and it
        called all three broken and returned one. A budget this script chose
        is not a defect in the documentation, and putting the two together
        made the real failures unfindable.
        """
        _, _, slow = script.run('python -c "import time; time.sleep(30)"', 0.5)
        _, _, failed = script.run("rel train no-such-agent --env cliff", 120.0)
        assert slow.startswith(script.TIMED_OUT)
        assert not failed.startswith(script.TIMED_OUT)


class TestAgainstTheRealPage:
    def test_every_command_is_a_command(self, script: ModuleType) -> None:
        commands, _ = script.read(script.ROOT / "docs" / "algorithms.md")
        assert commands
        assert all(command.startswith(("python ", "rel ")) for command in commands)

    def test_the_page_states_a_great_many_numbers(self, script: ModuleType) -> None:
        # The point of the exercise. If this drops to nothing, the parsing
        # broke and the script would report a clean page either way.
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        assert sum(len(claim.numbers) for claim in claims) > 800

    def test_no_claim_holds_a_command_line(self, script: ModuleType) -> None:
        # The closing table of the page is a list of commands. Every number in
        # it is a setting rather than a result, so none of its rows may reach
        # a claim.
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        for claim in claims:
            for row in claim.rows:
                assert "scripts/measure" not in row, row

    def test_more_tables_than_commands_that_could_own_them(
        self, script: ModuleType
    ) -> None:
        """Not an accident of counting. Several sections write one command and
        then a table for each of four environments, so the page has tables it
        never names a command for, and matching is the only thing that can
        say which those are."""
        commands, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        nearest = {claim.nearest for claim in claims}
        assert len(nearest) < len(commands)


class TestWhatTheExitCodeMeans:
    """One exit code, for the one thing here that is unambiguous.

    A number that moved needs a person to look at it: this cannot tell a
    table that is stale from a table attached to the wrong command. A
    documented command that will not run at all is a defect either way, so
    that is what the exit code is for.
    """

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        *rest: str,
    ) -> int:
        monkeypatch.setattr(
            sys, "argv", ["check_numbers", "--doc", str(page), "--all", *rest]
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        return int(script.main())

    def test_a_command_that_runs_and_matches_is_zero(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = write(
            tmp_path,
            '```console\n$ python -c "print(7.5)"\n```\n\n| a | b |\n| --- | --- |\n| x | 7.5 |\n',
        )
        assert self.go(script, monkeypatch, page) == 0

    def test_a_number_that_moved_is_still_zero(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = write(
            tmp_path,
            '```console\n$ python -c "print(7.5)"\n```\n\n| a | b |\n| --- | --- |\n| x | 9.5 |\n',
        )
        assert self.go(script, monkeypatch, page) == 0

    def test_a_command_that_will_not_run_is_one(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = write(
            tmp_path,
            "```console\n$ definitely-not-a-command\n```\n\n| a | b |\n| --- | --- |\n| x | 1 |\n",
        )
        assert self.go(script, monkeypatch, page) == 1

    def test_it_says_which_numbers_moved_and_where(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = write(
            tmp_path,
            '```console\n$ python -c "print(7.5)"\n```\n\n| a | b |\n| --- | --- |\n| x | 9.5 |\n',
        )
        self.go(script, monkeypatch, page)
        printed = capsys.readouterr().out
        assert "line 5" in printed
        assert "no command on this page prints any of its 1 numbers" in printed

    def test_the_listing_runs_nothing(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A page whose only command cannot be run. `--list` says what it
        # would do, so it has to say it without finding that out.
        page = write(
            tmp_path,
            "```console\n$ definitely-not-a-command\n```\n\n| a | b |\n| --- | --- |\n| x | 1 |\n",
        )
        monkeypatch.setattr(
            sys, "argv", ["check_numbers", "--doc", str(page), "--list"]
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        assert script.main() == 0
        assert "definitely-not-a-command" in capsys.readouterr().out


class TestATableThatAsksNotToBeChecked:
    """An HTML comment above a table, invisible in the rendered page.

    There has to be a way to say this or the tool dies of its own noise. Three
    tables on this page report timings, which differ on every machine and
    every run, and a report that lists the same twenty unfixable numbers every
    time teaches a reader to stop reading it.
    """

    PAGE = (
        "```console\n$ python x.py\n```\n\n"
        "<!-- not checked: these are seconds and belong to the machine -->\n\n"
        "| what | time |\n| --- | ---: |\n| a run | 1.42 |\n"
    )

    def test_the_reason_is_kept(self, script: ModuleType, tmp_path: Path) -> None:
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert claims[0].exempt == "these are seconds and belong to the machine"

    def test_the_table_is_still_read(self, script: ModuleType, tmp_path: Path) -> None:
        # Read and then set aside, rather than skipped in the parser. A table
        # nobody parses is a table nobody can report the existence of.
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert claims[0].numbers == ["1.42"]

    def test_a_table_without_the_comment_is_not_exempt(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = "```console\n$ python x.py\n```\n\n| what | time |\n| --- | ---: |\n| a | 1.42 |\n"
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].exempt == ""

    def test_the_exemption_does_not_carry_to_the_next_table(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """The one way this could quietly turn the whole page off.

        Every table after an exempt one would stop being checked, and the
        report would say so only by counting, which nobody reads.
        """
        page = (
            self.PAGE
            + "\nSome prose.\n\n| what | value |\n| --- | ---: |\n| b | 2.5 |\n"
        )
        _, claims = script.read(write(tmp_path, page))
        assert [claim.exempt for claim in claims] == [
            "these are seconds and belong to the machine",
            "",
        ]

    def test_a_reason_that_wraps_onto_a_second_line_still_counts(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """The mistake this caught, in the page rather than in the script.

        Four tables were marked and two took. The two that did not had a
        reason long enough to wrap, and the marker read only its first line,
        so the table quietly went back to being checked.
        """
        page = (
            "```console\n$ python x.py\n```\n\n"
            "<!-- not checked: the microseconds belong to the machine, and the\n"
            "ratio between the two is what the section is about -->\n\n"
            "| a | b |\n| --- | ---: |\n| x | 1.0 |\n"
        )
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].exempt == (
            "the microseconds belong to the machine, and the ratio between the "
            "two is what the section is about"
        )

    def test_a_comment_with_no_reason_says_so(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        # The reason is what stops a table exempting itself to hide a real
        # defect, so a missing one is recorded rather than accepted quietly.
        page = (
            "```console\n$ python x.py\n```\n\n<!-- not checked: -->\n\n"
            "| a | b |\n| --- | ---: |\n| x | 1.0 |\n"
        )
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].exempt == "no reason given"

    def test_the_report_lists_them_apart_from_the_findings(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = write(
            tmp_path,
            '```console\n$ python -c "print(9.9)"\n```\n\n'
            "<!-- not checked: these are seconds -->\n\n"
            "| what | time |\n| --- | ---: |\n| a run | 1.42 |\n",
        )
        monkeypatch.setattr(sys, "argv", ["check_numbers", "--doc", str(page), "--all"])
        monkeypatch.setattr(script, "ROOT", page.parent)
        assert script.main() == 0

        printed = capsys.readouterr().out
        assert "1 tables ask not to be checked" in printed
        assert "these are seconds" in printed
        assert "0 of 0 tables are wholly accounted for" in printed
