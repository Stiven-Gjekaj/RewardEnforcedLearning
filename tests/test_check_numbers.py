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
import json
import re
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

    def test_each_table_remembers_the_whole_block_above_it(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """All of the block's commands rather than its last one.

        A block that shows four commands and then four tables has all four
        above every one of them. Taking the last made three of the four read
        as attributed to a command that was not theirs, which is a false
        alarm in the one place the report has to be trusted.
        """
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert [claim.near for claim in claims] == [
            ["python first.py"],
            ["python first.py"],
            ["python second.py", "python third.py"],
        ]

    def test_a_table_before_any_command_is_still_a_table(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        _, claims = script.read(write(tmp_path, "| a | 1 |\n\n```console\n$ x\n```\n"))
        assert [claim.near for claim in claims] == [[]]

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
        made = script.Claim(1, [nearest] if nearest else [])
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
        blocks = {tuple(claim.near) for claim in claims}
        assert len(blocks) < len(commands)


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


class TestATableThatExemptsOneColumn:
    """One column out, the rest still checked.

    Two tables here report model steps, returns and a time side by side. The
    time belongs to the machine and the other five columns are the numbers the
    section is about, so exempting the table whole to silence one cell would
    drop the coverage that matters.
    """

    PAGE = (
        "```console\n$ python x.py\n```\n\n"
        "<!-- not checked, column time: seconds belong to the machine -->\n\n"
        "| agent | steps | time |\n| --- | ---: | ---: |\n| mcts | 15000 | 324s |\n"
    )

    def test_the_named_column_is_left_out(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert claims[0].numbers == ["15000"]

    def test_the_table_is_still_checked(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        # `exempt` stays empty, which is what keeps it out of the list of
        # tables that asked to be left alone.
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert claims[0].exempt == ""
        assert claims[0].skipped == "time"

    def test_the_column_is_found_by_its_heading(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert claims[0].dropped == 2

    def test_a_column_that_is_not_there_drops_nothing(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """A marker naming a column the table does not have does nothing.

        Read off the heading row rather than trusted, so a heading that gets
        renamed leaves the report short of numbers and saying so, rather than
        quietly passing a table nobody is checking any more.
        """
        page = self.PAGE.replace("column time", "column seconds")
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].dropped == -1
        assert claims[0].numbers == ["15000", "324"]

    def test_naming_no_column_still_exempts_the_whole_table(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = self.PAGE.replace("not checked, column time:", "not checked:")
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].exempt == "seconds belong to the machine"
        assert claims[0].skipped == ""

    def test_the_exemption_does_not_carry_to_the_next_table(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = self.PAGE + "\nProse.\n\n| a | time |\n| --- | ---: |\n| x | 9 |\n"
        _, claims = script.read(write(tmp_path, page))
        assert [claim.skipped for claim in claims] == ["time", ""]
        assert claims[1].numbers == ["9"]


class TestTheListingSaysWhatIsCovered:
    """`--list` runs nothing, so it is the only way to see coverage cheaply.

    A whole run takes about two and a half hours. A reader deciding whether to
    start one wants to know how much of the page it would check before they
    wait for it.
    """

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
    ) -> str:
        monkeypatch.setattr(
            sys, "argv", ["check_numbers", "--doc", str(page), "--list"]
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        return str(script.main())

    def test_it_counts_the_checked_and_the_unchecked_apart(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = write(
            tmp_path,
            "```console\n$ python x.py\n```\n\n"
            "| a | b |\n| --- | ---: |\n| x | 1 | 2 |\n\n"
            "<!-- not checked: seconds -->\n\n"
            "| c | time |\n| --- | ---: |\n| y | 9 |\n",
        )
        self.go(script, monkeypatch, page)
        assert "2 numbers are checked and 1 are not" in capsys.readouterr().out

    def test_a_column_left_out_is_named(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = write(
            tmp_path,
            "```console\n$ python x.py\n```\n\n"
            "<!-- not checked, column time: seconds -->\n\n"
            "| a | steps | time |\n| --- | ---: | ---: |\n| x | 15 | 9 |\n",
        )
        self.go(script, monkeypatch, page)
        printed = capsys.readouterr().out
        assert "column time" in printed
        assert "1 numbers are checked" in printed

    def test_most_of_the_page_is_checked(self, script: ModuleType) -> None:
        """A smoke alarm rather than a guard, and it has been moved twice.

        A ratio cannot tell a good reason from a bad one, so it was never the
        thing protecting this. It started at twenty to one and is at more than
        half, because the honest exemptions turned out to include one sweep of
        twenty five runs that is 65 numbers on its own.

        Moving it again is the smell. The test below is what actually holds.
        """
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        checked = sum(len(c.numbers) for c in claims if not c.exempt)
        left = sum(len(c.numbers) for c in claims if c.exempt)
        assert checked > left

    def test_most_tables_are_checked(self, script: ModuleType) -> None:
        # Counted in tables rather than numbers, because one large sweep can
        # carry more numbers than a dozen small tables and would otherwise
        # decide the ratio above on its own.
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        exempt = [claim for claim in claims if claim.exempt]
        assert len(exempt) * 2 < len(claims)

    def test_every_exemption_gives_a_reason(self, script: ModuleType) -> None:
        """The real guard, and the reason the marker demands one.

        A table that exempts itself without saying why is how a number that
        moved gets hidden, and it would look exactly like a table that cannot
        be checked. The reason is what a reader disagrees with.
        """
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        for claim in claims:
            assert claim.exempt != "no reason given", claim.line
            if claim.skipped:
                assert claim.dropped >= 0, claim.line


class TestTheCacheOfWhatEachCommandPrinted:
    """A whole run is about three hours, and the answer changes on every edit.

    Checking a fix should not cost three hours again, so `--cache` keeps what
    each command printed and runs only what is missing from it.
    """

    PAGE = '```console\n$ python -c "print(7.5)"\n```\n\n| a | b |\n| --- | --- |\n| x | 7.5 |\n'

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        cache: Path,
    ) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            ["check_numbers", "--doc", str(page), "--all", "--cache", str(cache)],
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        return int(script.main())

    def test_a_run_writes_what_it_saw(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cache = tmp_path / "kept.json"
        self.go(script, monkeypatch, write(tmp_path, self.PAGE), cache)
        capsys.readouterr()
        held = json.loads(cache.read_text())["printed"]
        assert list(held) == ['python -c "print(7.5)"']
        assert held['python -c "print(7.5)"']["numbers"] == ["7.5"]
        assert held['python -c "print(7.5)"']["code"] == script.fingerprint(
            'python -c "print(7.5)"'
        )

    def test_a_second_run_does_not_run_the_command_again(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        self.go(script, monkeypatch, page, cache)
        capsys.readouterr()

        self.go(script, monkeypatch, page, cache)
        printed = capsys.readouterr().out
        assert "1 of 1 commands come from" in printed
        assert "[1/1]" not in printed

    def test_an_edited_page_is_rechecked_against_the_same_outputs(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The reason it exists. Fix a table, ask again, pay nothing."""
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE.replace("7.5 |", "9.9 |"))
        self.go(script, monkeypatch, page, cache)
        assert "no command on this page" in capsys.readouterr().out

        page.write_text(self.PAGE)
        self.go(script, monkeypatch, page, cache)
        assert "1 of 1 tables are wholly accounted for" in capsys.readouterr().out

    def test_a_command_the_page_no_longer_names_is_not_used(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A cache outliving the command it was made for would otherwise
        account for a table with the output of something the page has
        stopped saying to run, which is the opposite of the point."""
        cache = tmp_path / "kept.json"
        old = 'python -c "print(1)"'
        cache.write_text(
            json.dumps(
                {
                    "printed": {
                        old: {"code": script.fingerprint(old), "numbers": ["7.5"]}
                    }
                }
            )
        )
        self.go(script, monkeypatch, write(tmp_path, self.PAGE), cache)
        capsys.readouterr()
        held = json.loads(cache.read_text())
        assert list(held["printed"]) == ['python -c "print(7.5)"']


class TestItDoesNotRunItself:
    """The page documents this script the way it documents every other one.

    A console block is a list of things to run, and one of the things it now
    lists is this script with `--all`. Running that from inside a run would
    spend three hours to say nothing, recursively.
    """

    def test_a_command_naming_this_script_is_not_collected(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = (
            "```console\n"
            "$ python scripts/check_numbers.py --all\n"
            "$ python scripts/measure_agents.py --runs 10\n"
            "```\n\n| a | b |\n| --- | ---: |\n| x | 1 |\n"
        )
        commands, _ = script.read(write(tmp_path, page))
        assert commands == ["python scripts/measure_agents.py --runs 10"]

    def test_the_real_page_documents_it_and_does_not_run_it(
        self, script: ModuleType
    ) -> None:
        commands, _ = script.read(script.ROOT / "docs" / "algorithms.md")
        assert "check_numbers" in (script.ROOT / "docs" / "algorithms.md").read_text()
        assert not any(script.is_this_script(command) for command in commands)


class TestTheCacheKnowsWhenTheCodeChanged:
    """The fault this script exists to catch, committed by this script.

    The cache holds what a command printed, and what a command prints depends
    on the code under it. A cache made before a change would otherwise confirm
    numbers the code has stopped producing, which is exactly a documented
    number that moved and nothing saying so.
    """

    PAGE = '```console\n$ python -c "print(7.5)"\n```\n\n| a | b |\n| --- | --- |\n| x | 7.5 |\n'

    def test_a_cache_from_other_code_is_not_used(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cache = tmp_path / "kept.json"
        cache.write_text(
            json.dumps(
                {
                    "printed": {
                        'python -c "print(7.5)"': {
                            "code": "from some other tree",
                            "numbers": ["7.5"],
                        }
                    }
                }
            )
        )
        page = write(tmp_path, self.PAGE)
        monkeypatch.setattr(
            sys,
            "argv",
            ["check_numbers", "--doc", str(page), "--all", "--cache", str(cache)],
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()

        printed = capsys.readouterr().out
        assert "made from code that has since moved" in printed
        assert "[1/1]" in printed

    def test_a_script_moving_moves_only_its_own_commands(
        self, script: ModuleType
    ) -> None:
        """Why the fingerprint is per command rather than per repository.

        Hashing every script together meant that adding an option to one of
        them threw away the cached output of the other fifty, which have
        never heard of it. That happened twice in one sitting and cost hours
        each time.
        """
        mine = "python scripts/measure_sweeping.py --episodes 400"
        yours = "python scripts/measure_importance.py --episodes 1200"
        before = (script.fingerprint(mine), script.fingerprint(yours))

        path = script.ROOT / "scripts" / "measure_sweeping.py"
        was = path.read_text()
        try:
            path.write_text(was.replace("PLANNERS = (", "PLANNERS = tuple(", 1))
            assert script.fingerprint(mine) != before[0]
            assert script.fingerprint(yours) == before[1]
        finally:
            path.write_text(was)
        assert script.fingerprint(mine) == before[0]

    def test_the_package_moving_moves_every_command(self, script: ModuleType) -> None:
        # Everything runs the package, so a change there is a change to all
        # of them and the whole cache goes.
        commands = ["python scripts/measure_sweeping.py", "rel train sarsa"]
        before = [script.fingerprint(command) for command in commands]

        path = script.ROOT / "rel" / "agents" / "basis.py"
        was = path.read_text()
        try:
            path.write_text(was.replace("return optimism", "return optimism * 2", 1))
            after = [script.fingerprint(command) for command in commands]
            assert all(one != two for one, two in zip(before, after, strict=True))
        finally:
            path.write_text(was)

    def test_a_command_that_names_no_script_is_the_package_alone(
        self, script: ModuleType
    ) -> None:
        # `rel` commands run the package rather than a file in `scripts/`.
        assert script.script_of("rel train sarsa --env cliff") is None
        assert script.fingerprint("rel train sarsa") == script.fingerprint()

    def test_a_docstring_does_not_move_it(self, script: ModuleType) -> None:
        """What a command prints depends on what the code does, not on what it
        says about itself.

        Hashing the bytes of a file made every corrected comment throw away
        hours of cache, which is a cost with nothing bought.
        """
        before = script.fingerprint()
        path = script.ROOT / "rel" / "agents" / "basis.py"
        was = path.read_text()
        try:
            path.write_text(was.replace('"""Turns a point', '"""Turns a POINT', 1))
            assert script.fingerprint() == before
        finally:
            path.write_text(was)

    def test_a_comment_does_not_move_it(self, script: ModuleType) -> None:
        before = script.fingerprint()
        path = script.ROOT / "rel" / "agents" / "basis.py"
        was = path.read_text()
        try:
            path.write_text(was + "\n# a comment that changes nothing\n")
            assert script.fingerprint() == before
        finally:
            path.write_text(was)

    def test_code_does_move_it(self, script: ModuleType) -> None:
        before = script.fingerprint()
        path = script.ROOT / "rel" / "agents" / "basis.py"
        was = path.read_text()
        try:
            path.write_text(was.replace("return optimism", "return optimism * 2", 1))
            assert script.fingerprint() != before
        finally:
            path.write_text(was)
        assert script.fingerprint() == before

    def test_the_interpreter_is_part_of_it(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same code on two Pythons really does print different numbers.

        CPython 3.12 gave `sum()` compensated summation over floats, and
        twenty episodes of the cart pole is enough for that to reach the
        figures a digest hashes. A cache made on one is not a cache for the
        other.
        """
        before = script.fingerprint()
        monkeypatch.setattr(sys, "version_info", (9, 9, 9, "final", 0))
        assert script.fingerprint() != before

    def test_taking_the_prose_out_leaves_the_code(self, script: ModuleType) -> None:
        one = script.without_prose('def f():\n    """Says a thing."""\n    return 1\n')
        two = script.without_prose("def f():\n    return 1\n")
        assert one == two

        three = script.without_prose("def f():\n    return 2\n")
        assert one != three

    def test_a_string_that_is_not_a_docstring_stays(self, script: ModuleType) -> None:
        # Only the first statement of a module, class or function is a
        # docstring. A string used as a value is code.
        one = script.without_prose('def f():\n    return "a value"\n')
        two = script.without_prose('def f():\n    return "another"\n')
        assert one != two

    def test_this_script_is_not_part_of_its_own_fingerprint(
        self, script: ModuleType
    ) -> None:
        """Nothing a documented command prints depends on the checker, and
        including it would throw away hours of cache every time the report's
        wording changed."""
        before = script.fingerprint()
        here = Path(script.__file__)
        was = here.read_bytes()
        try:
            here.write_bytes(was + b"\n# a comment\n")
            assert script.fingerprint() == before
        finally:
            here.write_bytes(was)


class TestANarrowedRunKeepsTheWholeCache:
    """The bug this class is named after would delete hours of work.

    `--only` narrows which commands run. The cache was written back from the
    narrowed list, so asking a smaller question threw away every command the
    question did not ask about. Hours of runs, gone, by looking at one table.
    """

    PAGE = (
        "```console\n"
        '$ python -c "print(1.5)"\n'
        '$ python -c "print(2.5)"\n'
        "```\n\n| a | b |\n| --- | --- |\n| x | 1.5 | 2.5 |\n"
    )

    def run_with(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        cache: Path,
        *rest: str,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_numbers",
                "--doc",
                str(page),
                "--cache",
                str(cache),
                "--all",
                *rest,
            ],
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()

    def held(self, cache: Path) -> dict[str, object]:
        loaded: dict[str, object] = json.loads(cache.read_text())["printed"]
        return loaded

    def test_a_whole_run_caches_both(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cache = tmp_path / "kept.json"
        self.run_with(script, monkeypatch, write(tmp_path, self.PAGE), cache)
        capsys.readouterr()
        assert len(self.held(cache)) == 2

    def test_a_narrowed_run_leaves_the_other_alone(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        self.run_with(script, monkeypatch, page, cache)
        capsys.readouterr()

        # Narrow to one of the two, and make it a command the cache does not
        # hold, so the run has something to write and therefore something to
        # write over.
        page.write_text(self.PAGE.replace("print(1.5)", "print(3.5)"))
        self.run_with(script, monkeypatch, page, cache, "--only", "3.5")
        capsys.readouterr()

        assert set(self.held(cache)) == {
            'python -c "print(2.5)"',
            'python -c "print(3.5)"',
        }

    def test_a_command_the_page_dropped_is_still_pruned(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Pruning against the whole page rather than the narrowed run.

        Keeping everything would be the other mistake: a cache that grows for
        ever and accounts for a table with the output of a command the page
        stopped naming.
        """
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        self.run_with(script, monkeypatch, page, cache)
        capsys.readouterr()

        page.write_text(self.PAGE.replace('$ python -c "print(1.5)"\n', ""))
        self.run_with(script, monkeypatch, page, cache, "--only", "2.5")
        capsys.readouterr()
        assert set(self.held(cache)) == {'python -c "print(2.5)"'}


class TestABlockOfSeveralCommandsIsAllAboveItsTables:
    """The false alarm this fixes, at the smallest size that shows it.

    `The same table, four more environments` writes four commands and then
    four tables. Every table matches one of the four, and taking only the
    last command as "the one above it" made three of the four report as
    attributed to a command that was not theirs. A report has to be trusted
    to be read, and that is three false alarms out of four.
    """

    PAGE = (
        "```console\n"
        '$ python -c "print(1.5)"\n'
        '$ python -c "print(2.5)"\n'
        "```\n\n"
        "| a | b |\n| --- | --- |\n| x | 1.5 |\n\n"
        "| a | b |\n| --- | --- |\n| y | 2.5 |\n"
    )

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> str:
        monkeypatch.setattr(sys, "argv", ["check_numbers", "--doc", str(page), "--all"])
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()
        return capsys.readouterr().out

    def test_neither_table_is_called_misplaced(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        printed = self.go(script, monkeypatch, write(tmp_path, self.PAGE), capsys)
        assert "2 of 2 tables are wholly accounted for" in printed
        assert "in no block above it" not in printed

    def test_a_table_matching_a_command_from_another_block_is_named(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The case the message is actually for: the numbers come from a
        # command written somewhere else on the page.
        page = self.PAGE + '\n```console\n$ python -c "print(9.5)"\n```\n\n'
        page += "| a | b |\n| --- | --- |\n| z | 1.5 |\n"
        printed = self.go(script, monkeypatch, write(tmp_path, page), capsys)
        assert "in no block above it" in printed


class TestABlockCountsAsOneWhenNoSingleCommandExplainsATable:
    """A section that runs one script over three grids and shows one table.

    The exploration section does exactly that: a row for the cliff walk, one
    for four rooms and one for the Dyna maze, from three runs of the same
    script. No one of the three accounts for the table, and reporting it as
    two thirds missing would report the shape of the page rather than a
    number that moved.
    """

    PAGE = (
        "```console\n"
        '$ python -c "print(1.5)"\n'
        '$ python -c "print(2.5)"\n'
        "```\n\n"
        "| grid | value |\n| --- | --- |\n| a | 1.5 |\n| b | 2.5 |\n"
    )

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> str:
        monkeypatch.setattr(sys, "argv", ["check_numbers", "--doc", str(page), "--all"])
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()
        return capsys.readouterr().out

    def test_the_two_together_account_for_it(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        printed = self.go(script, monkeypatch, write(tmp_path, self.PAGE), capsys)
        assert "1 of 1 tables are wholly accounted for" in printed
        assert "in no block above it" not in printed

    def test_neither_of_them_does_on_its_own(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        # The half of it that makes the union necessary rather than tidy.
        _, claims = script.read(write(tmp_path, self.PAGE))
        for one in ('python -c "print(1.5)"', 'python -c "print(2.5)"'):
            _command, absent = script.attribute(claims[0], {one: ["1.5"]})
            assert absent, one

    def test_a_number_in_neither_is_still_missing(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The union must not become a way of explaining anything.

        It is the block's commands and only those, so a number that none of
        them printed is still reported.
        """
        page = write(
            tmp_path, self.PAGE.replace("| b | 2.5 |", "| b | 2.5 |\n| c | 9.9 |")
        )
        printed = self.go(script, monkeypatch, page, capsys)
        assert "missing 9.9" in printed

    def test_one_command_is_never_joined_to_itself(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        # A block of one has nothing to unite, and a name like "x and x"
        # appearing in a report would be nonsense.
        page = '```console\n$ python -c "print(1.5)"\n```\n\n| a | b |\n| --- | --- |\n| x | 1.5 |\n'
        _, claims = script.read(write(tmp_path, page))
        command, absent = script.attribute(
            claims[0], {'python -c "print(1.5)"': ["1.5"]}
        )
        assert command == 'python -c "print(1.5)"'
        assert absent == []


class TestThePageSaysHowMuchOfItselfIsChecked:
    """The one number about this exercise that lives in a sentence.

    `check_numbers.py` reads tables. A count written in prose is exactly the
    kind of number it cannot see, and it went stale within an hour of being
    written: nine more exemptions moved it from 981 to 841 and nothing said
    so. So the sentence is held against what the parsing actually finds.
    """

    def stated(self, script: ModuleType) -> tuple[int, int]:
        page = (script.ROOT / "docs" / "algorithms.md").read_text()
        found = re.search(r"\*\*(\d+) of (\d+) numbers are checked", page)
        assert found is not None, "the page no longer says how much it checks"
        return int(found[1]), int(found[2])

    def test_the_sentence_matches_the_parsing(self, script: ModuleType) -> None:
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        checked = sum(len(c.numbers) for c in claims if not c.exempt)
        every = sum(len(c.numbers) for c in claims)
        assert self.stated(script) == (checked, every)
