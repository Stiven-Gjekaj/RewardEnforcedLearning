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

import ast
import importlib.util
import json
import re
import sys
import time
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


@pytest.fixture(autouse=True)
def anything_may_run(script: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pages written inline below use `python -c` to print a number.

    The script runs only this project's own commands, so that pointing it at
    a page with an install block cannot install anything, and `python -c` is
    neither a script under `scripts/` nor the package's command line. The
    tuple is widened here rather than in the script, and
    `TestItRunsOnlyThisProjectsCommands` reads the real rule out of the source
    file so that this cannot hide a change to it.
    """
    monkeypatch.setattr(script, "OURS", (*script.OURS, "python -c"))


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "page.md"
    path.write_text(text)
    return path


def prose_numbers(script: ModuleType, page: Path) -> int:
    """How many numbers are on the page outside a table, a fence or a marker.

    The same reading the page states: a table cell is a result by
    construction, and everything here is what the tool never looks at.
    """
    found = 0
    fenced = False
    for line in page.read_text().splitlines():
        text = line.strip()
        if text.startswith("```"):
            fenced = not fenced
            continue
        if fenced or text.startswith(("|", "<!--")):
            continue
        found += len(script.NUMBER.findall(text.replace(",", "")))
    return found


def console_commands(page: Path) -> list[str]:
    """Every command in a console block, including this script's own.

    `read` drops the ones that run this script, so that a run does not run
    itself. The closing table is allowed to name it, so the tests that hold
    that table against the page need the unfiltered list.
    """
    import check_numbers

    lines = page.read_text().splitlines()
    found: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith("```console"):
            fence: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                fence.append(lines[index])
                index += 1
            found.extend(check_numbers.commands_in(fence))
        index += 1
    return found


def index_of(page: Path, heading: str) -> list[str]:
    """The commands one closing table names, in the order it names them."""
    rows = []
    inside = False
    for line in page.read_text().splitlines():
        text = line.strip()
        if text == heading:
            inside = True
            continue
        if inside and not text.startswith("|"):
            break
        if inside and set(text) - set("|-: "):
            rows.append(text)
    return [row.split("`")[-2] for row in rows if "`" in row]


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


class TestATrailingComment:
    """A shell drops what follows a hash and this did not.

    `docs/specification-gaming.md` writes `rel gaming  # all three, with the
    repairs`, and five of its commands were run with the words after the hash
    handed to them as arguments. All five were reported as commands that
    would not run at all, which is the one thing this script's exit code is
    for, so the page read as broken when it was right.
    """

    def test_the_comment_is_taken_off(self, script: ModuleType) -> None:
        assert (
            script.without_comment("rel gaming   # all three, with the repairs")
            == "rel gaming"
        )

    def test_a_command_with_no_comment_is_untouched(self, script: ModuleType) -> None:
        assert script.without_comment("rel gaming") == "rel gaming"

    def test_a_hash_inside_a_quoted_argument_stays(self, script: ModuleType) -> None:
        # Cutting at the first hash would leave an unbalanced quote, so the
        # words after the cut would not be the words a shell reads.
        command = 'python x.py --label "a # in a name"'
        assert script.without_comment(command) == command

    def test_a_line_that_is_only_a_comment_is_dropped(self, script: ModuleType) -> None:
        assert script.without_comment("# just a note") == ""
        assert script.commands_in(["$ # just a note", "$ rel gaming"]) == ["rel gaming"]

    def test_the_command_it_finds_is_the_one_that_runs(
        self, script: ModuleType
    ) -> None:
        found = script.commands_in(["$ rel gaming    # all three"])
        assert found == ["rel gaming"]
        printed, _, trouble = script.run(found[0] + " --no-learn", 300.0)
        assert trouble == ""
        assert printed


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

    def test_a_command_that_reads_a_terminal_is_not_left_waiting(
        self, script: ModuleType
    ) -> None:
        """It would sit there for the whole budget and be called slow.

        Nothing documented here reads from a terminal, and the report for a
        command that did would say the budget was too small when the budget
        was never the problem.
        """
        printed, taken, trouble = script.run(
            'python -c "import sys; print(len(sys.stdin.read()))"', 30.0
        )
        assert trouble == ""
        assert printed.strip() == "0"
        assert taken < 10.0

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

    def test_the_index_names_commands_the_page_runs(self, script: ModuleType) -> None:
        """The one table this script cannot check, checked here instead.

        The closing table says which command is behind each table on the
        page, and its cells are commands rather than results, so it is set
        aside rather than checked. That left nobody looking at it. Five rows
        had dropped a setting, and four of the five sent a reader to a
        different table: the importance sampling row ran the default 1500
        episodes where the block runs 1200.
        """
        page = script.ROOT / "docs" / "algorithms.md"
        named = index_of(page, "| Table | Command |")
        assert len(named) > 15

        # Every console command, this script's own included. `read` drops
        # that one so a run does not run itself, and the index still has to
        # be allowed to name it.
        runs = set(console_commands(page))
        for command in named:
            assert command in runs, command

    def test_the_commands_behind_no_table_are_kept_apart(
        self, script: ModuleType
    ) -> None:
        # The rows that are examples rather than the source of a table. They
        # have their own heading, so a reader can tell which kind of row is
        # in front of them and the test above can hold the other kind.
        page = script.ROOT / "docs" / "algorithms.md"
        examples = index_of(page, "| What it does | Command |")
        assert examples
        assert not set(examples) & set(console_commands(page))

    def test_neither_index_reaches_a_claim(self, script: ModuleType) -> None:
        # Both are cells full of commands. A number in one of them is part of
        # a setting, so checking it would ask whether `--seed 7` is still
        # printed.
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        for claim in claims:
            for row in claim.rows:
                assert "scripts/" not in row, (claim.line, row)
                assert "`rel " not in row, (claim.line, row)

    def test_every_exempted_column_is_a_column_of_its_table(
        self, script: ModuleType
    ) -> None:
        """The guard on the four column markers the page carries.

        A marker names its column by heading. Rename the heading and the
        marker stops doing anything, and the only sign of it is six timings
        reported as numbers that moved. That is the noise the marker exists
        to stop, so this fails instead.
        """
        _, claims = script.read(script.ROOT / "docs" / "algorithms.md")
        marked = [claim for claim in claims if claim.skipped]
        assert marked
        for claim in marked:
            assert claim.unknown == [], f"line {claim.line}: {claim.headings}"

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
        """One of this project's own, because those are the only ones it runs.

        A page can name `git clone` or `pip install .`, and this neither runs
        them nor holds the page to them. What the exit code is for is a
        command of this project that has stopped working: a renamed script,
        or a flag that was taken away.
        """
        page = write(
            tmp_path,
            "```console\n$ python scripts/not-a-script.py\n```\n\n"
            "| a | b |\n| --- | --- |\n| x | 1 |\n",
        )
        assert self.go(script, monkeypatch, page) == 1

    def test_a_command_that_is_not_ours_is_zero(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Not run, so nothing is known about it, so it is not a defect this
        # can report. The report names it instead.
        page = write(
            tmp_path,
            "```console\n$ definitely-not-a-command\n```\n\n"
            "| a | b |\n| --- | --- |\n| x | 1 |\n",
        )
        assert self.go(script, monkeypatch, page) == 0

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

    def test_a_marker_that_never_closes_stops_the_run(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """The worst answer this script has, and it used to give it silently.

        A marker with no `-->` read to the bottom of the file, so every
        command and every table under it went into the reason. The report
        then said nothing was wrong with a page it had never read.
        """
        page = write(
            tmp_path,
            "<!-- not checked: a reason that never closes\n\n"
            "```console\n$ python x.py\n```\n\n"
            "| a | b |\n| --- | ---: |\n| x | 1.0 |\n",
        )
        with pytest.raises(SystemExit) as stopped:
            script.read(page)
        assert "line 1" in str(stopped.value)
        assert "-->" in str(stopped.value)

    def test_a_marker_that_closes_on_a_later_line_is_fine(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        # The guard is a blank line, not the second line, because a reason
        # worth giving is allowed to wrap as far as it needs to.
        page = write(
            tmp_path,
            "```console\n$ python x.py\n```\n\n"
            "<!-- not checked: one\ntwo\nthree\nfour -->\n\n"
            "| a | b |\n| --- | ---: |\n| x | 1.0 |\n",
        )
        _, claims = script.read(page)
        assert claims[0].exempt == "one two three four"

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
        assert claims[0].skipped == ["time"]

    def test_the_column_is_found_by_its_heading(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert claims[0].dropped == {2}

    def test_a_column_that_is_not_there_drops_nothing(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """A marker naming a column the table does not have does nothing.

        Read off the heading row rather than trusted, so a heading that gets
        renamed cannot leave a marker exempting whichever column moved into
        that position.
        """
        page = self.PAGE.replace("column time", "column seconds")
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].dropped == set()
        assert claims[0].numbers == ["15000", "324"]

    def test_a_column_that_is_not_there_is_named(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = self.PAGE.replace("column time", "column seconds")
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].unknown == ["seconds"]

    def test_a_column_that_is_there_is_not_named(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        _, claims = script.read(write(tmp_path, self.PAGE))
        assert claims[0].unknown == []

    def test_the_report_says_the_marker_did_nothing(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Otherwise the only sign is six timings listed as numbers that moved.

        That is the noise the marker exists to stop, arriving with no hint of
        where it came from. A person reading it has no way to tell a renamed
        heading from a result that really changed.
        """
        page = write(tmp_path, self.PAGE.replace("column time", "column seconds"))
        monkeypatch.setattr(
            sys, "argv", ["check_numbers", "--doc", str(page), "--list"]
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()

        printed = capsys.readouterr().out
        assert "1 markers name a column their table has not got" in printed
        assert "seconds" in printed

    def test_naming_no_column_still_exempts_the_whole_table(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = self.PAGE.replace("not checked, column time:", "not checked:")
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].exempt == "seconds belong to the machine"
        assert claims[0].skipped == []

    def test_the_exemption_does_not_carry_to_the_next_table(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = self.PAGE + "\nProse.\n\n| a | time |\n| --- | ---: |\n| x | 9 |\n"
        _, claims = script.read(write(tmp_path, page))
        assert [claim.skipped for claim in claims] == [["time"], []]
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
        assert "not time" in printed
        assert "1 numbers are checked" in printed

    def test_the_reasons_go_under_the_table_rather_than_in_it(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A column is as wide as its widest cell.

        The widest reason on the real page is two hundred characters, and
        holding it in a column pushed the command that made each table off
        the side of the terminal. A report nobody can read is a report
        nobody reads.
        """
        reason = "a reason long enough to push everything after it off the screen"
        page = write(
            tmp_path,
            "```console\n$ python x.py\n```\n\n"
            f"<!-- not checked: {reason} -->\n\n"
            "| a | b |\n| --- | ---: |\n| x | 9 |\n",
        )
        self.go(script, monkeypatch, page)
        printed = capsys.readouterr().out

        rows = [line for line in printed.splitlines() if "python x.py" in line]
        assert rows and all(reason not in row for row in rows)
        assert "1 tables ask not to be checked, and say why:" in printed
        assert reason in printed

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


class TestACommandThatRanOutOfTime:
    """The slowest commands on the page are why the cache exists.

    One of them takes over half an hour. A run that recorded nothing about
    running out of time re-ran every one of those, at the full budget each,
    so resuming cost the thing the cache is for.
    """

    PAGE = (
        '```console\n$ python -c "import time; time.sleep(30)"\n```\n\n'
        "| a | b |\n| --- | --- |\n| x | 7.5 |\n"
    )
    COMMAND = 'python -c "import time; time.sleep(30)"'

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        cache: Path,
        budget: str,
    ) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_numbers",
                "--doc",
                str(page),
                "--all",
                "--cache",
                str(cache),
                "--timeout",
                budget,
            ],
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        return int(script.main())

    def test_the_budget_it_gave_up_at_is_written_down(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cache = tmp_path / "kept.json"
        self.go(script, monkeypatch, write(tmp_path, self.PAGE), cache, "1")
        capsys.readouterr()

        held = json.loads(cache.read_text())["printed"][self.COMMAND]
        assert held["gave_up"] == 1.0
        assert held["numbers"] == []

    def test_a_command_that_finished_says_so(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Zero rather than absent, so the two are told apart by their value
        # and not by whether somebody remembered to write the key.
        cache = tmp_path / "kept.json"
        page = write(
            tmp_path,
            '```console\n$ python -c "print(7.5)"\n```\n\n'
            "| a | b |\n| --- | --- |\n| x | 7.5 |\n",
        )
        self.go(script, monkeypatch, page, cache, "60")
        capsys.readouterr()

        held = json.loads(cache.read_text())["printed"]['python -c "print(7.5)"']
        assert held["gave_up"] == 0.0

    def test_the_same_budget_does_not_run_it_again(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        self.go(script, monkeypatch, page, cache, "1")
        capsys.readouterr()

        started = time.perf_counter()
        self.go(script, monkeypatch, page, cache, "1")
        spent = time.perf_counter() - started

        printed = capsys.readouterr().out
        assert "1 of them ran out of time and are not run again" in printed
        assert f"{script.TIMED_OUT} 1s before" in printed
        assert spent < 1.0

    def test_a_smaller_budget_does_not_run_it_again(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # It took more than two seconds, so it takes more than one.
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        self.go(script, monkeypatch, page, cache, "2")
        capsys.readouterr()

        self.go(script, monkeypatch, page, cache, "1")
        assert "1 of them ran out of time" in capsys.readouterr().out

    def test_a_larger_budget_runs_it_again(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The only way a command ever gets out of this.

        Two seconds says nothing about three, so a person raising the budget
        has to get the command run rather than the answer from last time.
        """
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        self.go(script, monkeypatch, page, cache, "1")
        capsys.readouterr()

        self.go(script, monkeypatch, page, cache, "2")
        printed = capsys.readouterr().out
        assert "ran out of time and are not run again" not in printed
        assert f"{script.TIMED_OUT} 2s, so nothing below" in printed

    def test_the_summary_names_the_budget_each_one_had(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A command remembered from a larger budget than this run offers
        # would otherwise be listed under this run's budget, which is not the
        # number to raise `--timeout` past.
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        self.go(script, monkeypatch, page, cache, "2")
        capsys.readouterr()

        self.go(script, monkeypatch, page, cache, "1")
        printed = capsys.readouterr().out
        assert "took longer than the budget they were given" in printed
        assert f"  2s: {self.COMMAND}" in printed

    def test_it_is_not_counted_as_a_command_that_will_not_run(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Running out of time is this script's budget rather than anything
        # about the page, so it is not what the exit code is for.
        cache = tmp_path / "kept.json"
        page = write(tmp_path, self.PAGE)
        assert self.go(script, monkeypatch, page, cache, "1") == 0
        capsys.readouterr()
        assert self.go(script, monkeypatch, page, cache, "1") == 0
        assert "would not run at all" not in capsys.readouterr().out


class TestItRunsOnlyThisProjectsCommands:
    """Checking a document must not change the machine it is checked on.

    `--doc` takes any page, and the readme's install block holds `git clone`
    and `pip install .`. Pointing this at the readme ran both of them to
    check a table of line counts: it cloned the repository into itself and
    installed the package. A rule that has to be remembered is not a rule.
    """

    PAGE = (
        "```console\n$ pip install .\n$ python scripts/lines.py\n```\n\n"
        "| a | b |\n| --- | --- |\n| x | 1 |\n"
    )

    def test_the_rule_is_what_the_source_says(self, script: ModuleType) -> None:
        # Read out of the file, because the fixture at the top of this module
        # widens the tuple in the loaded script for every other test here.
        found = re.search(r"^OURS = (\(.*?\))$", SCRIPT.read_text(), re.MULTILINE)
        assert found is not None
        assert ast.literal_eval(found[1]) == (
            "python scripts/",
            "python -m rel",
            "rel ",
        )

    def test_the_project_owns_its_own_commands(self, script: ModuleType) -> None:
        assert script.is_ours("python scripts/measure_agents.py --runs 10")
        assert script.is_ours("python -m rel train sarsa")
        assert script.is_ours("rel gaming")

    def test_everything_else_is_not_ours(self, script: ModuleType) -> None:
        for command in (
            "pip install .",
            "git clone https://example.com/a/b",
            "cd somewhere",
            "pytest",
        ):
            assert not script.is_ours(command), command

    def test_it_does_not_run_them(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = write(tmp_path, self.PAGE)
        monkeypatch.setattr(sys, "argv", ["check_numbers", "--doc", str(page), "--all"])
        monkeypatch.setattr(script, "ROOT", page.parent)

        ran: list[str] = []
        real = script.run
        monkeypatch.setattr(
            script,
            "run",
            lambda command, seconds: (ran.append(command), real(command, seconds))[1],
        )
        script.main()

        assert "pip install ." not in ran
        printed = capsys.readouterr().out
        assert "1 commands are not this project's, so none is run" in printed
        assert "pip install ." in printed

    def test_the_listing_names_them_too(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # `--list` runs nothing either way, and a reader deciding whether to
        # start a run wants to know what it will leave out.
        page = write(tmp_path, self.PAGE)
        monkeypatch.setattr(
            sys, "argv", ["check_numbers", "--doc", str(page), "--list"]
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()
        assert "pip install ." in capsys.readouterr().out


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

    def test_the_same_source_is_parsed_once(self, script: ModuleType) -> None:
        """Why the parse is held on to at all.

        Every command's fingerprint parses the whole package, so a run of
        this page parsed the same fifty six modules sixty three times. That
        was fourteen seconds of a run whose whole point is to cost nothing
        when everything is cached.
        """
        package = len(list(script.ROOT.glob("rel/**/*.py")))
        script.without_prose.cache_clear()

        script.fingerprint("python scripts/measure_sweeping.py")
        assert script.without_prose.cache_info().misses == package + 1

        script.fingerprint("python scripts/measure_importance.py")
        after = script.without_prose.cache_info()
        # The package again, and one script it has not seen before. Only the
        # script was parsed: the package came back out of the cache whole.
        assert after.misses == package + 2
        assert after.hits == package

    def test_a_file_that_changes_is_parsed_again(self, script: ModuleType) -> None:
        # Held on to by source text and not by path, so a file edited under a
        # running process moves the fingerprint. Keying on the path would
        # make the cache lie about exactly what it exists to catch.
        before = script.fingerprint()
        path = script.ROOT / "rel" / "agents" / "basis.py"
        was = path.read_text()
        try:
            path.write_text(was.replace("return optimism", "return optimism * 3", 1))
            assert script.fingerprint() != before
        finally:
            path.write_text(was)
        assert script.fingerprint() == before

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

    def test_the_count_of_numbers_in_prose_matches(self, script: ModuleType) -> None:
        """The page says how much of itself this tool cannot see at all.

        A table cell is a result by construction and a sentence is not, so
        the tool reads tables. That leaves a quarter of the page's numbers
        unread, and saying so is the difference between a limitation and a
        blind spot.
        """
        page = script.ROOT / "docs" / "algorithms.md"
        found = re.search(
            r"\*\*(\d+) of this\n  page's numbers are in prose", page.read_text()
        )
        assert found is not None, "the page no longer says how many are in prose"
        assert int(found[1]) == prose_numbers(script, page)

    def test_the_count_it_is_measured_against_matches_too(
        self, script: ModuleType
    ) -> None:
        """The third number in the same sentence, which nothing was holding.

        It said 1074 in cells where the parsing found 1067, and the two moved
        apart again the moment a section was added. Two of the three numbers
        in that paragraph were held and the third was not, which is how a
        sentence about counts going stale ends up with one that has.
        """
        page = script.ROOT / "docs" / "algorithms.md"
        found = re.search(r"against (\d+) in cells", page.read_text())
        assert found is not None, "the page no longer says how many are in cells"

        _, claims = script.read(page)
        assert int(found[1]) == sum(len(claim.numbers) for claim in claims)

    def test_it_counts_neither_tables_nor_fences_as_prose(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        # Otherwise the count above would be the whole page and would say
        # nothing about what the tool misses.
        page = write(
            tmp_path,
            "Two numbers here, 1 and 2.\n\n"
            "```console\n$ python x.py --runs 3\n```\n\n"
            "| a | b |\n| --- | --- |\n| 4 | 5 |\n\n"
            "<!-- not checked: 6 -->\n\nAnd 7.\n",
        )
        assert prose_numbers(script, page) == 3

    def test_a_marker_can_name_two_columns(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """A table can have more than one column nothing can check.

        The corridor table has both: a rule column whose cells hold settings
        like `softmax:0.02`, and a time column that belongs to the machine.
        Exempting the table whole would drop the four results between them.
        """
        page = (
            "```console\n$ python x.py\n```\n\n"
            "<!-- not checked, column rule, time: the rule column holds settings\n"
            "and the time column belongs to the machine -->\n\n"
            "| rule | steps | time |\n| --- | ---: | ---: |\n"
            "| `softmax:0.02` | 20 | 31s |\n"
        )
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].skipped == ["rule", "time"]
        assert claims[0].dropped == {0, 2}
        assert claims[0].numbers == ["20"]

    def test_a_marker_does_not_carry_across_a_console_block(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """A marker applies to the table straight after it.

        Prose between the two already reset it. A console block did not, so
        a marker that never got its table would carry down the page and
        quietly exempt somebody else's, which is the silent failure the
        marker exists to avoid.
        """
        page = (
            "<!-- not checked: this marker has no table -->\n\n"
            "```console\n$ python x.py\n```\n\n"
            "| a | b |\n| --- | ---: |\n| x | 1.0 |\n"
        )
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].exempt == ""

    def test_a_column_marker_does_not_carry_either(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = (
            "<!-- not checked, column time: no table follows this -->\n\n"
            "```console\n$ python x.py\n```\n\n"
            "| a | time |\n| --- | ---: |\n| x | 9 |\n"
        )
        _, claims = script.read(write(tmp_path, page))
        assert claims[0].skipped == []
        assert claims[0].numbers == ["9"]


class TestSeveralPatternsAtOnce:
    """`--only` takes a list, because continuous integration runs a subset.

    One pattern at a time would mean one job per command, and each of those
    would re-read the page and re-run nothing else. The list is what lets a
    named handful of fast commands be one run.
    """

    PAGE = (
        "```console\n"
        '$ python -c "print(1.5)"\n'
        "```\n\n| a |\n| --- |\n| 1.5 |\n\n"
        "```console\n"
        '$ python -c "print(2.5)"\n'
        "```\n\n| b |\n| --- |\n| 2.5 |\n\n"
        "```console\n"
        '$ python -c "print(3.5)"\n'
        "```\n\n| c |\n| --- |\n| 3.5 |\n"
    )

    def run_with(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *rest: str,
    ) -> int:
        page = tmp_path / "page.md"
        page.write_text(self.PAGE)
        monkeypatch.setattr(sys, "argv", ["check_numbers", "--doc", str(page), *rest])
        monkeypatch.setattr(script, "ROOT", page.parent)
        return int(script.main())

    def test_one_pattern_runs_one_command(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self.run_with(script, monkeypatch, tmp_path, "--only", "1.5")
        assert "1 of 1 tables" in capsys.readouterr().out

    def test_two_patterns_run_two(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self.run_with(script, monkeypatch, tmp_path, "--only", "1.5", "3.5")
        assert "2 of 2 tables" in capsys.readouterr().out

    def test_a_pattern_that_matches_nothing_is_an_error(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Rather than a run that quietly checks less than it was asked to.

        A job that names a subset of the page and one of those names stops
        matching would keep passing while covering nothing, which is the one
        way a check like this fails without saying so.
        """
        got = self.run_with(script, monkeypatch, tmp_path, "--only", "1.5", "nope")
        assert got == 1
        assert "match no command" in capsys.readouterr().out

    def test_every_pattern_matching_is_not_an_error(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        assert self.run_with(script, monkeypatch, tmp_path, "--only", "1.5") == 0


class TestStrictFailsOnAStaleNumber:
    """What makes this script a gate rather than a report.

    Without `--strict` a table nothing accounts for is printed and the run
    exits zero, which is right for a person reading it and wrong for a job
    whose whole output is a tick.
    """

    def run_with(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        page: str,
        *rest: str,
    ) -> int:
        written = tmp_path / "page.md"
        written.write_text(page)
        monkeypatch.setattr(
            sys, "argv", ["check_numbers", "--doc", str(written), "--all", *rest]
        )
        monkeypatch.setattr(script, "ROOT", written.parent)
        return int(script.main())

    RIGHT = '```console\n$ python -c "print(1.5)"\n```\n\n| a |\n| --- |\n| 1.5 |\n'
    STALE = '```console\n$ python -c "print(1.5)"\n```\n\n| a |\n| --- |\n| 2.5 |\n'

    def test_a_page_that_holds_passes(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert self.run_with(script, monkeypatch, tmp_path, self.RIGHT, "--strict") == 0

    def test_a_stale_number_fails(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        assert self.run_with(script, monkeypatch, tmp_path, self.STALE, "--strict") == 1

    def test_without_strict_the_same_page_passes(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Which is the behaviour a person wants and a job does not.
        assert self.run_with(script, monkeypatch, tmp_path, self.STALE) == 0

    def test_a_command_that_ran_out_of_time_fails_it_too(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A subset that cannot finish inside its budget is a subset that
        # checked nothing, and a job cannot tell that from a tick.
        slow = (
            "```console\n"
            '$ python -c "import time; time.sleep(5); print(1.5)"\n'
            "```\n\n| a |\n| --- |\n| 1.5 |\n"
        )
        got = self.run_with(
            script, monkeypatch, tmp_path, slow, "--strict", "--timeout", "0.5"
        )
        assert got == 1

    def test_it_says_which_of_the_three_went_wrong(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self.run_with(script, monkeypatch, tmp_path, self.STALE, "--strict")
        printed = capsys.readouterr().out
        assert "Strict: 1 tables are not wholly accounted for" in printed

    def test_a_page_that_holds_says_so(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self.run_with(script, monkeypatch, tmp_path, self.RIGHT, "--strict")
        assert "Strict: all 1 tables are accounted for" in capsys.readouterr().out

    def test_a_run_that_checked_no_table_fails(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The one way a check like this passes while covering nothing.

        The commands are real and no table sits under any of them, which is
        what an edit that moves a table away from its command leaves behind.
        Every other way of asking says the run was clean.
        """
        page = '```console\n$ python -c "print(1.5)"\n```\n\nNo table here.\n'
        got = self.run_with(script, monkeypatch, tmp_path, page, "--strict")
        assert got == 1
        assert "no table was checked at all" in capsys.readouterr().out

    def test_without_strict_the_same_run_passes(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        page = '```console\n$ python -c "print(1.5)"\n```\n\nNo table here.\n'
        assert self.run_with(script, monkeypatch, tmp_path, page) == 0


class TestRunningSeveralCommandsAtOnce:
    """A whole run of the page is about three hours on one processor.

    Every command here is seeded and prints the same numbers whatever else is
    running, so running four at a time changes the wall clock and nothing
    else. These hold that: the same report, the same exit code, the same
    cache, from the same page.
    """

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        capsys: pytest.CaptureFixture[str],
        *rest: str,
    ) -> tuple[int, str]:
        monkeypatch.setattr(
            sys, "argv", ["check_numbers", "--doc", str(page), "--all", *rest]
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        code = int(script.main())
        return code, capsys.readouterr().out

    def a_page(self, tmp_path: Path) -> Path:
        return write(
            tmp_path,
            "".join(
                f'```console\n$ python -c "print({value})"\n```\n\n'
                f"| a | b |\n| --- | --- |\n| x | {value} |\n\n"
                for value in (1.5, 2.5, 3.5, 4.5, 5.5)
            ),
        )

    def test_it_finds_what_one_at_a_time_finds(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = self.a_page(tmp_path)
        one, alone = self.go(script, monkeypatch, page, capsys, "--jobs", "1")
        many, together = self.go(script, monkeypatch, page, capsys, "--jobs", "4")

        assert one == many
        assert "5 of 5 tables" in alone
        assert "5 of 5 tables" in together

    def test_every_command_is_run_exactly_once(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = self.a_page(tmp_path)
        _, printed = self.go(script, monkeypatch, page, capsys, "--jobs", "4")
        for value in (1.5, 2.5, 3.5, 4.5, 5.5):
            assert printed.count(f"print({value})") == 1

    def test_the_count_in_front_reaches_the_total(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # It counts how many have finished rather than where a command sits
        # on the page, because with several running at once there is no other
        # order to count in. It still has to reach the total exactly once.
        page = self.a_page(tmp_path)
        _, printed = self.go(script, monkeypatch, page, capsys, "--jobs", "4")
        for number in range(1, 6):
            assert printed.count(f"[{number}/5]") == 1

    def test_a_commands_two_lines_are_not_split_apart(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The whole report for a command is printed under the lock, so the
        # line naming it is always followed by the line about it.
        page = self.a_page(tmp_path)
        _, printed = self.go(script, monkeypatch, page, capsys, "--jobs", "4")
        lines = [one for one in printed.splitlines() if one.strip()]
        for index, line in enumerate(lines):
            if line.startswith("["):
                assert lines[index + 1].startswith("    "), lines[index : index + 2]

    def test_the_cache_holds_every_command_afterwards(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Two threads writing the same file would leave a half of each, so
        # this is the test that the lock is really around the write.
        page = self.a_page(tmp_path)
        kept = tmp_path / "outputs.json"
        self.go(script, monkeypatch, page, capsys, "--jobs", "4", "--cache", str(kept))

        held = json.loads(kept.read_text())["printed"]
        assert len(held) == 5
        for value in (1.5, 2.5, 3.5, 4.5, 5.5):
            name = f'python -c "print({value})"'
            assert held[name]["numbers"] == [str(value)]

    def test_a_cache_written_by_four_is_read_by_one(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = self.a_page(tmp_path)
        kept = tmp_path / "outputs.json"
        self.go(script, monkeypatch, page, capsys, "--jobs", "4", "--cache", str(kept))
        _, again = self.go(
            script, monkeypatch, page, capsys, "--jobs", "1", "--cache", str(kept)
        )
        assert "5 of 5 commands come from" in again

    def test_one_at_a_time_is_what_it_does_unasked(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # So a run that asks for nothing is the run it always was.
        page = self.a_page(tmp_path)
        _, printed = self.go(script, monkeypatch, page, capsys)
        assert "at a time" not in printed


class TestTheLongestGoFirst:
    """Four at a time finishes when the last one does.

    A command that takes twenty five minutes and starts last adds twenty five
    minutes to the whole run. Over the times of a real run of this page, page
    order takes 74 minutes on four processors and longest first takes 62,
    which is the total divided by four and therefore the floor.

    The times come from the cache and are read whatever the code stamp says,
    which the numbers beside them are not. That is the point: the usual reason
    for a whole run is that something changed, so every output is stale, and
    if the times went stale with them there would be nothing to order by.
    """

    PAGE = "".join(
        f'```console\n$ python -c "print({value})"\n```\n\n'
        f"| a | b |\n| --- | --- |\n| x | {value} |\n\n"
        for value in (1.5, 2.5, 3.5)
    )

    def go(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        page: Path,
        cache: Path,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["check_numbers", "--doc", str(page), "--all", "--cache", str(cache)],
        )
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()

    def test_a_run_records_how_long_each_took(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = write(tmp_path, self.PAGE)
        cache = tmp_path / "kept.json"
        self.go(script, monkeypatch, page, cache)
        capsys.readouterr()

        held = json.loads(cache.read_text())["printed"]
        assert len(held) == 3
        for entry in held.values():
            assert "spent" in entry
            assert entry["spent"] >= 0.0

    def test_the_time_is_read_even_when_the_code_has_moved(
        self, script: ModuleType
    ) -> None:
        entry = {"code": "a stamp from another version", "numbers": [], "spent": 12.5}
        assert script.seconds_of(entry) == 12.5

    def test_an_entry_from_before_this_says_nothing(self, script: ModuleType) -> None:
        assert script.seconds_of({"code": "x", "numbers": []}) == 0.0

    def test_the_slowest_is_run_first(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        page = write(tmp_path, self.PAGE)
        cache = tmp_path / "kept.json"

        # A cache whose outputs are all stale and whose times say the middle
        # command is the slow one.
        names = [f'python -c "print({value})"' for value in (1.5, 2.5, 3.5)]
        cache.write_text(
            json.dumps(
                {
                    "printed": {
                        names[0]: {"code": "moved", "numbers": [], "spent": 1.0},
                        names[1]: {"code": "moved", "numbers": [], "spent": 99.0},
                        names[2]: {"code": "moved", "numbers": [], "spent": 5.0},
                    }
                }
            )
        )

        self.go(script, monkeypatch, page, cache)
        printed = capsys.readouterr().out

        ran = [
            line.split("] ", 1)[1]
            for line in printed.splitlines()
            if line.startswith("[")
        ]
        assert ran[0] == names[1], ran

    def test_a_first_run_keeps_the_order_the_page_has(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # With no cache nothing is known about any of them, so the ordering is
        # paid for by a run that has already happened.
        page = write(tmp_path, self.PAGE)
        monkeypatch.setattr(sys, "argv", ["check_numbers", "--doc", str(page), "--all"])
        monkeypatch.setattr(script, "ROOT", page.parent)
        script.main()

        printed = capsys.readouterr().out
        ran = [
            line.split("] ", 1)[1]
            for line in printed.splitlines()
            if line.startswith("[")
        ]
        assert ran == [f'python -c "print({value})"' for value in (1.5, 2.5, 3.5)]
