"""The parsing in `scripts/check_numbers.py`, which is where it can go wrong.

The script asks whether a number the documentation states still comes out of
the command above it. Everything hard about that is in deciding which command
a table belongs to and which characters in a cell are a number, and both of
those are decided here rather than by running anything.

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


class TestWhichTableBelongsToWhichCommand:
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

    def test_every_block_is_found(self, script: ModuleType, tmp_path: Path) -> None:
        blocks = script.read(write(tmp_path, self.PAGE))
        assert [block.commands for block in blocks] == [
            ["python first.py"],
            ["python second.py", "python third.py"],
        ]

    def test_a_heading_does_not_end_the_tables(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """The page states a command once and then discusses what it printed
        under several headings. Stopping at the first heading would drop most
        of the numbers on the page and call the rest of them checked."""
        blocks = script.read(write(tmp_path, self.PAGE))
        assert len(blocks[0].claims) == 2
        assert blocks[0].numbers == 2

    def test_the_next_command_does_end_them(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        blocks = script.read(write(tmp_path, self.PAGE))
        assert len(blocks[1].claims) == 1

    def test_a_table_before_any_command_belongs_to_nothing(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        blocks = script.read(write(tmp_path, "| a | 1 |\n\n```console\n$ x\n```\n"))
        assert blocks[0].numbers == 0

    def test_a_block_that_is_not_console_is_not_a_command(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        page = "```python\nx = 1\n```\n\n| a | 1 |\n"
        assert script.read(write(tmp_path, page)) == []

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
        assert script.read(write(tmp_path, page))[0].numbers == 0


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
    def claim(self, script: ModuleType, row: str) -> object:
        made = script.Claim(1)
        made.rows = [row]
        return made

    def test_a_number_anywhere_in_the_output_counts(self, script: ModuleType) -> None:
        claim = self.claim(script, "| a | -27.70 |")
        assert script.missing(claim, "sarsa   -27.70 +/- 1.64") == []

    def test_a_number_that_moved_is_reported(self, script: ModuleType) -> None:
        claim = self.claim(script, "| a | -27.70 |")
        assert script.missing(claim, "sarsa   -28.10") == ["-27.70"]

    def test_the_same_number_twice_needs_it_twice(self, script: ModuleType) -> None:
        """Counted with multiplicity on purpose. A table that states -13.00
        four times against an output that prints it twice has changed, and
        matching on membership alone would call it unchanged."""
        claim = self.claim(script, "| a | -13.00 | -13.00 |")
        assert script.missing(claim, "one -13.00 line") == ["-13.00"]
        assert script.missing(claim, "-13.00 and -13.00") == []


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
        assert "took longer than" in trouble
        assert taken < 5.0


class TestAgainstTheRealPage:
    def test_every_block_has_a_command(self, script: ModuleType) -> None:
        for block in script.read(script.ROOT / "docs" / "algorithms.md"):
            assert block.commands
            assert all(command for command in block.commands)

    def test_the_page_states_a_great_many_numbers(self, script: ModuleType) -> None:
        # The point of the exercise. If this drops to nothing, the parsing
        # broke and the script would report a clean page either way.
        blocks = script.read(script.ROOT / "docs" / "algorithms.md")
        assert sum(block.numbers for block in blocks) > 800

    def test_no_claim_holds_a_command_line(self, script: ModuleType) -> None:
        # The closing table of the page is a list of commands. Every number in
        # it is a setting rather than a result, so none of its rows may reach
        # a claim.
        for block in script.read(script.ROOT / "docs" / "algorithms.md"):
            for claim in block.claims:
                for row in claim.rows:
                    assert "scripts/measure" not in row, row
