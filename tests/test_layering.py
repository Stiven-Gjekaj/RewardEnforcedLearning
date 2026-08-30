"""Rules about which module may import which, enforced by reading the imports.

Four claims in the documentation rest on these, and none of them can be
checked by running an agent:

- The package imports nothing outside the standard library.
- An environment never imports an agent, and an agent never imports an
  environment. An agent that knew which environment it was in could be tuned
  for it, and the tuning would be invisible in the returns.
- Nothing that decides anything imports the drawing code, so a whole
  experiment runs with no terminal attached.
- Every module of the package is counted in the table of sizes in the readme.
  A total that quietly leaves a module out is worse than no total, and two of
  them had been missing from it for as long as they had existed.

These are read out of the source with the standard library's own parser rather
than by importing anything, so a module that would fail to import is still
checked.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "rel"


def modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def imported_by(path: Path) -> set[str]:
    """Every top level module name that this file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import stays inside the package.
                continue
            if node.module:
                names.add(node.module)

    return names


def test_there_is_something_to_check() -> None:
    assert len(modules()) > 15


@pytest.mark.parametrize("path", modules(), ids=lambda path: path.name)
class TestNothingOutsideTheStandardLibrary:
    def test_every_import_is_the_standard_library_or_this_package(
        self, path: Path
    ) -> None:
        # This is the claim the README makes first. A dependency added by
        # habit would not fail any other test: it would install, import and
        # work, and the claim would simply have stopped being true.
        for name in imported_by(path):
            root = name.split(".")[0]
            assert root == "rel" or root in sys.stdlib_module_names, (
                f"{path.name} imports {name}, which is neither this package "
                f"nor the standard library"
            )


class TestTheLayers:
    def test_an_environment_never_imports_an_agent(self) -> None:
        for path in PACKAGE.joinpath("envs").rglob("*.py"):
            for name in imported_by(path):
                assert not name.startswith("rel.agents"), f"{path.name}: {name}"

    def test_an_agent_never_imports_an_environment(self) -> None:
        # The registry in rel/agents/__init__.py is not an agent. It reads two
        # numbers off an environment to build one, and it is the only file
        # under this directory allowed to know that environments exist. Even
        # it imports the contract rather than any particular environment.
        for path in PACKAGE.joinpath("agents").rglob("*.py"):
            for name in imported_by(path):
                assert not name.startswith("rel.envs"), f"{path.name}: {name}"

    def test_nothing_that_decides_imports_the_drawing(self) -> None:
        for area in ("envs", "agents"):
            for path in PACKAGE.joinpath(area).rglob("*.py"):
                for name in imported_by(path):
                    assert not name.startswith("rel.ui"), f"{path.name}: {name}"

    def test_the_core_contract_imports_nothing_from_the_layers(self) -> None:
        # rel/core.py is what both sides agree on. If it reached into either
        # of them, the agreement would have a direction and the layering would
        # be a suggestion.
        for name in imported_by(PACKAGE / "core.py"):
            assert not name.startswith(("rel.envs", "rel.agents", "rel.ui"))


def test_the_readme_says_how_many_tests_there_are() -> None:
    """The readme states the count in three places and nothing held it.

    It was right when it was written and every test added since made it
    less right. A number written by hand beside a number a command produces
    is the fault this project wrote `scripts/check_numbers.py` for, and that
    checker reads `docs/algorithms.md` rather than this file.

    Collection is asked for in a separate interpreter, because asking the
    running session how many tests it holds gives a different answer under
    `pytest -k something`, and a test that passes or fails by how it was
    invoked is worse than no test.
    """
    root = PACKAGE.parent
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert done.returncode == 0, done.stdout[-2000:]
    counted = re.search(r"(\d+) tests collected", done.stdout)
    assert counted is not None, done.stdout[-2000:]
    collected = counted.group(1)

    readme = (root / "README.md").read_text()
    said = re.findall(r"(\d{4}) tests", readme) + re.findall(
        r"tests-(\d{4})_passing", readme
    )
    assert len(said) == 3, said
    for one in said:
        assert one == collected, f"the readme says {one} and there are {collected}"


#: The numbers under a hundred, written the way the readme writes them.
ONES = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def spelled(count: int) -> str:
    """A count under a hundred, in words."""
    if count < 20:
        return ONES[count - 1]
    tens, units = divmod(count, 10)
    return TENS[tens - 2] if not units else f"{TENS[tens - 2]} {ONES[units - 1]}"


def test_the_speller_writes_what_the_readme_writes() -> None:
    # The check above passes if the readme holds the string this makes, so a
    # speller that was wrong in the same way as the readme would agree with
    # it and say nothing.
    assert [spelled(count) for count in (9, 15, 19, 20, 21, 34, 40, 99)] == [
        "nine",
        "fifteen",
        "nineteen",
        "twenty",
        "twenty one",
        "thirty four",
        "forty",
        "ninety nine",
    ]


def test_the_readme_counts_the_agents_and_the_environments() -> None:
    """Two more numbers written in words beside a registry that can count.

    They are right today and nothing was holding them. Every other count
    about this repository has gone stale at least once: the readme said 2083
    tests where the suite collects 2119, and the untested scripts were
    eleven in one sentence and twelve in the next.

    The registries are imported here rather than at the top of the file,
    because every other test in it reads the source with the parser instead
    of importing anything, and that is what lets them check a module which
    would fail to import.
    """
    from rel.agents import AGENTS
    from rel.envs import ENVIRONMENTS

    readme = (PACKAGE.parent / "README.md").read_text()
    assert f"{spelled(len(ENVIRONMENTS))} environments" in readme, len(ENVIRONMENTS)
    assert f"{spelled(len(AGENTS))} agents" in readme, len(AGENTS)


def test_both_documents_count_the_networks() -> None:
    """The layer list said two networks and there have been three since the
    value network track. The readme said three all along, so the two
    documents disagreed and neither was held to the code."""
    layers = ast.parse((PACKAGE / "nn" / "layers.py").read_text())
    networks = [
        node.name
        for node in layers.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("Network")
    ]
    assert len(networks) >= 2, networks

    said = spelled(len(networks))
    for name in ("README.md", "docs/architecture.md"):
        text = (PACKAGE.parent / name).read_text()
        assert f"{said} networks" in text, (name, said)


def test_the_milestones_name_the_scripts_that_have_no_test() -> None:
    """Named rather than counted, because the count went wrong twice.

    The first version of that entry said eleven untested scripts in its
    heading and twelve in its next sentence, and the real number was nine.
    A list of names is checkable and a count is only re-countable.

    A script counts as tested when some test loads it by its path, which is
    how `scripts/` is used from a test: it is not a package, so importing it
    as one would be a different arrangement than the one that runs.
    """
    root = PACKAGE.parent
    scripts = {found.name for found in (root / "scripts").glob("*.py")}
    tested = {
        f"{name}.py"
        for text in (found.read_text() for found in (root / "tests").glob("*.py"))
        for name in re.findall(r"scripts[\"/\s]+([a-z_]+)\.py", text)
    }

    #: The two that measure nothing. One draws the readme banner and one runs
    #: the gate that continuous integration runs, so neither belongs in a
    #: note about measurement scripts without tests.
    apart = {"make_banner.py", "verify.py"}

    # The one sentence that lists them, rather than every mention of a
    # measurement script in the file. Reading the whole file would let a
    # script named in that sentence gain a test with nothing complaining,
    # because the name would still be in the file somewhere else.
    note = (root / "docs" / "milestones.md").read_text()
    listed = note.split("So the seven are", 1)
    assert len(listed) == 2, "the milestones no longer list them"
    said = set(re.findall(r"`(measure_[a-z_]+\.py)`", listed[1].split(".\n\n")[0]))

    assert said == scripts - tested - apart, sorted(said ^ (scripts - tested - apart))


def test_every_module_is_counted_in_the_size_table() -> None:
    """`scripts/lines.py` is what the readme table is copied from.

    It adds up the areas it is given and says nothing about a file in none of
    them, so a new module is simply absent from a total that still looks like
    a total. This loads the script by its path, because `scripts/` is not a
    package and importing it as one would be a different arrangement than the
    one that runs.
    """
    path = PACKAGE.parent / "scripts" / "lines.py"
    spec = importlib.util.spec_from_file_location("lines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    left_out = [str(found.relative_to(PACKAGE.parent)) for found in module.missed()]
    assert left_out == []
