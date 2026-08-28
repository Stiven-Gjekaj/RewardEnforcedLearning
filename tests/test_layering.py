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
