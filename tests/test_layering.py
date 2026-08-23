"""Rules about which module may import which, enforced by reading the imports.

Three claims in the documentation rest on these, and none of them can be
checked by running an agent:

- The package imports nothing outside the standard library.
- An environment never imports an agent, and an agent never imports an
  environment. An agent that knew which environment it was in could be tuned
  for it, and the tuning would be invisible in the returns.
- Nothing that decides anything imports the drawing code, so a whole
  experiment runs with no terminal attached.

These are read out of the source with the standard library's own parser rather
than by importing anything, so a module that would fail to import is still
checked.
"""

from __future__ import annotations

import ast
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
