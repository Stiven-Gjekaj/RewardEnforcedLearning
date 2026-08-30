"""The gate a developer runs, against the gate continuous integration runs.

`scripts/verify.py` says it runs what CI runs. It said so while running four of
the six things CI runs: the link check was missing, and so was the job that
builds an interpreter with nothing but this package in it.

The second one is missing on purpose and the file says why. The first was
missing because nobody compared the two lists, so this compares them.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

#: Steps of the workflow this gate does not run, and why each one is left out.
#: `pip install` is how CI gets the tools that a developer already has. The
#: rest is the `standalone` job, which needs a fresh virtual environment and a
#: network to prove something `tests/test_layering.py` proves by reading the
#: imports.
SKIPPED = ("pip install", "venv", "/tmp/clean")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflow_commands() -> list[str]:
    """Every single line `run:` in the workflow, in the order it appears."""
    found = re.findall(r"^\s+run: (.+)$", WORKFLOW.read_text(), flags=re.MULTILINE)
    return [line.strip() for line in found if not line.strip().startswith("|")]


class TestTheGateMatchesTheWorkflow:
    def test_the_workflow_still_has_steps_to_compare(self) -> None:
        # If the workflow were rewritten into a form this cannot read, every
        # test below would pass by finding nothing.
        assert len(workflow_commands()) >= 5

    def test_every_step_it_does_not_skip_is_in_the_gate(
        self, script: ModuleType
    ) -> None:
        gate = {" ".join(command) for command in script.GATE}
        for command in workflow_commands():
            if any(part in command for part in SKIPPED):
                continue
            assert command in gate, command

    def test_the_gate_runs_nothing_the_workflow_does_not(
        self, script: ModuleType
    ) -> None:
        # The other direction. A gate that ran something CI does not would
        # fail a developer for a thing that cannot fail the branch.
        commands = set(workflow_commands())
        for command in script.GATE:
            assert " ".join(command) in commands, command

    def test_the_link_check_is_first(self, script: ModuleType) -> None:
        # The cheapest of them, and its message is one line. The file says
        # the order is chosen to put the short message first.
        assert script.GATE[0] == ["bash", "scripts/check-links.sh"]

    def test_the_standalone_job_is_left_out_and_said_to_be(
        self, script: ModuleType
    ) -> None:
        assert "standalone" in script.__doc__
        assert not any("venv" in " ".join(command) for command in script.GATE)
