"""The construction in `scripts/measure_noise.py`, which is the whole of it.

The script measures how large a difference two identical agents show. That is
only worth anything if the two runs really are identical code differing only in
chance, and if each run gets its own chance rather than a whole comparison
sharing one draw. The first version shared it, and reported an interval
excluding zero on every comparison it made.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_noise.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_noise", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestOneRun:
    def test_the_same_two_seeds_give_the_same_number(self, script: ModuleType) -> None:
        first = script.one_run("cliff", "q-learning", 3, 99, 20)
        again = script.one_run("cliff", "q-learning", 3, 99, 20)
        assert first == again

    def test_a_different_agent_seed_gives_a_different_number(
        self, script: ModuleType
    ) -> None:
        """Which is the noise the script exists to measure. Without this the
        whole table would be zeros and look like a very tight result."""
        one = script.one_run("cliff", "q-learning", 3, 99, 20)
        other = script.one_run("cliff", "q-learning", 3, 100, 20)
        assert one != other

    def test_on_a_deterministic_grid_the_environment_seed_does_nothing(
        self, script: ModuleType
    ) -> None:
        """Worth knowing, and it surprised this test into existence.

        The cliff walk has no chance in it. Every seed builds the identical
        grid, so a run of it varies only by the agent's own draws, and the
        pairing this script does by environment seed pairs nothing there. It
        is still the right construction, because what it holds fixed is
        everything the two sides could otherwise have differed by.
        """
        here = script.one_run("cliff", "q-learning", 3, 99, 20)
        there = script.one_run("cliff", "q-learning", 4, 99, 20)
        assert here == there

    def test_on_a_grid_that_slips_it_does_something(self, script: ModuleType) -> None:
        """The frozen lake moves the agent sideways two times in three, so
        there the environment seed is a real second source of variation and
        the pairing is doing work."""
        here = script.one_run("lake", "q-learning", 3, 99, 200)
        there = script.one_run("lake", "q-learning", 5, 99, 200)
        assert here != there


class TestEachRunGetsItsOwnChance:
    """The fault the first version had, kept as a test.

    One agent seed per comparison makes the paired differences lean together,
    which answers "is this agent seed better than that one". The seeds a trial
    hands out have to differ across the runs inside it as well as across the
    trials.
    """

    def test_the_seeds_inside_one_trial_are_all_different(
        self, script: ModuleType
    ) -> None:
        runs = 5
        trial = 3
        mine = [10_000 + trial * runs + seed for seed in range(1, runs + 1)]
        yours = [500_000 + trial * runs + seed for seed in range(1, runs + 1)]
        assert len(set(mine)) == runs
        assert len(set(yours)) == runs

    def test_no_seed_is_used_by_both_sides_or_by_two_trials(
        self, script: ModuleType
    ) -> None:
        runs = 5
        seen: set[int] = set()
        for trial in range(200):
            for seed in range(1, runs + 1):
                seen.add(10_000 + trial * runs + seed)
                seen.add(500_000 + trial * runs + seed)
        assert len(seen) == 200 * runs * 2


class TestOneTrial:
    def test_it_answers_a_gap_and_two_verdicts(self, script: ModuleType) -> None:
        gap, certain, significant = script.one_trial(
            "cliff", "q-learning", 3, 20, 0, Rng(1)
        )
        assert gap >= 0.0
        assert isinstance(certain, bool)
        assert isinstance(significant, bool)

    def test_three_seeds_can_never_be_significant(self, script: ModuleType) -> None:
        """The floor at three seeds is 0.25, so the last verdict is fixed."""
        _, _, significant = script.one_trial("cliff", "q-learning", 3, 20, 0, Rng(1))
        assert not significant
