"""The measurement behind the sum tree tables on the algorithms page.

The load bearing test is `TestTheBoundarySearch`. Every number in the first
table is the distance from a boundary the tree draws at to the running total
the scan compares against, and the boundary is found by halving a range of
bit patterns rather than by sampling. A search that stopped one place early
would still print a table, the shares would still be tiny, and nothing would
say they were tiny for the wrong reason.

`TestTheBitPatterns` is under it, because the search is only a search if a
double and its pattern run in the same order.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.agents.sums import Sums
from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_tree.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_tree", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestTheBitPatterns:
    def test_a_double_and_its_pattern_go_back_and_forth(
        self, script: ModuleType
    ) -> None:
        for value in (0.0, 1e-300, 1e-6, 1.0, 3.25, 1e300):
            assert script.as_float(script.as_bits(value)) == value

    def test_patterns_run_in_the_same_order_as_the_doubles(
        self, script: ModuleType
    ) -> None:
        rng = Rng(3)
        values = sorted(rng.uniform(0.0, 1e6) for _ in range(200))
        patterns = [script.as_bits(value) for value in values]
        assert patterns == sorted(patterns)

    def test_the_next_pattern_is_the_next_double(self, script: ModuleType) -> None:
        for value in (1e-6, 1.0, 12345.678):
            stepped = script.as_float(script.as_bits(value) + 1)
            assert stepped == math.nextafter(value, math.inf)


class TestTheBoundarySearch:
    def test_it_lands_on_the_first_target_the_tree_sends_that_far(
        self, script: ModuleType
    ) -> None:
        weights = [1.0, 2.0, 3.0, 4.0]
        tree = script.tree_of(weights)
        total = sum(weights)
        for place in range(1, len(weights)):
            found = script.first_target_above(tree, place, total)
            assert tree.find(found) >= place
            assert tree.find(math.nextafter(found, 0.0)) < place

    def test_it_finds_the_exact_boundary_on_whole_numbers(
        self, script: ModuleType
    ) -> None:
        # These weights add exactly, so the tree and the scan agree and the
        # boundary is the running total itself rather than a neighbour of it.
        weights = [1.0, 2.0, 4.0, 8.0]
        tree = script.tree_of(weights)
        running = list(itertools.accumulate(weights))
        for place in range(1, len(weights)):
            assert (
                script.first_target_above(tree, place, running[-1])
                == (running[place - 1])
            )

    def test_it_finds_a_boundary_for_every_place(self, script: ModuleType) -> None:
        rng = Rng(4)
        weights = [rng.uniform(0.5, 2.0) for _ in range(64)]
        tree = script.tree_of(weights)
        total = sum(weights)
        found = [
            script.first_target_above(tree, place, total) for place in range(1, 64)
        ]
        assert found == sorted(found)


class TestWhatItCounts:
    def test_the_share_is_never_negative_and_never_a_whole_draw(
        self, script: ModuleType
    ) -> None:
        share, places, widest, narrowest = script.disagreement(256, 5)
        assert 0.0 <= share < 1e-6
        assert places >= 0
        assert widest >= 0.0
        assert narrowest > 0.0

    def test_the_widest_gap_is_far_below_the_narrowest_place(
        self, script: ModuleType
    ) -> None:
        # This is what makes a disagreement a neighbour rather than a jump.
        _, _, widest, narrowest = script.disagreement(256, 5)
        assert narrowest > widest * 1e6

    def test_weights_of_the_same_seed_are_the_same_weights(
        self, script: ModuleType
    ) -> None:
        assert script.weights_of(16, 2) == script.weights_of(16, 2)
        assert script.weights_of(16, 2) != script.weights_of(16, 3)

    def test_the_weights_it_loads_are_the_weights_the_tree_holds(
        self, script: ModuleType
    ) -> None:
        weights = script.weights_of(20, 7)
        tree = script.tree_of(weights)
        assert isinstance(tree, Sums)
        assert [tree[place] for place in range(20)] == weights


class TestTheCostReading:
    def test_a_reading_is_a_positive_number_of_microseconds(
        self, script: ModuleType
    ) -> None:
        assert script.one_cost(64, False, 20, 4) > 0.0
        assert script.one_cost(64, True, 20, 4) > 0.0

    def test_both_sides_time_the_same_work(self, script: ModuleType) -> None:
        # The reading would mean nothing if one side drew a batch and the
        # other drew a batch and put its errors back.
        source = SCRIPT.read_text()
        body = source[source.index("def one_cost(") : source.index("def cost_section(")]
        assert body.count("buffer.sample(batch)") == 1
        assert body.count("buffer.reprioritise(") == 1
        assert body.count("if tree") == 0


class TestItRuns:
    def test_the_short_run_prints_all_three_sections(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.agreement_section((64,), 5)
        script.cost_section((64,), 4, 400)
        script.agent_section(3, 1, "cartpole", (64,))
        printed = capsys.readouterr().out
        assert "part company" in printed
        assert "What one update costs" in printed
        assert "inside an agent" in printed
        assert "same digests" in printed

    def test_the_agent_learns_the_same_thing_either_way(
        self, script: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script.agent_section(5, 1, "cartpole", (64,))
        assert "NO" not in capsys.readouterr().out
