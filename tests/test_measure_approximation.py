"""The construction in `scripts/measure_approximation.py`.

The script compares two encoders on cost and on what they learn. The cost half
is the one that needed care. A whole run is the obvious thing to time and it is
the wrong thing: an agent that learns faster runs shorter episodes and finishes
sooner while being no quicker per step, so timing runs would report the policy
and call it the encoder.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_approximation.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_approximation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheCostIsPerStep:
    def test_it_reports_the_feature_count_the_agent_really_has(
        self, script: ModuleType
    ) -> None:
        _, features = script.microseconds_per_step("mountaincar", "tile-sarsa", 50)
        assert features == 8 * 9 * 9

    def test_the_tile_coder_has_far_more_features_and_costs_far_less(
        self, script: ModuleType
    ) -> None:
        """The finding the script exists to report, at the smallest size that
        still shows it. Reading only the feature counts gives the opposite
        answer, which is the reason both columns are printed."""
        tiles, tile_features = script.microseconds_per_step(
            "mountaincar", "tile-sarsa", 400
        )
        basis, basis_features = script.microseconds_per_step(
            "mountaincar", "rbf-sarsa", 400
        )
        assert tile_features > basis_features
        assert tiles < basis

    def test_the_step_count_asked_for_is_the_step_count_timed(
        self, script: ModuleType
    ) -> None:
        # Not a tautology. The loop resets the environment when an episode
        # ends, and an earlier version of that loop would have stopped there.
        few, _ = script.microseconds_per_step("mountaincar", "tile-sarsa", 50)
        many, _ = script.microseconds_per_step("mountaincar", "tile-sarsa", 400)
        assert few > 0.0 and many > 0.0

    def test_it_passes_the_settings_through_to_the_encoder(
        self, script: ModuleType
    ) -> None:
        _, features = script.microseconds_per_step(
            "mountaincar", "rbf-sarsa", 50, bins=4
        )
        assert features == 16


class TestTheBreakdown:
    def test_it_reports_every_piece(self, script: ModuleType) -> None:
        rows = script.breakdown(2, 4, 20)
        assert len(rows) == 5
        assert all(seconds > 0.0 for _, seconds in rows)

    def test_the_centre_count_is_in_the_first_line(self, script: ModuleType) -> None:
        assert "16 centres" in script.breakdown(2, 4, 20)[0][0]

    def test_the_distances_are_most_of_the_encoding(self, script: ModuleType) -> None:
        """The claim the section is written to support.

        If the exponentials or the normalising were the expensive part, then
        dropping features would be worth doing and `kept` would be the
        default. They are not, so it is not.
        """
        rows = dict(script.breakdown(4, 5, 30))
        distances = rows["the distance to all 625 centres"]
        whole = rows["and normalising them"]
        assert distances > 0.5 * whole


class TestOneRun:
    def test_the_same_seed_gives_the_same_number(self, script: ModuleType) -> None:
        first = script.one_run("mountaincar", "rbf-sarsa", 3, 4)
        again = script.one_run("mountaincar", "rbf-sarsa", 3, 4)
        assert first == again

    def test_a_different_seed_gives_a_different_number(
        self, script: ModuleType
    ) -> None:
        one = script.one_run("mountaincar", "rbf-sarsa", 3, 4)
        other = script.one_run("mountaincar", "rbf-sarsa", 4, 4)
        assert one != other

    def test_the_width_reaches_the_encoder(self, script: ModuleType) -> None:
        # The width sweep would print six identical rows if it did not, and
        # six identical rows read as a setting that does not matter rather
        # than as a setting that was never applied.
        narrow = script.one_run("mountaincar", "rbf-sarsa", 3, 6, width=0.5)
        wide = script.one_run("mountaincar", "rbf-sarsa", 3, 6, width=3.0)
        assert narrow != wide

    def test_the_environment_and_agent_seeds_are_not_the_same_number(
        self, script: ModuleType
    ) -> None:
        """Both come from the one seed the caller gives, and they must not be
        the same stream. Two agents on seed 4 have to face the same
        environment for the pairing to mean anything, and the offset is what
        keeps the agent's own draws away from the environment's."""
        source = SCRIPT.read_text()
        assert 'Rng(seed).stream("env")' in source
        assert 'Rng(500 + seed).stream("agent")' in source


class TestTheSections:
    def test_the_cost_table_has_a_row_for_each_pairing(
        self, script: ModuleType
    ) -> None:
        rows = script.cost_section(("mountaincar",), 50)
        assert [row[1] for row in rows] == [
            "tile-sarsa",
            "rbf-sarsa",
            "rbf-sarsa kept=8",
        ]

    def test_the_learning_table_names_both_agents_and_every_seed(
        self, script: ModuleType
    ) -> None:
        rows, verdict = script.learning_section(
            "mountaincar", 3, 4, Rng(1).stream("compare")
        )
        assert [row[0] for row in rows] == ["tile-sarsa", "rbf-sarsa"]
        assert all(len(row[2].split()) == 3 for row in rows)
        assert "95 percent interval" in verdict

    def test_the_warning_appears_exactly_when_the_interval_crosses_zero(
        self, script: ModuleType
    ) -> None:
        """Checked against the interval it printed, not against a guess.

        The first version of this assumed four episodes would settle nothing
        and asserted the warning was there. Four episodes settles a great
        deal: a radial basis generalises so much more widely at the start
        that the interval was nowhere near zero. So this reads the numbers
        the verdict states and asks whether the sentence matches them.
        """
        _, verdict = script.learning_section(
            "mountaincar", 3, 4, Rng(1).stream("compare")
        )
        inside = verdict.split("[")[1].split("]")[0]
        low, high = (float(part) for part in inside.split(", "))
        assert ("not told apart" in verdict) == (low < 0.0 < high)

    def test_the_width_table_has_a_row_for_each_width(self, script: ModuleType) -> None:
        rows = script.width_section("mountaincar", 2, 4)
        assert [row[0] for row in rows] == ["0.5", "0.75", "1", "1.5", "2", "3"]
