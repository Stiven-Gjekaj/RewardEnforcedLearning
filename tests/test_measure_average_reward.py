"""The part of `scripts/measure_average_reward.py` that makes a claim.

`solved` says which loop the exactly optimal policy under a discount takes and
what that policy really collects. Everything the table asserts rests on those
two numbers agreeing with the environment's own crossover, so that is what is
held here.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rel.envs.continuing import TwoLoops
from rel.rng import Rng

SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "measure_average_reward.py"
)


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_average_reward", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestWhatTheModelSays:
    def test_below_the_crossover_it_takes_the_short_loop(
        self, script: ModuleType
    ) -> None:
        env = TwoLoops(Rng(1))
        loop, rate = script.solved(env, env.crossover() - 0.01)
        assert loop == "short"
        assert rate == pytest.approx(1.0)

    def test_above_it_the_long_one(self, script: ModuleType) -> None:
        env = TwoLoops(Rng(1))
        loop, rate = script.solved(env, env.crossover() + 0.01)
        assert loop == "long"
        assert rate == pytest.approx(2.0)

    def test_every_discount_in_the_table_is_on_one_side_or_the_other(
        self, script: ModuleType
    ) -> None:
        """No row of the printed table sits on the crossover itself, where
        which loop wins is decided by the last bit of a float."""
        env = TwoLoops(Rng(1))
        for discount in script.DISCOUNTS:
            assert abs(discount - env.crossover()) > 0.004

    def test_the_table_covers_both_sides(self, script: ModuleType) -> None:
        env = TwoLoops(Rng(1))
        answers = {script.solved(env, one)[0] for one in script.DISCOUNTS}
        assert answers == {"short", "long"}
