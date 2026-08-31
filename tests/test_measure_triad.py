"""The measurement behind the deadly triad tables on the algorithms page.

The load bearing test is `TestTheCrossingAgreesWithTheArithmetic`. The script
finds the discount where the counterexample starts to run away by running the
update, and `rel.envs.baird` works the same number out in closed form. Neither
reads the other, so the two agreeing is two answers rather than one printed
twice, and a test checks that neither reads the other.

The script is loaded by its path, because `scripts/` is not a package and
importing it as one would be a different arrangement than the one that runs.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rel.agents.base import Transition
from rel.agents.linear_prediction import SemiGradientTD, fixed
from rel.agents.lookup import Lookup
from rel.envs.baird import SOLID, STARTING_WEIGHTS, Baird
from rel.rng import Rng

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_triad.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_triad", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def an_environment(upper: int = 6) -> Baird:
    return Baird(Rng(1).stream("env"), upper=upper)


def body_of(name: str) -> str:
    """One function of the script, as text, found by name rather than by slice.

    The first version of this cut the source between one `def` and the next,
    which held until a function moved and then quietly compared an empty
    string against everything it was asked about.
    """
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} is not in the script")


class HalfAndHalf(Baird):
    """The counterexample under a behaviour policy that dashes half the time.

    It is not the counterexample any more, because the closed form for the
    crossing assumes an even spread over the states. It is here so that
    `visits` has something to work out: at these shares the agent spends half
    its time in the lower state and a twelfth in each of the others.
    """

    @property
    def behaviour_shares(self) -> tuple[float, ...]:
        return (0.5, 0.5)


class TestWhereTheAgentIs:
    def test_the_behaviour_policy_is_everywhere_equally(
        self, script: ModuleType
    ) -> None:
        for upper in (3, 6, 20):
            spread = script.visits(an_environment(upper))
            assert len(spread) == upper + 1
            assert max(spread) - min(spread) < 1e-12, upper
            assert sum(spread) == pytest.approx(1.0)

    def test_a_spread_that_is_not_even_comes_out_uneven(
        self, script: ModuleType
    ) -> None:
        # The even answer is also the guess it starts from, so on this
        # environment a version that never iterated at all would agree with
        # one that does. This is an environment where they cannot.
        spread = script.visits(HalfAndHalf(Rng(1).stream("env")))
        assert sum(spread) == pytest.approx(1.0)
        assert spread[6] == pytest.approx(0.5)
        for state in range(6):
            assert spread[state] == pytest.approx(0.5 / 6), state

    def test_it_is_worked_out_rather_than_written_down(
        self, script: ModuleType
    ) -> None:
        # A hard coded even spread would be right on this environment and
        # would stop being right the moment anything about it moved.
        body = body_of("visits")
        assert "transitions" in body
        assert "behaviour_shares" in body


class TestReadingTheScriptItself:
    def test_the_helper_finds_a_function_that_is_there(self) -> None:
        assert "def visits" in body_of("visits")

    def test_it_says_so_rather_than_returning_nothing(self) -> None:
        # The fault the first version of these tests had: a slice that found
        # nothing compared an empty string against everything.
        with pytest.raises(AssertionError, match="is not in the script"):
            body_of("no_such_function")


class TestTheStartItRunsFrom:
    def test_six_upper_states_give_the_vector_in_the_environment(
        self, script: ModuleType
    ) -> None:
        assert script.starting_weights(an_environment()) == list(STARTING_WEIGHTS)

    def test_the_ten_lands_on_the_lower_states_own_weight(
        self, script: ModuleType
    ) -> None:
        weights = script.starting_weights(an_environment(upper=4))
        assert weights == [1.0, 1.0, 1.0, 1.0, 10.0, 1.0]


class TestTheExpectedUpdate:
    def test_weights_of_zero_have_nothing_to_change(self, script: ModuleType) -> None:
        # Nothing pays anything, so a value of zero everywhere has no error
        # anywhere, and the answer the agent is looking for is already in it.
        env = an_environment()
        coder = Lookup(env.feature_rows)
        change = script.expected_change(
            env, coder, script.visits(env), [0.0] * coder.features, 0.99
        )
        assert change == [0.0] * coder.features

    def test_it_is_the_agents_own_update_averaged_over_the_steps(
        self, script: ModuleType
    ) -> None:
        """The strongest thing that can be said about the expected update.

        Put all the weight of the state distribution on one state, and what
        comes back is the step the agent takes there, times how often the
        behaviour policy takes the action that leads to it. Anything the
        expected update forgets, from the ratio to the divisor the step size
        is shared out by, moves one side of this and not the other.
        """
        env = an_environment()
        coder = Lookup(env.feature_rows)
        before = script.starting_weights(env)
        step = 0.05

        agent = SemiGradientTD(
            Rng(1).stream("agent"),
            env.action_space,
            coder,
            fixed(env.behaviour_shares),
            fixed(env.target_shares),
            step_size=step,
            discount=0.99,
        )
        agent.weights[:] = list(before)
        agent.observe(Transition(0, SOLID, 0.0, env.lower, False, False))
        moved = [after - was for after, was in zip(agent.weights, before, strict=True)]

        here = [0.0] * env.observation_space.n
        here[0] = 1.0
        change = script.expected_change(env, coder, here, list(before), 0.99)

        share = env.behaviour_shares[SOLID]
        assert [step * one for one in change] == pytest.approx(
            [share * one for one in moved]
        )

    def test_it_settles_below_the_crossing(self, script: ModuleType) -> None:
        settled = script.expected_size(an_environment(), 0.8, script.EXPECTED_STEP, 400)
        assert settled < 10.0

    def test_it_runs_away_above_the_crossing(self, script: ModuleType) -> None:
        assert (
            script.expected_size(an_environment(), 0.99, script.EXPECTED_STEP, 400)
            > 1e3
        )

    def test_the_growth_rate_says_which_side_it_is_on(self, script: ModuleType) -> None:
        env = an_environment()
        below = script.growth_rate(env, 0.8, script.EXPECTED_STEP, 600)
        above = script.growth_rate(env, 0.99, script.EXPECTED_STEP, 600)
        assert below <= script.RUNNING_AWAY
        assert above > script.RUNNING_AWAY

    def test_the_rate_is_read_over_the_second_half_only(
        self, script: ModuleType
    ) -> None:
        """What is measured is the rate it settles into, not the start.

        The weights begin with a ten on one of them, which is a transient of
        its own, and a rate averaged over the whole run would carry it.
        """
        assert "steps // 2" in body_of("growth_rate")


class TestTheRateWhenThereIsLittleToRead:
    def test_a_single_step_is_still_a_rate(self, script: ModuleType) -> None:
        # One step halves to none, and a rate divided by no steps at all
        # would be a crash rather than an answer.
        rate = script.growth_rate(an_environment(), 0.99, script.EXPECTED_STEP, 1)
        assert rate == pytest.approx(rate)

    def test_weights_that_are_all_zero_are_the_opposite_of_running_away(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vector of zeros cannot be scaled back to a largest of one.

        Nothing reaches it from the start this script uses, and the rate is
        read by scaling the weights after every step, so the one input that
        cannot be scaled is worth an answer rather than a division by zero.
        """
        monkeypatch.setattr(
            script, "starting_weights", lambda env: [0.0] * len(env.feature_rows[0])
        )
        rate = script.growth_rate(an_environment(), 0.99, script.EXPECTED_STEP, 4)
        assert rate == float("-inf")


class TestTheCrossingAgreesWithTheArithmetic:
    @pytest.mark.parametrize("upper", [5, 6, 10])
    def test_the_measured_crossing_matches_the_closed_form(
        self, script: ModuleType, upper: int
    ) -> None:
        measured = script.crossing(upper, script.EXPECTED_STEP, script.RATE_STEPS)
        assert measured == pytest.approx(
            an_environment(upper).runs_away_above(), abs=0.01
        )

    def test_four_upper_states_never_run_away(self, script: ModuleType) -> None:
        measured = script.crossing(4, script.EXPECTED_STEP, script.RATE_STEPS)
        assert measured != measured

    def test_nothing_that_measures_reads_the_closed_form(
        self, script: ModuleType
    ) -> None:
        """Two answers rather than one printed twice.

        `runs_away_above` is the arithmetic. Every function that works the
        crossing out by running the update has to be free of it, or the table
        would be a number checked against itself.
        """
        tree = ast.parse(SCRIPT.read_text())
        measuring = {"visits", "expected_change", "expected_size", "growth_rate"}
        measuring |= {"runs_away", "crossing", "starting_weights"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in measuring:
                names = {
                    inner.attr
                    for inner in ast.walk(node)
                    if isinstance(inner, ast.Attribute)
                }
                assert "runs_away_above" not in names, node.name


class TestTheTwoAgents:
    def test_the_semi_gradient_one_runs_away(self, script: ModuleType) -> None:
        assert script.one_run("linear-td", 1, episodes=5) > 1e6

    def test_the_corrected_one_does_not(self, script: ModuleType) -> None:
        assert script.one_run("gradient-td", 1, episodes=5) < 10.0

    def test_below_the_crossing_neither_does(self, script: ModuleType) -> None:
        assert script.one_run("linear-td", 1, discount=0.8, episodes=5) < 10.0

    def test_an_agent_it_does_not_know_is_refused(self, script: ModuleType) -> None:
        with pytest.raises(KeyError):
            script.one_run("q-learning", 1, episodes=1)


class TestTheReport:
    @staticmethod
    def _run(
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        *extra: str,
    ) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "measure_triad",
                "--runs",
                "1",
                "--episodes",
                "2",
                "--expected-steps",
                "200",
                "--rate-steps",
                "200",
                *extra,
            ],
        )
        return int(script.main())

    def test_every_section_is_printed(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert self._run(script, monkeypatch, "--sizes", "6") == 0
        printed = capsys.readouterr().out
        assert "linear-td" in printed
        assert "gradient-td" in printed
        assert "It is not the step size" in printed
        assert "The crossing" in printed
        assert "closed form" in printed

    def test_the_first_table_runs_both_sides_of_the_crossing(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The point of the second pair of rows is that nothing but the
        # discount is different, so a report with only the first pair would
        # leave the reader to take that on trust.
        self._run(script, monkeypatch, "--sizes", "6")
        first = capsys.readouterr().out.split("It is not the step size")[0]
        assert first.count("linear-td") == 2
        assert first.count("gradient-td") == 2
        assert "0.99" in first
        assert "0.5" in first

    def test_the_discounts_it_was_given_are_the_ones_it_runs(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--sizes", "6", "--discounts", "0.7")
        first = capsys.readouterr().out.split("It is not the step size")[0]
        assert first.count("linear-td") == 1
        assert "0.7" in first

    def test_the_step_sizes_it_was_given_are_the_ones_it_sweeps(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--sizes", "6", "--step-sizes", "0.03", "0.07")
        printed = capsys.readouterr().out
        assert "0.03" in printed
        assert "0.07" in printed
        assert "0.005" not in printed

    def test_the_sizes_it_was_given_are_the_ones_it_measures(
        self,
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(script, monkeypatch, "--sizes", "5", "6")
        table = capsys.readouterr().out.split("The size of the problem")[1]
        assert "0.9333" in table
        assert "0.8824" in table
        assert "0.7600" not in table
