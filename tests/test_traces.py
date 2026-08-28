"""Tests for eligibility traces.

The claim a trace makes is that it is one dial between one step learning and
crediting the whole episode. The sharpest test of that is the end of the dial
that can be checked exactly: at zero the agent has to be the one step agent it
was built on, cell for cell, and not merely close to it.
"""

from __future__ import annotations

import pytest

from rel.agents.base import Transition
from rel.agents.dp import evaluate_policy
from rel.agents.td import Sarsa
from rel.agents.traces import CUTOFF, KINDS, SarsaLambda, WatkinsQLambda
from rel.envs.classic import cliff_walk
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train

WALK = (
    Transition(0, 0, -1.0, 1, terminated=False, truncated=False),
    Transition(1, 1, -1.0, 2, terminated=False, truncated=False),
    Transition(2, 0, -1.0, 3, terminated=False, truncated=False),
    Transition(3, 1, 10.0, 4, terminated=True, truncated=False),
)


def feed(agent, steps=WALK) -> None:  # type: ignore[no-untyped-def]
    agent.start_episode()
    for step in steps:
        agent.observe(step)
    agent.end_episode()


class TestTheDialAtZero:
    def test_no_decay_is_the_one_step_agent_cell_for_cell(self) -> None:
        # This is the whole claim in one assertion. A trace method that did
        # not collapse to its one step form would be a different algorithm
        # wearing the name.
        plain: Sarsa[int] = Sarsa(
            Rng(1).stream("a"), Discrete(2), step_size=0.5, discount=0.9, epsilon=0.0
        )
        traced: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"),
            Discrete(2),
            step_size=0.5,
            discount=0.9,
            epsilon=0.0,
            trace_decay=0.0,
        )
        feed(plain)
        feed(traced)
        assert traced.q == plain.q

    def test_a_decay_above_zero_reaches_further_back(self) -> None:
        # The first cell of the walk is three steps from the reward. A one step
        # agent cannot have moved it at all after one episode.
        short: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"),
            Discrete(2),
            step_size=0.5,
            discount=0.9,
            epsilon=0.0,
            trace_decay=0.0,
        )
        long: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"),
            Discrete(2),
            step_size=0.5,
            discount=0.9,
            epsilon=0.0,
            trace_decay=1.0,
        )
        feed(short)
        feed(long)
        assert long.q[0][0] != short.q[0][0]


class TestTheTraces:
    def test_a_trace_decays_by_the_discount_times_the_dial(self) -> None:
        agent: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"),
            Discrete(2),
            discount=0.9,
            trace_decay=0.5,
            traces="replacing",
        )
        agent.bump(7, 1)
        assert agent.e[(7, 1)] == 1.0
        agent.fade()
        assert agent.e[(7, 1)] == pytest.approx(0.45)

    def test_a_trace_that_says_nothing_is_dropped(self) -> None:
        agent: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"), Discrete(2), discount=1.0, trace_decay=0.5
        )
        agent.bump(7, 1)
        for _ in range(40):
            agent.fade()
        # Half, forty times over, is far below the cutoff.
        assert (7, 1) not in agent.e
        assert agent.e == {}

    def test_the_traces_do_not_reach_across_an_ending(self) -> None:
        agent: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"), Discrete(2), trace_decay=0.9
        )
        agent.bump(7, 1)
        assert agent.e
        agent.start_episode()
        assert agent.e == {}

    def test_accumulating_adds_and_can_pass_one(self) -> None:
        agent: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"), Discrete(2), traces="accumulating"
        )
        agent.bump(7, 1)
        agent.bump(7, 1)
        assert agent.e[(7, 1)] == 2.0

    def test_replacing_sets_and_cannot_pass_one(self) -> None:
        agent: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"), Discrete(2), traces="replacing"
        )
        agent.bump(7, 1)
        agent.bump(7, 1)
        assert agent.e[(7, 1)] == 1.0

    def test_dutch_sits_between_the_other_two(self) -> None:
        agent: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"), Discrete(2), step_size=0.25, traces="dutch"
        )
        agent.bump(7, 1)
        agent.bump(7, 1)
        assert 1.0 < agent.e[(7, 1)] < 2.0
        assert agent.e[(7, 1)] == pytest.approx(1.75)

    def test_the_three_kinds_are_the_same_on_a_cell_seen_once(self) -> None:
        raised = []
        for kind in KINDS:
            agent: SarsaLambda[int] = SarsaLambda(
                Rng(1).stream("a"), Discrete(2), step_size=0.25, traces=kind
            )
            agent.bump(7, 1)
            raised.append(agent.e[(7, 1)])
        assert raised == [1.0, 1.0, 1.0]

    def test_a_dial_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="trace_decay"):
            SarsaLambda(Rng(1).stream("a"), Discrete(2), trace_decay=1.5)

    def test_a_kind_that_does_not_exist_is_refused(self) -> None:
        with pytest.raises(ValueError, match="traces is one of"):
            SarsaLambda(Rng(1).stream("a"), Discrete(2), traces="sticky")  # type: ignore[arg-type]


class TestWatkins:
    def test_it_bootstraps_from_the_best_action_and_not_the_taken_one(self) -> None:
        # The two differ only in the target, so a walk where the taken next
        # action is the worse one separates them.
        sarsa: SarsaLambda[int] = SarsaLambda(
            Rng(1).stream("a"),
            Discrete(2),
            step_size=0.5,
            discount=0.9,
            epsilon=0.0,
            trace_decay=0.0,
        )
        watkins: WatkinsQLambda[int] = WatkinsQLambda(
            Rng(1).stream("a"),
            Discrete(2),
            step_size=0.5,
            discount=0.9,
            epsilon=0.0,
            trace_decay=0.0,
        )
        for agent in (sarsa, watkins):
            agent.values(2)[0] = -5.0
            agent.values(2)[1] = 5.0
            agent.start_episode()
            agent.observe(Transition(0, 0, 0.0, 2, terminated=False, truncated=False))
            agent.observe(Transition(2, 0, 0.0, 3, terminated=True, truncated=False))
        assert watkins.q[0][0] != sarsa.q[0][0]

    def test_an_exploratory_action_cuts_every_trace(self) -> None:
        agent: WatkinsQLambda[int] = WatkinsQLambda(
            Rng(1).stream("a"), Discrete(2), epsilon=0.0, trace_decay=0.9
        )
        # Action 1 is the greedy one at state 2, so taking action 0 there is
        # the agent leaving its own policy.
        agent.values(2)[0] = -5.0
        agent.values(2)[1] = 5.0
        agent.start_episode()
        agent.observe(Transition(0, 0, 0.0, 2, terminated=False, truncated=False))
        agent.observe(Transition(2, 0, 0.0, 3, terminated=False, truncated=False))
        assert agent.e == {}

    def test_a_greedy_action_leaves_the_traces_alone(self) -> None:
        agent: WatkinsQLambda[int] = WatkinsQLambda(
            Rng(1).stream("a"), Discrete(2), epsilon=0.0, trace_decay=0.9
        )
        agent.values(2)[0] = -5.0
        agent.values(2)[1] = 5.0
        agent.start_episode()
        agent.observe(Transition(0, 0, 0.0, 2, terminated=False, truncated=False))
        agent.observe(Transition(2, 1, 0.0, 3, terminated=False, truncated=False))
        assert agent.e


class TestTheyLearn:
    @pytest.mark.parametrize("cls", [SarsaLambda, WatkinsQLambda])
    def test_it_reaches_a_good_cliff_walk_policy(self, cls: type) -> None:
        rng = Rng(3)
        env = cliff_walk(rng.stream("env"))
        agent = cls(
            rng.stream("agent"),
            env.action_space,
            step_size=0.2,
            epsilon=0.1,
            trace_decay=0.8,
        )
        train(env, agent, 400, discount=1.0)

        policy = [agent.greedy(state) for state in range(env.observation_space.n)]
        report = evaluate_policy(env, policy, discount=1.0)
        assert report.reaches_end
        assert report.start_value >= -20.0

    def test_the_trace_table_does_not_grow_without_bound(self) -> None:
        # Four hundred episodes of a grid with forty eight cells. Without the
        # cutoff the table holds every cell and action ever touched, for ever.
        rng = Rng(3)
        env = cliff_walk(rng.stream("env"))
        agent: SarsaLambda[int] = SarsaLambda(
            rng.stream("agent"), env.action_space, trace_decay=0.9, discount=1.0
        )
        train(env, agent, 50, discount=1.0)
        assert len(agent.e) <= env.observation_space.n * env.action_space.n
        assert all(value >= CUTOFF for value in agent.e.values())
