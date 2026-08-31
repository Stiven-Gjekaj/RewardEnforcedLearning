"""Tests for the agents that learn from one step.

Most of these state a table, hand one transition to an agent, and check the one
number that changed against arithmetic written in the test. That is slower to
write than driving a grid for a thousand episodes, and it is the only way to
tell a correct update rule from one that also happens to go up.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rel.agents.base import RandomAgent, Transition
from rel.agents.td import DoubleQ, ExpectedSarsa, NStepSarsa, QLearning, Sarsa
from rel.core import Env, EnvSpec, Step
from rel.envs.classic import cliff_walk
from rel.envs.gridworld import GridWorld
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import evaluate, train

TWO = Discrete(2)


def go(state: int, action: int, reward: float, landed: int) -> Transition[int, int]:
    return Transition(state, action, reward, landed, False, False)


def ends(state: int, action: int, reward: float, landed: int) -> Transition[int, int]:
    return Transition(state, action, reward, landed, True, False)


def cut(state: int, action: int, reward: float, landed: int) -> Transition[int, int]:
    return Transition(state, action, reward, landed, False, True)


class TestQLearning:
    def test_the_update_is_the_one_written_in_the_book(self) -> None:
        agent = QLearning(Rng(1), TWO, step_size=0.5, discount=0.9, epsilon=0.0)
        agent.q[0] = [1.0, 2.0]
        agent.q[1] = [5.0, 3.0]

        agent.observe(go(0, 0, -1.0, 1))

        # 1.0 + 0.5 * (-1 + 0.9 * max(5, 3) - 1.0) = 1.0 + 0.5 * 2.5
        assert agent.q[0][0] == pytest.approx(2.25)
        assert agent.q[0][1] == 2.0

    def test_it_takes_the_best_next_action_and_not_the_policy(self) -> None:
        # The whole of the off-policy difference is here. The behaviour policy
        # would sometimes take action 1, and the target never counts that.
        agent = QLearning(Rng(1), TWO, step_size=1.0, discount=1.0, epsilon=0.9)
        agent.q[0] = [0.0, 0.0]
        agent.q[1] = [10.0, -10.0]

        agent.observe(go(0, 0, 0.0, 1))
        assert agent.q[0][0] == 10.0

    def test_a_terminated_step_drops_the_future(self) -> None:
        agent = QLearning(Rng(1), TWO, step_size=1.0, discount=1.0)
        agent.q[1] = [99.0, 99.0]
        agent.observe(ends(0, 0, 5.0, 1))
        assert agent.q[0][0] == 5.0

    def test_a_truncated_step_keeps_the_future(self) -> None:
        # A step limit is not an ending. The state at the limit has a future,
        # and an agent that drops it teaches itself that a long episode ends
        # somewhere worthless.
        agent = QLearning(Rng(1), TWO, step_size=1.0, discount=1.0)
        agent.q[1] = [99.0, 99.0]
        agent.observe(cut(0, 0, 5.0, 1))
        assert agent.q[0][0] == 104.0


class TestSarsa:
    def test_the_update_waits_for_the_action_that_was_taken(self) -> None:
        # Nothing happens on the first transition, because the action that
        # decides the target has not been taken yet.
        agent = Sarsa(Rng(1), TWO, step_size=0.5, discount=0.9, epsilon=0.0)
        agent.q[0] = [1.0, 2.0]
        agent.q[1] = [5.0, 3.0]

        agent.observe(go(0, 0, -1.0, 1))
        assert agent.q[0][0] == 1.0

        agent.observe(go(1, 1, 0.0, 2))
        # 1.0 + 0.5 * (-1 + 0.9 * Q(1, 1) - 1.0), and Q(1, 1) is 3, not 5.
        assert agent.q[0][0] == pytest.approx(1.0 + 0.5 * (-1 + 2.7 - 1.0))

    def test_it_uses_the_action_taken_and_not_the_one_it_would_choose(self) -> None:
        # This is the regression test for a real fault. The first version of
        # this agent chose the next action itself and handed it back when `act`
        # was called. That is correct only while the loop takes the action it
        # was given, and it updated the wrong cell as soon as a caller took a
        # different one.
        agent = Sarsa(Rng(1), TWO, step_size=1.0, discount=1.0, epsilon=0.0)
        agent.q[0] = [0.0, 0.0]
        agent.q[1] = [100.0, -100.0]

        agent.observe(go(0, 0, 0.0, 1))
        agent.observe(go(1, 1, 0.0, 2))  # action 1, which greedy would not pick

        assert agent.q[0][0] == -100.0

    def test_a_terminated_step_updates_at_once(self) -> None:
        agent = Sarsa(Rng(1), TWO, step_size=1.0, discount=1.0)
        agent.q[1] = [99.0, 99.0]
        agent.observe(ends(0, 0, 5.0, 1))
        assert agent.q[0][0] == 5.0

    def test_a_truncated_step_bootstraps_from_the_policy(self) -> None:
        agent = Sarsa(Rng(1), TWO, step_size=1.0, discount=1.0, epsilon=0.0)
        agent.q[1] = [7.0, 3.0]
        agent.observe(cut(0, 0, 5.0, 1))
        assert agent.q[0][0] == 12.0

    def test_the_last_step_of_an_episode_is_not_left_unlearned(self) -> None:
        agent = Sarsa(Rng(1), TWO, step_size=1.0, discount=1.0)
        agent.observe(go(0, 0, -4.0, 1))
        agent.end_episode()
        assert agent.q[0][0] == -4.0


class TestExpectedSarsa:
    def test_the_target_is_the_average_over_the_policy(self) -> None:
        agent = ExpectedSarsa(Rng(1), TWO, step_size=1.0, discount=1.0, epsilon=0.2)
        agent.q[0] = [0.0, 0.0]
        agent.q[1] = [10.0, 0.0]

        agent.observe(go(0, 0, 0.0, 1))

        # Epsilon greedy over two actions with epsilon 0.2 gives 0.9 to the
        # best and 0.1 to the other: 0.9 * 10 + 0.1 * 0 = 9.
        assert agent.q[0][0] == pytest.approx(9.0)

    def test_with_no_exploration_it_matches_q_learning(self) -> None:
        # With epsilon at zero the average over the policy is the maximum, so
        # the two rules are the same rule.
        expected = ExpectedSarsa(Rng(1), TWO, step_size=0.5, discount=0.9, epsilon=0.0)
        greedy = QLearning(Rng(1), TWO, step_size=0.5, discount=0.9, epsilon=0.0)
        for agent in (expected, greedy):
            agent.q[0] = [1.0, 2.0]
            agent.q[1] = [5.0, 3.0]
            agent.observe(go(0, 0, -1.0, 1))
        assert expected.q[0] == greedy.q[0]

    def test_ties_share_the_greedy_weight(self) -> None:
        agent = ExpectedSarsa(Rng(1), TWO, step_size=1.0, discount=1.0, epsilon=0.0)
        agent.q[1] = [4.0, 4.0]
        agent.observe(go(0, 0, 0.0, 1))
        assert agent.q[0][0] == pytest.approx(4.0)


class TestDoubleQ:
    def test_only_one_table_moves_on_a_step(self) -> None:
        agent = DoubleQ(Rng(1), TWO, step_size=1.0, discount=1.0)
        agent.observe(go(0, 0, 5.0, 1))
        moved = [agent.values(0)[0] != 0.0, agent.other_values(0)[0] != 0.0]
        assert moved.count(True) == 1

    def test_the_value_comes_from_the_table_that_did_not_choose(self) -> None:
        # A table that both picks the best action and says what it is worth
        # grades its own choice, and the error never averages out.
        #
        # The numbers are chosen so that self grading is visible. One table
        # thinks action 0 is worth 100 and the other thinks action 1 is worth
        # 7. If either table ever valued its own choice, a cell here would come
        # out at 101 or at 8. Neither ever does.
        outcomes = set()
        for seed in range(20):
            agent = DoubleQ(Rng(seed), TWO, step_size=1.0, discount=1.0, epsilon=0.0)
            agent.q[1] = [100.0, 0.0]
            agent.q_other[1] = [5.0, 7.0]

            agent.observe(go(0, 0, 1.0, 1))
            outcomes.add((agent.values(0)[0], agent.other_values(0)[0]))

        # Chose with the first table: 1 + the other table's 5. Chose with the
        # second: 1 + the first table's 0.
        assert outcomes == {(6.0, 0.0), (0.0, 1.0)}

    def test_the_policy_ranks_by_the_sum_of_the_two_tables(self) -> None:
        agent = DoubleQ(Rng(1), TWO, epsilon=0.0)
        agent.q[5] = [10.0, 0.0]
        agent.q_other[5] = [0.0, 100.0]
        assert agent.greedy(5) == 1
        assert list(agent.action_scores(5)) == [10.0, 100.0]

    def test_it_leans_on_the_maximum_far_less_than_q_learning(self) -> None:
        # Sutton and Barto, figure 6.5. From the start state, going right ends
        # at once and pays nothing. Going left reaches a state with ten
        # actions, each paying a reward drawn around minus a tenth. Right is
        # the better move, and a maximum taken over ten noisy estimates is
        # above zero for a long time, so Q-learning goes left far more often
        # than it should.
        class Biased(Env[int]):
            def __init__(self, rng: Rng) -> None:
                super().__init__(rng)
                self.observation_space = Discrete(3)
                self.action_space = Discrete(10)
                self.spec = EnvSpec("biased", "The maximisation bias example.")
                self.at = 0
                self.went_left = 0

            def _reset(self) -> int:
                self.at = 0
                return 0

            def _step(self, action: int) -> Step[int]:
                if self.at == 0:
                    if action == 0:
                        self.went_left += 1
                        self.at = 1
                        return Step(1, 0.0, terminated=False, truncated=False)
                    return Step(2, 0.0, terminated=True, truncated=False)
                return Step(2, self.rng.normal(-0.1, 1.0), True, False)

        def share_going_left(builder: type[QLearning] | type[DoubleQ]) -> float:
            rng = Rng(3)
            env = Biased(rng.stream("env"))
            agent = builder(
                rng.stream("agent"), env.action_space, step_size=0.1, epsilon=0.1
            )
            train(env, agent, 300)
            return env.went_left / 300

        single = share_going_left(QLearning)
        double = share_going_left(DoubleQ)
        assert single > double + 0.15, f"single {single:.3f} double {double:.3f}"


class TestNStep:
    def test_one_step_is_sarsa(self) -> None:
        steps = [go(0, 0, -1.0, 1), go(1, 1, -1.0, 2), ends(2, 0, 5.0, 3)]

        def table(agent: Sarsa | NStepSarsa) -> dict[int, list[float]]:
            for _ in range(4):
                for transition in steps:
                    agent.observe(transition)
                agent.end_episode()
            return dict(agent.q)

        one = table(Sarsa(Rng(1), TWO, step_size=0.5, discount=0.9, epsilon=0.0))
        many = table(
            NStepSarsa(Rng(1), TWO, n=1, step_size=0.5, discount=0.9, epsilon=0.0)
        )
        assert one == many

    def test_the_return_carries_n_rewards(self) -> None:
        agent = NStepSarsa(Rng(1), TWO, n=3, step_size=1.0, discount=0.5, epsilon=0.0)
        for transition in (go(0, 0, -1.0, 1), go(1, 1, -1.0, 2), ends(2, 0, 5.0, 3)):
            agent.observe(transition)
        agent.end_episode()

        # -1 + 0.5 * -1 + 0.25 * 5, and no tail because the episode ended
        # inside the window.
        assert agent.q[0][0] == pytest.approx(-0.25)

    def test_it_uses_the_action_taken_and_not_the_one_it_would_choose(self) -> None:
        # The same fault as in SARSA, and the one that found it. An n step
        # agent that wrote down the action it expected updated the row for that
        # action rather than for the action the caller took.
        agent = NStepSarsa(Rng(1), TWO, n=2, step_size=1.0, discount=1.0, epsilon=0.0)
        agent.q[1] = [100.0, 0.0]

        for transition in (go(0, 0, 0.0, 1), go(1, 1, 0.0, 2), ends(2, 0, 0.0, 3)):
            agent.observe(transition)
        agent.end_episode()

        # The row for action 1 of state 1 is the one that moved, not action 0.
        assert agent.q[1][1] == 0.0
        assert agent.q[1][0] == 100.0

    def test_a_shorter_episode_than_n_still_learns(self) -> None:
        agent = NStepSarsa(Rng(1), TWO, n=8, step_size=1.0, discount=1.0)
        agent.observe(ends(0, 0, 3.0, 1))
        agent.end_episode()
        assert agent.q[0][0] == 3.0

    def test_n_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one or more"):
            NStepSarsa(Rng(1), TWO, n=0)

    def test_the_buffer_does_not_carry_over_between_episodes(self) -> None:
        agent = NStepSarsa(Rng(1), TWO, n=4, step_size=1.0, discount=1.0)
        agent.observe(go(0, 0, 1.0, 1))
        agent.end_episode()
        first = agent.q[0][0]

        agent.observe(ends(9, 1, 2.0, 10))
        agent.end_episode()
        assert agent.q[0][0] == first
        assert agent.q[9][1] == 2.0


class TestTieBreaking:
    def test_an_untouched_table_does_not_always_choose_the_first_action(self) -> None:
        # Every entry of a new table is equal, so every action ties. Taking the
        # first one turns the starting policy into "always go up", which is not
        # a neutral start on a grid: it explores one direction far more than
        # the others for as long as the tie lasts.
        agent = QLearning(Rng(4), Discrete(4), epsilon=0.0)
        assert len({agent.greedy(0) for _ in range(200)}) == 4

    def test_a_clear_best_is_always_chosen(self) -> None:
        agent = QLearning(Rng(4), Discrete(4), epsilon=0.0)
        agent.q[0] = [1.0, 9.0, 1.0, 1.0]
        assert {agent.greedy(0) for _ in range(50)} == {1}

    def test_only_the_tied_actions_are_drawn_between(self) -> None:
        agent = QLearning(Rng(4), Discrete(4), epsilon=0.0)
        agent.q[0] = [9.0, 9.0, 1.0, 1.0]
        assert {agent.greedy(0) for _ in range(200)} == {0, 1}


class TestLearningReallyHappens:
    """Every agent has to beat the policy that does not learn.

    An agent that learns nothing still returns a number. A curve that rises can
    rise because the environment got easier, or because episodes got shorter
    for a reason that has nothing to do with the policy. Comparing against the
    random policy on the same environment and the same seed is the only check
    that separates those.

    The number compared is the return while learning, not the return of the
    greedy policy. The greedy policy has a failure of its own, covered below,
    and it would put that failure into every one of these.
    """

    AGENTS: Sequence[type] = (QLearning, Sarsa, ExpectedSarsa, DoubleQ, NStepSarsa)

    @pytest.mark.parametrize("builder", AGENTS)
    def test_it_beats_the_random_policy(self, builder: type) -> None:
        seed = 5
        rng = Rng(seed)
        env = cliff_walk(rng.stream("env"))
        nothing = RandomAgent(rng.stream("agent"), env.action_space)
        baseline = train(env, nothing, 100).final()

        rng = Rng(seed)
        env = cliff_walk(rng.stream("env"))
        agent = builder(
            rng.stream("agent"), env.action_space, step_size=0.5, epsilon=0.1
        )
        learned = train(env, agent, 400).final(100)

        assert learned > baseline + 1000, (
            f"{builder.__name__} reached {learned:.1f} while learning, "
            f"and the random policy reached {baseline:.1f}"
        )

    @pytest.mark.parametrize("builder", AGENTS)
    def test_the_table_ends_up_worth_something(self, builder: type) -> None:
        # A separate question from the one above: not "did the returns rise"
        # but "does the table say anything true". The value of the start cell
        # under the greedy policy has to be near the length of a real path
        # rather than near the zero it began at.
        seed = 5
        rng = Rng(seed)
        env = cliff_walk(rng.stream("env"))
        agent = builder(
            rng.stream("agent"), env.action_space, step_size=0.5, epsilon=0.1
        )
        train(env, agent, 400)

        start = env.state_of(3, 0)
        values = agent.action_values(start)
        assert values is not None
        assert -60.0 < max(values) < -10.0, f"{builder.__name__}: {max(values)}"


class TestAGreedyPolicyCanDeadlock:
    """A greedy run can stop moving, and the step limit is what shows it.

    A cell where the best action does not change the cell is a trap for a
    policy with no exploration in it. The agent takes that action, arrives
    where it started, and takes it again. Nothing is wrong with the table and
    nothing raises. The episode runs to its step limit and the return is the
    limit.

    On the cliff walk this happens to the two on-policy agents and to neither
    of the others, and the reason is what each of them learned. SARSA learns
    the path along the top row, where a wasted step costs one out of about
    eighteen, so the four actions at a cell are within a point of each other
    and the noise in the estimates decides the order. Q-learning learns the
    path along the cliff edge, where a wrong action costs a hundred, and no
    amount of noise reorders that.

    Measured over thirty seeds, five hundred episodes, step size 0.5, epsilon
    0.1: SARSA deadlocks on 8, n step SARSA on 9, and Q-learning, Expected
    SARSA and Double Q on none. The command is in docs/algorithms.md.

    This is why `train` and `evaluate` are both reported and why a report says
    how many evaluation episodes were truncated. One number would hide it.
    """

    def test_a_self_loop_that_wins_by_a_hair_traps_the_greedy_policy(self) -> None:
        env = GridWorld(Rng(1), ["S.G"], max_episode_steps=50)
        agent = QLearning(Rng(1), env.action_space, epsilon=0.0)

        # Moving up runs into the edge and stays put. Here it is better than
        # moving right by one hundredth.
        agent.q[0] = [-10.00, -10.01, -10.02, -10.03]

        record = evaluate(env, agent, 1)
        assert record.returns == [-50.0]
        assert record.terminated == [False]

    def test_the_same_table_with_the_order_swapped_finishes(self) -> None:
        env = GridWorld(Rng(1), ["S.G"], max_episode_steps=50)
        agent = QLearning(Rng(1), env.action_space, epsilon=0.0)
        agent.q[0] = [-10.01, -10.00, -10.02, -10.03]
        agent.q[1] = [-10.01, -10.00, -10.02, -10.03]

        record = evaluate(env, agent, 1)
        assert record.returns == [-2.0]
        assert record.terminated == [True]

    def test_the_record_says_which_episodes_were_cut_off(self) -> None:
        # Without this the two runs above report -50 and -2 and nothing says
        # that the first one never finished. A reader takes -50 for a long
        # path rather than for a policy that stopped moving.
        env = GridWorld(Rng(1), ["S.G"], max_episode_steps=50)
        agent = QLearning(Rng(1), env.action_space, epsilon=0.0)
        agent.q[0] = [-10.00, -10.01, -10.02, -10.03]
        assert evaluate(env, agent, 3).terminated == [False, False, False]
