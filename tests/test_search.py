"""Tests for planning at the moment of choosing.

Two of these matter more than the rest. A tiny model with one right answer says
the search finds it, which nothing about a grid result can say on its own. And
`greedy` has to leave no trace, because a renderer asks it for every cell of a
grid and an evaluation asks it for every state: a search that grew the agent's
tree or spent the agent's chance would change the run it is being asked about.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rel.agents.search import Node, TreeSearch
from rel.core import EnvSpec, Outcome, Step, TabularEnv
from rel.envs.classic import cliff_walk, dyna_maze
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import train


class Fork(TabularEnv):
    """One choice, made once, with an obvious right answer.

    From the start, action 1 ends the episode paying 1 and action 0 ends it
    paying nothing. Nothing about a rollout, a discount or a depth changes
    which of those is better, so a search that cannot find this one is broken
    rather than short of budget.
    """

    def __init__(self, rng: Rng) -> None:
        super().__init__(rng)
        self.observation_space = Discrete(3)
        self.action_space = Discrete(2)
        self.spec = EnvSpec("fork", "One choice with an obvious answer.")
        self.at = 0

    def _reset(self) -> int:
        self.at = 0
        return 0

    def _step(self, action: int) -> Step[int]:
        branch = self.transitions(self.at, action)[0]
        self.at = branch.observation
        return Step(branch.observation, branch.reward, branch.terminated, False)

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if state != 0:
            return [Outcome(1.0, state, 0.0, terminated=True)]
        if action == 1:
            return [Outcome(1.0, 2, 1.0, terminated=True)]
        return [Outcome(1.0, 1, 0.0, terminated=True)]

    def start_states(self) -> Sequence[tuple[float, int]]:
        return [(1.0, 0)]


def searcher(rng: Rng, model: TabularEnv, **extra: object) -> TreeSearch:
    settings: dict[str, object] = {"simulations": 20, "depth": 10, **extra}
    return TreeSearch(rng, model.action_space, model, **settings)  # type: ignore[arg-type]


class TestANode:
    def test_it_starts_at_nothing(self) -> None:
        node = Node(3)
        assert node.counts == [0, 0, 0]
        assert node.visits == 0

    def test_an_untried_action_reads_as_zero(self) -> None:
        """Nothing acts on it. The descent takes every untried action before it
        ranks anything, so a mean is only read where the count is above zero."""
        assert Node(2).means() == [0.0, 0.0]

    def test_crediting_moves_the_count_and_the_mean(self) -> None:
        node = Node(2)
        node.credit(0, 4.0)
        node.credit(0, 6.0)
        assert node.counts == [2, 0]
        assert node.means()[0] == pytest.approx(5.0)

    def test_visits_add_up_over_the_actions(self) -> None:
        node = Node(3)
        node.credit(0, 1.0)
        node.credit(2, 1.0)
        node.credit(2, 1.0)
        assert node.visits == 3

    def test_it_says_what_it_is(self) -> None:
        node = Node(2)
        node.credit(1, 0.0)
        assert "1" in repr(node)


class TestTheSettingsAreChecked:
    def test_a_search_runs_at_least_one_simulation(self) -> None:
        model = Fork(Rng(1))
        with pytest.raises(ValueError, match="at least one simulation"):
            TreeSearch(Rng(1), model.action_space, model, simulations=0)

    def test_a_simulation_runs_at_least_one_step(self) -> None:
        model = Fork(Rng(1))
        with pytest.raises(ValueError, match="at least one step"):
            TreeSearch(Rng(1), model.action_space, model, depth=0)

    def test_the_discount_is_between_zero_and_one(self) -> None:
        model = Fork(Rng(1))
        with pytest.raises(ValueError, match="between 0 and 1"):
            TreeSearch(Rng(1), model.action_space, model, discount=1.5)


class TestItFindsTheRightAnswer:
    def test_the_paying_action_is_the_one_it_takes(self) -> None:
        model = Fork(Rng(1))
        agent = searcher(Rng(2), model)
        assert agent.act(0) == 1

    def test_it_finds_it_on_every_seed(self) -> None:
        model = Fork(Rng(1))
        for seed in range(1, 21):
            assert searcher(Rng(seed), model).act(0) == 1

    def test_the_paying_action_is_the_one_with_the_visits(self) -> None:
        model = Fork(Rng(2))
        agent = searcher(Rng(2), model)
        agent.act(0)
        node = agent.tree[0]
        assert node.counts[1] > node.counts[0]

    def test_what_it_believes_matches_the_model(self) -> None:
        model = Fork(Rng(2))
        agent = searcher(Rng(2), model)
        agent.act(0)
        values = agent.action_values(0)
        assert values is not None
        assert values[1] == pytest.approx(1.0)
        assert values[0] == pytest.approx(0.0)


class TestReadingThePolicyLeavesNoTrace:
    """The fault this project has already had once, in another place.

    Reading the greedy policy of a Q-learning agent spent its generator on tie
    breaks, and the episode it first reached the optimum on moved from 68 to
    258. A search is that fault waiting to happen twice over: it would spend
    the generator and it would grow the tree.
    """

    def test_it_does_not_grow_the_tree(self) -> None:
        model = dyna_maze(Rng(1))
        agent = searcher(Rng(3), model, discount=0.95)
        agent.act(model.reset())

        before = {state: node.visits for state, node in agent.tree.items()}
        for state in range(model.observation_space.n):
            agent.greedy(state)
        after = {state: node.visits for state, node in agent.tree.items()}
        assert after == before

    def test_it_does_not_spend_the_generator(self) -> None:
        model = dyna_maze(Rng(1))
        agent = searcher(Rng(3), model, discount=0.95)
        agent.act(model.reset())

        before = agent.rng.snapshot()
        for state in range(model.observation_space.n):
            agent.greedy(state)
        assert agent.rng.snapshot() == before

    def test_reading_it_does_not_move_the_run(self) -> None:
        """The whole point, checked end to end on the digest.

        One run reads the policy of every cell after every episode and the
        other reads nothing. Both have to walk the same path.
        """

        def run(probe: bool) -> str:
            root = Rng(5)
            env = dyna_maze(root.stream("env"))
            agent = searcher(root.stream("agent"), env, discount=0.95)
            record = train(env, agent, 3, discount=0.95)
            if probe:
                for state in range(env.observation_space.n):
                    agent.greedy(state)
            return record.digest.hexdigest()

        assert run(probe=True) == run(probe=False)


class TestReuse:
    def test_off_it_starts_from_nothing_every_time(self) -> None:
        model = dyna_maze(Rng(1))
        agent = searcher(Rng(3), model, discount=0.95, reuse=False)
        agent.act(0)
        first = dict(agent.tree)
        agent.act(20)
        assert 0 not in agent.tree
        assert first != agent.tree

    def test_on_it_keeps_what_it_worked_out(self) -> None:
        model = dyna_maze(Rng(1))
        agent = searcher(Rng(3), model, discount=0.95, reuse=True)
        agent.act(0)
        agent.act(20)
        assert 0 in agent.tree


class TestWhatItSpends:
    def test_it_counts_the_steps_it_simulated(self) -> None:
        model = Fork(Rng(1))
        agent = searcher(Rng(2), model)
        agent.act(0)
        # Every simulation of this model ends on its first step.
        assert agent.simulated == 20

    def test_a_deterministic_model_costs_no_draw_to_step(self) -> None:
        """A grid with no chance in it would otherwise spend the generator
        thousands of times per real step, for a branch that was never in
        doubt."""
        model = Fork(Rng(1))
        agent = searcher(Rng(2), model)
        drawn = agent._draw(agent.rng, model.transitions(0, 1))
        assert drawn.observation == 2


class TestWhatItKnows:
    def test_a_state_it_has_not_searched_is_unknown(self) -> None:
        model = dyna_maze(Rng(1))
        agent = searcher(Rng(3), model, discount=0.95)
        assert not agent.knows(40)
        assert agent.action_values(40) is None

    def test_the_lines_it_reports_are_in_a_fixed_order(self) -> None:
        model = dyna_maze(Rng(1))
        agent = searcher(Rng(3), model, discount=0.95)
        agent.act(model.reset())
        states = [int(line.split("|")[0]) for line in agent.learned()]
        assert states == sorted(states)

    def test_it_says_what_it_is(self) -> None:
        model = Fork(Rng(1))
        assert "simulations=20" in repr(searcher(Rng(1), model))


class TestOneSeedIsOneRun:
    def _digest(self, seed: int) -> str:
        root = Rng(seed)
        env = dyna_maze(root.stream("env"))
        agent = searcher(root.stream("agent"), env, discount=0.95)
        return train(env, agent, 3, discount=0.95).digest.hexdigest()

    def test_the_same_seed_replays(self) -> None:
        assert self._digest(4) == self._digest(4)

    def test_another_seed_does_not(self) -> None:
        assert self._digest(4) != self._digest(5)


class TestWhatARolloutCanAndCannotSee:
    """A rollout that reaches no ending gives the search nothing to rank by.

    This is the explanation for where `mcts` fails, and it is measurable
    without running the agent at all. On the cliff walk every step pays -1 and
    the discount is one, so a rollout that stops at the depth limit is worth
    exactly minus the depth wherever it went. The only thing that can separate
    two branches is reaching an ending, and a random policy does not.
    """

    @staticmethod
    def _arrivals(env: TabularEnv, budget: int, tries: int = 200) -> int:
        rng = Rng(2)
        arrived = 0
        for _ in range(tries):
            env.reset()
            for _ in range(budget):
                outcome = env.step(rng.below(env.action_space.n))
                if outcome.terminated:
                    arrived += 1
                    break
                if outcome.truncated:
                    break
        return arrived

    def test_thirty_random_steps_reach_no_ending_on_the_cliff_walk(self) -> None:
        assert self._arrivals(cliff_walk(Rng(1)), 30) == 0

    def test_thirty_random_steps_reach_no_ending_on_the_maze_either(self) -> None:
        """So the rollout is not what makes the search work on the maze.

        What works there is the tree, which keeps what the real steps found.
        """
        assert self._arrivals(dyna_maze(Rng(1)), 30) == 0

    def test_a_much_longer_rollout_does_arrive_sometimes(self) -> None:
        """The depth that would be needed, which is why this is a weakness
        rather than a setting somebody forgot to turn up."""
        assert self._arrivals(dyna_maze(Rng(1)), 500) > 20
