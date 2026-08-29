"""Planning at the moment of choosing, rather than in between.

Dyna and prioritised sweeping plan in the background. They spend a model on
improving a table, and when the moment comes to act they read the table. The
work is done before the question is asked, and it is done for every state
whether or not the agent will ever stand there.

Monte Carlo tree search does the opposite. It spends the model on the state it
is standing in, right now, and on the futures that actually follow from there.
The table it builds is about this decision.

## The four steps

    descend    from the state, take the action the tree ranks highest, until
               a state the tree has not seen
    expand     put that state in the tree
    roll out   act at random from there until an ending or a depth limit
    back up    credit every state and action on the way down with the return

Repeat that `simulations` times and take the action with the most visits at the
root. The most visits rather than the best mean: a mean over three samples is
high by luck often enough to matter, and the count is what the descent has
already agreed with.

## The tree is keyed by state, so it is a graph

Textbook tree search gives each path its own node. Here a state reached two
ways is one node, which is a transposition table by another name and is right
for a Markov problem: what a state is worth does not depend on how the agent
arrived. Two paths that meet pool their evidence instead of splitting it.

What it gives up is depth. A state met at depth 2 and at depth 40 shares one
node, and under a depth limit those two are not asking quite the same question.

## The descent rule is the count bonus

The rule that picks an action inside the tree is `CountBonus` from
`rel.agents.explore`, applied to the means in a node rather than to a learned
table. That is not a coincidence dressed up. Upper confidence selection inside
a tree and upper confidence exploration in a grid are the same rule, and
writing it once means the two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from rel.agents.base import Agent
from rel.agents.explore import CountBonus, argmax
from rel.core import DIGEST_FIGURES, Outcome, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete


class Node:
    """One state of the tree: how often each action was tried, and what it paid.

    The mean of an untried action reads as zero, and nothing acts on it. The
    descent takes every untried action before it ranks anything, so a mean is
    only read for an action with a count above zero.
    """

    __slots__ = ("counts", "totals")

    def __init__(self, actions: int) -> None:
        self.counts = [0] * actions
        self.totals = [0.0] * actions

    def credit(self, action: int, total: float) -> None:
        self.counts[action] += 1
        self.totals[action] += total

    def copy(self) -> Node:
        """A node that can be credited without moving this one.

        `greedy` searches on a copy of the tree, and a copy of a dictionary
        holds the same nodes. Crediting one of those would move the agent's own
        tree while claiming to leave no trace, which is what the test named
        after that claim caught.
        """
        made = Node(len(self.counts))
        made.counts[:] = self.counts
        made.totals[:] = self.totals
        return made

    def means(self) -> list[float]:
        return [
            0.0 if count == 0 else total / count
            for total, count in zip(self.totals, self.counts, strict=True)
        ]

    @property
    def visits(self) -> int:
        return sum(self.counts)

    def __repr__(self) -> str:
        return f"Node(visits={self.visits})"


class TreeSearch(Agent[int]):
    """Runs simulations from the state it is in, then acts on what they said.

    It is given the environment's own model rather than one it learned. That
    makes it a reference the way `optimal` is a reference: what it measures is
    what planning at decision time buys when the model is not the problem.
    `dyna-q` on the same grid learns its model, so a comparison of the two is a
    comparison of an agent that was given something against one that was not,
    and the page that reports it says so.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        model: TabularEnv,
        *,
        simulations: int = 50,
        discount: float = 0.95,
        confidence: float = 1.0,
        depth: int = 60,
        reuse: bool = True,
    ) -> None:
        super().__init__(rng, actions)

        if simulations < 1:
            raise ValueError("A search runs at least one simulation.")
        if depth < 1:
            raise ValueError("A simulation runs at least one step.")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("The discount is between 0 and 1.")

        self.model = model
        self.simulations = simulations
        self.discount = discount
        self.depth = depth
        #: Whether the tree survives from one decision to the next.
        #:
        #: Off, every decision starts from nothing and the agent carries
        #: nothing at all across a run: it is planning and only planning. On,
        #: the work done for one state is still there when the agent comes
        #: back to it. Both are measured, because the difference between them
        #: is the difference between planning and remembering.
        self.reuse = reuse
        self.rule = CountBonus(confidence)

        self.tree: dict[int, Node] = {}
        #: Simulated steps taken, which is what this agent spends instead of
        #: real ones. Every other planner here counts its updates and this
        #: counts the same thing.
        self.simulated = 0

    # -- Acting --------------------------------------------------------------

    def act(self, observation: int) -> int:
        if not self.reuse:
            self.tree = {}
        self._search(self.tree, self.rng, observation)
        return self.actions.start + self._best(self.tree, self.rng, observation)

    def greedy(self, observation: int) -> int:
        """The action a search from here would take, without leaving a trace.

        A renderer asks this for every cell of a grid and an evaluation asks it
        for every state. Both have to be free of side effects, and for this
        agent that means two of them. A search that grew the agent's tree would
        put nodes in it for cells the agent has never stood in, and `knows`
        counts on that not happening. A search that spent the agent's own
        chance would move the run it is being asked about, which is a fault
        this project has already had once and measured.

        So this searches on a copy of the tree with a copy of the generator.
        """
        tree = {state: node.copy() for state, node in self.tree.items()}
        probe = Rng.restore(*self.rng.snapshot())
        self._search(tree, probe, observation)
        return self.actions.start + self._best(tree, probe, observation)

    def _best(self, tree: dict[int, Node], rng: Rng, state: int) -> int:
        node = tree.get(state)
        if node is None:
            return rng.below(self.actions.n)
        return argmax(rng, [float(count) for count in node.counts])

    # -- Searching -----------------------------------------------------------

    def _search(self, tree: dict[int, Node], rng: Rng, root: int) -> None:
        for _ in range(self.simulations):
            self._simulate(tree, rng, root)

    def _simulate(self, tree: dict[int, Node], rng: Rng, root: int) -> None:
        """One descent, one expansion, one rollout and one backup."""
        path: list[tuple[int, int]] = []
        rewards: list[float] = []

        state = root
        ended = False
        depth = 0

        while depth < self.depth:
            node = tree.get(state)
            if node is None:
                tree[state] = Node(self.actions.n)
                break

            action = self.rule.choose(rng, node.means(), node.counts, 0, 0)
            reward, landed, ended = self._step(rng, state, action)
            path.append((state, action))
            rewards.append(reward)

            state = landed
            depth += 1
            if ended:
                break

        total = 0.0 if ended else self._roll_out(rng, state, self.depth - depth)
        for (visited, action), reward in zip(
            reversed(path), reversed(rewards), strict=True
        ):
            total = reward + self.discount * total
            tree[visited].credit(action, total)

    def _roll_out(self, rng: Rng, state: int, budget: int) -> float:
        """A random policy from here, and what it collected.

        The crudest possible estimate of what a leaf is worth, and the only
        part of this that needs no table, which is what lets the search say
        something about a state nothing has ever visited.
        """
        total = 0.0
        scale = 1.0
        for _ in range(budget):
            reward, state, ended = self._step(rng, state, rng.below(self.actions.n))
            total += scale * reward
            scale *= self.discount
            if ended:
                break
        return total

    def _step(self, rng: Rng, state: int, action: int) -> tuple[float, int, bool]:
        """One step of the model, with a branch drawn by its probability."""
        self.simulated += 1
        drawn = self._draw(rng, self.model.transitions(state, action))
        return drawn.reward, drawn.observation, drawn.terminated

    def _draw(self, rng: Rng, branches: Sequence[Outcome]) -> Outcome:
        if len(branches) == 1:
            # The common case on these grids, and worth not drawing for. An
            # environment with no chance in it would otherwise spend the
            # generator once per simulated step, which is thousands of draws
            # per real one.
            return branches[0]
        return branches[rng.weighted_index([one.probability for one in branches])]

    # -- What it knows -------------------------------------------------------

    def action_values(self, observation: int) -> Sequence[float] | None:
        node = self.tree.get(observation)
        return None if node is None else node.means()

    def knows(self, observation: int) -> bool:
        return observation in self.tree

    def learned(self) -> Iterator[str]:
        """The tree, as counts and means.

        An agent that is not reusing its tree has learned nothing by the end of
        a run, and this reports the tree of its last decision. That is what it
        has, and saying so is better than reporting nothing and letting the two
        settings look alike.
        """
        for state in sorted(self.tree):
            node = self.tree[state]
            counts = ",".join(str(count) for count in node.counts)
            means = ",".join(f"{one:.{DIGEST_FIGURES}g}" for one in node.means())
            yield f"{state}|{counts}|{means}"

    def __repr__(self) -> str:
        return (
            f"TreeSearch(simulations={self.simulations}, "
            f"discount={self.discount:g}, depth={self.depth}, reuse={self.reuse})"
        )


__all__ = ["Node", "TreeSearch"]
