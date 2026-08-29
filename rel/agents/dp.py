"""Solving an environment exactly, by reading its model.

Nothing here learns. These functions take the transitions an environment
declares and work out what the best possible policy is worth, by arithmetic.

That number is what makes every other claim in this project checkable. "The
agent reached -17" says nothing on its own. "The agent reached -17 and the best
possible is -13" says what is left to gain, and it comes from the environment
rather than from a paper.

## Two ways for a policy to be bad

A learned policy can be bad in two different ways, and one number cannot report
both.

It can be slow: it reaches the goal, and it takes longer than it needs to. Its
value is finite and worse than the best.

It can be stuck: it never reaches the goal at all. With no discount its value
is minus infinity where every step costs something, and evaluating it does not
converge. Running more sweeps does not help, because there is nothing to
converge to.

`evaluate_policy` reports which of the two happened rather than returning a
large negative number for both. The check is a search over the graph the policy
makes, and it is exact.

## Why policy iteration starts where it does

Policy iteration as the book writes it starts from any policy and evaluates it
to convergence. With no discount, that only works when the policy it starts
from reaches an ending. "Take action zero everywhere" does not: on the cliff
walk it walks into the top left corner and pushes at the wall for ever.

This was found by running it. The first version of this module sat in policy
evaluation for twenty thousand sweeps on every round.

The fix is to start from a policy that reaches an ending by construction.
`_proper_policy` builds one with a backward search from the terminal states and
no arithmetic at all: a state takes any action that gives it a chance of
landing somewhere already known to finish. Every policy after it is at least as
good, so every evaluation converges, and policy iteration then costs about what
value iteration costs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rel.agents.base import Agent
from rel.core import TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete

#: How small the largest change in a sweep has to be before the sweep stops.
TOLERANCE = 1e-9

#: A cap on the sweeps of one evaluation. Reaching it is reported as a failure
#: rather than accepted, because a value taken from a run that was cut short is
#: whatever the count happened to reach.
MAX_SWEEPS = 20_000


@dataclass(frozen=True, slots=True)
class Solution:
    """What dynamic programming worked out."""

    values: tuple[float, ...]
    policy: tuple[int, ...]
    sweeps: int
    start_value: float

    def __str__(self) -> str:
        return f"start value {self.start_value:.4f} after {self.sweeps} sweeps"


@dataclass(frozen=True, slots=True)
class PolicyReport:
    """What a fixed policy is worth, and whether it finishes at all.

    A value that is not a number belongs to a state this policy never reaches.
    That is different from a value of zero, which would be a claim about a
    state the policy does visit.
    """

    reaches_end: bool
    values: tuple[float, ...]
    start_value: float

    def __str__(self) -> str:
        if not self.reaches_end:
            return "does not reach an ending"
        return f"start value {self.start_value:.4f}"


class DidNotSettleError(RuntimeError):
    """A sweep ran to its cap with the values still moving."""


# -- The public calls -------------------------------------------------------


def value_iteration(
    env: TabularEnv,
    discount: float = 1.0,
    tolerance: float = TOLERANCE,
    max_sweeps: int = MAX_SWEEPS,
) -> Solution:
    """The best possible value of every state, and a policy that reaches it.

    The sweeps are in place rather than into a second array. That is Gauss and
    Seidel rather than Jacobi, and it converges in fewer sweeps because a value
    updated earlier in the sweep is used later in the same sweep. It gives the
    same answer.
    """
    count = env.observation_space.n
    values = [0.0] * count
    terminal = env.terminal_states()
    live = [state for state in range(count) if state not in terminal]

    for sweeps in range(1, max_sweeps + 1):  # noqa: B007
        largest = 0.0
        for state in live:
            best = max(
                _backup(env, values, state, action, discount)
                for action in env.action_space
            )
            largest = max(largest, abs(best - values[state]))
            values[state] = best
        if largest < tolerance:
            break
    else:
        raise DidNotSettleError(
            f"Value iteration ran {max_sweeps} sweeps and the values were "
            f"still moving by {largest:g}."
        )

    policy = tuple(
        _greedy_action(env, values, state, discount) for state in range(count)
    )
    return Solution(
        values=tuple(values),
        policy=policy,
        sweeps=sweeps,
        start_value=_start_value(env, values),
    )


def policy_iteration(
    env: TabularEnv,
    discount: float = 1.0,
    tolerance: float = TOLERANCE,
    max_sweeps: int = MAX_SWEEPS,
    max_rounds: int = 1000,
) -> Solution:
    """The same answer, reached by improving a policy rather than a value.

    This exists to be compared with `value_iteration`. Both give the optimal
    value, and a test holds them to the same numbers, which is a check on each
    of them that neither can give itself.
    """
    terminal = env.terminal_states()
    policy = _proper_policy(env, terminal)
    values: tuple[float, ...] = ()

    for rounds in range(1, max_rounds + 1):  # noqa: B007
        values = _evaluate(env, policy, discount, tolerance, max_sweeps, terminal)

        settled = True
        for state in range(env.observation_space.n):
            if state in terminal:
                continue
            better = _improve(env, values, state, policy[state], discount, tolerance)
            if better != policy[state]:
                policy[state] = better
                settled = False
        if settled:
            break
    else:
        raise DidNotSettleError(
            f"The policy was still changing after {max_rounds} rounds."
        )

    return Solution(
        values=values,
        policy=tuple(policy),
        sweeps=rounds,
        start_value=_start_value(env, values),
    )


def evaluate_policy(
    env: TabularEnv,
    policy: Sequence[int],
    discount: float = 1.0,
    tolerance: float = TOLERANCE,
    max_sweeps: int = MAX_SWEEPS,
) -> PolicyReport:
    """What this policy is worth in every state, worked out exactly.

    This is the honest way to score a learned policy on a tabular environment.
    Running the greedy policy for a few episodes and averaging gives an
    estimate with noise in it, and gives the step limit rather than an answer
    when the policy does not finish.
    """
    terminal = env.terminal_states()
    edges = _policy_edges(env, policy, terminal)
    forward = _reachable_from_start(env, edges)

    if not forward <= _can_finish(edges, terminal):
        return PolicyReport(
            reaches_end=False,
            values=(-math.inf,) * env.observation_space.n,
            start_value=-math.inf,
        )

    # Only the states the policy can actually be in. A state it can never
    # reach has no bearing on what it is worth, and sweeping one is not merely
    # wasted work: if the policy circles there, the sweep never settles, and
    # the whole evaluation fails over a corner of the grid that the policy
    # never visits. That happened to `rel train q-learning --env cliff`, which
    # reported no exact value at all until this was fixed.
    values = _evaluate(
        env, policy, discount, tolerance, max_sweeps, terminal, live=forward
    )
    return PolicyReport(
        reaches_end=True,
        values=values,
        start_value=_start_value(env, values),
    )


def average_reward(env: TabularEnv, policy: Sequence[int]) -> float | None:
    """The reward per step of following this policy for ever.

    The number a task with no ending is really scored by. A discounted value
    answers "what is the future worth from here, with later worth less", and on
    a task that never ends that question has an answer for every discount and a
    different one for each. This asks what the policy collects per step, which
    has one answer and no setting in it.

    `None` when the policy is not deterministic enough to walk, which here
    means when a transition it takes has more than one branch. Averaging over a
    stationary distribution would answer that case, and it needs a linear solve
    this project does not have. The environments with no ending in it are
    deterministic, so the walk is the whole of what is needed and it is exact.

    The walk goes forward from a start until a state repeats. What lies between
    the first visit and the second is the cycle the policy settles into, and
    the average over that cycle is the answer: the steps before the cycle are
    finitely many and a per step average over for ever does not see them.
    """
    seen: dict[int, int] = {}
    rewards: list[float] = []

    state = env.start_states()[0][1]
    while state not in seen:
        branches = env.transitions(state, policy[state])
        if len(branches) != 1:
            return None
        seen[state] = len(rewards)
        rewards.append(branches[0].reward)
        state = branches[0].observation

    cycle = rewards[seen[state] :]
    return sum(cycle) / len(cycle)


def reaches_end(env: TabularEnv, policy: Sequence[int]) -> bool:
    """True if the policy reaches an ending from every state it can be in.

    Two searches. The first walks forward from the start states and collects
    every state the policy can reach. The second walks backward from the
    terminal states over the same edges. A policy is proper when the first set
    fits inside the second.

    A policy that circles for ever is the common failure, and it is not
    something to detect by watching a number get large.
    """
    terminal = env.terminal_states()
    edges = _policy_edges(env, policy, terminal)
    forward = _reachable_from_start(env, edges)
    return forward <= _can_finish(edges, terminal)


def _reachable_from_start(env: TabularEnv, edges: dict[int, list[int]]) -> set[int]:
    """Every state the policy can put the agent in, starting from a start."""
    found: set[int] = set()
    frontier = [state for _, state in env.start_states()]
    while frontier:
        state = frontier.pop()
        if state in found:
            continue
        found.add(state)
        frontier.extend(edges[state])
    return found


# -- The searches -----------------------------------------------------------


def _policy_edges(
    env: TabularEnv, policy: Sequence[int], terminal: frozenset[int]
) -> dict[int, list[int]]:
    """Where each state can land, under this policy."""
    edges: dict[int, list[int]] = {}
    for state in range(env.observation_space.n):
        if state in terminal:
            edges[state] = []
            continue
        edges[state] = [
            outcome.observation
            for outcome in env.transitions(state, policy[state])
            if outcome.probability > 0.0
        ]
    return edges


def _can_finish(edges: dict[int, list[int]], terminal: frozenset[int]) -> set[int]:
    """Every state that can reach a terminal state by following the edges."""
    backward: dict[int, list[int]] = {state: [] for state in edges}
    for state, landings in edges.items():
        for landing in landings:
            backward[landing].append(state)

    found = set(terminal)
    frontier = list(terminal)
    while frontier:
        state = frontier.pop()
        for before in backward.get(state, ()):
            if before not in found:
                found.add(before)
                frontier.append(before)
    return found


def _proper_policy(env: TabularEnv, terminal: frozenset[int]) -> list[int]:
    """A policy that reaches an ending, built without evaluating anything.

    A backward search. A state joins the set of states known to finish as soon
    as it has an action with any chance of landing in that set, and it takes
    that action.

    Following such a policy moves the agent closer to the set with a
    probability above zero on every step, so it arrives with probability one.
    That is enough for the evaluations that follow to converge, and it costs no
    arithmetic to build.

    A state with no such action cannot reach an ending under any policy. It
    keeps action zero, and `reaches_end` reports it if it is ever reached.
    """
    policy = [env.action_space.start] * env.observation_space.n
    known = set(terminal)

    changed = True
    while changed:
        changed = False
        for state in range(env.observation_space.n):
            if state in known:
                continue
            for action in env.action_space:
                lands_well = any(
                    outcome.probability > 0.0
                    and (outcome.terminated or outcome.observation in known)
                    for outcome in env.transitions(state, action)
                )
                if lands_well:
                    policy[state] = action
                    known.add(state)
                    changed = True
                    break

    return policy


# -- The arithmetic ---------------------------------------------------------


def _start_value(env: TabularEnv, values: Sequence[float]) -> float:
    return sum(share * values[state] for share, state in env.start_states())


def _backup(
    env: TabularEnv,
    values: Sequence[float],
    state: int,
    action: int,
    discount: float,
) -> float:
    """What one action is worth, given what the states after it are worth."""
    total = 0.0
    for outcome in env.transitions(state, action):
        if outcome.probability <= 0.0:
            continue
        if outcome.terminated or discount == 0.0:
            total += outcome.probability * outcome.reward
            continue

        ahead = values[outcome.observation]
        if ahead == -math.inf:
            # Any chance at all of never ending makes the whole action worth
            # minus infinity. Working it out as a weighted sum would multiply a
            # probability by an infinity and give a NaN, which then spreads
            # through the table without any comparison complaining.
            return -math.inf
        total += outcome.probability * (outcome.reward + discount * ahead)
    return total


def _greedy_action(
    env: TabularEnv, values: Sequence[float], state: int, discount: float
) -> int:
    """The best action, with ties going to the lowest number.

    A fixed rule rather than a random one. Dynamic programming has no source of
    chance, and two runs of it have to give the same policy or the policy
    cannot be compared with anything.
    """
    best_action = env.action_space.start
    best_value = -math.inf
    for action in env.action_space:
        value = _backup(env, values, state, action, discount)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def _improve(
    env: TabularEnv,
    values: Sequence[float],
    state: int,
    current: int,
    discount: float,
    tolerance: float,
) -> int:
    """The action to switch to, or the current one if nothing clearly beats it.

    The margin matters. Two actions worth the same to within the last few bits
    of a float take turns being the larger one, so a policy built by taking the
    maximum keeps changing and policy iteration never settles. The frozen lake
    at size eight is one of those, and it ran for a thousand rounds without
    stopping before this margin was put in.
    """
    best = _greedy_action(env, values, state, discount)
    if best == current:
        return current

    gain = _backup(env, values, state, best, discount) - _backup(
        env, values, state, current, discount
    )
    return best if gain > tolerance else current


def _evaluate(
    env: TabularEnv,
    policy: Sequence[int],
    discount: float,
    tolerance: float,
    max_sweeps: int,
    terminal: frozenset[int],
    live: set[int] | None = None,
) -> tuple[float, ...]:
    """What each state is worth under this policy.

    `live` names the states to sweep. A state left out keeps a value that is
    not a number, which says "this policy never goes here" rather than zero,
    which would say "this policy goes here and it is worth nothing".
    """
    count = env.observation_space.n
    if live is None:
        values = [0.0] * count
        sweeping = [state for state in range(count) if state not in terminal]
    else:
        values = [0.0 if state in live else math.nan for state in range(count)]
        sweeping = [state for state in sorted(live) if state not in terminal]

    for _ in range(max_sweeps):
        largest = 0.0
        for state in sweeping:
            updated = _backup(env, values, state, policy[state], discount)
            largest = max(largest, abs(updated - values[state]))
            values[state] = updated
        if largest < tolerance:
            break
    else:
        raise DidNotSettleError(
            f"Policy evaluation ran {max_sweeps} sweeps and the values were "
            f"still moving by {largest:g}. The policy probably never ends."
        )

    return tuple(values)


class FixedPolicyAgent(Agent[int]):
    """Plays a policy that was worked out somewhere else and never changes it.

    The optimal policy from `value_iteration` becomes an agent through this,
    which is what lets the best possible policy run through the same loop, the
    same recorder and the same renderer as a learned one. A reference measured
    differently from the thing it references is not a reference.
    """

    def __init__(self, rng: Rng, actions: Discrete, policy: Sequence[int]) -> None:
        super().__init__(rng, actions)
        self.policy = tuple(policy)

    def act(self, observation: int) -> int:
        return self.policy[observation]

    def greedy(self, observation: int) -> int:
        return self.policy[observation]

    def __repr__(self) -> str:
        return f"FixedPolicyAgent(states={len(self.policy)})"


__all__ = [
    "MAX_SWEEPS",
    "TOLERANCE",
    "DidNotSettleError",
    "FixedPolicyAgent",
    "PolicyReport",
    "Solution",
    "average_reward",
    "evaluate_policy",
    "policy_iteration",
    "reaches_end",
    "value_iteration",
]
