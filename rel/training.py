"""Running an agent in an environment, and writing down what happened.

The loop is nine lines. Everything else in this module is about the record it
keeps, because a run that produces one number nobody can check is not worth the
electricity.

## The digest

Every run hashes its own transitions and prints the result. Two runs with the
same digest took the same path through the environment, step for step.

The digest is exact for a tabular environment and platform bound for a
continuous one. The cart pole and the mountain car call `cos` and `sin`, and a
C library is allowed to return a result one bit different from another one.
That bit grows: five hundred steps of a swinging pole turn it into a visible
difference, and no amount of rounding in the digest hides that. So a digest
from the cliff walk can be compared between two machines, and a digest from the
cart pole can be compared between two runs on one machine. Both are worth
having, and saying which is which is worth more.

## Learning and measuring are two runs

`train` reports the return of the policy that is exploring. `evaluate` reports
the return of the greedy policy with the learning switched off.

They answer different questions and they move in opposite directions near the
end of a run: exploration keeps costing what it costs while the greedy policy
keeps improving. A report that gives one of them and calls it the result is
half an answer, so `train` returns both.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rel.agents.base import Agent, Transition
from rel.core import DIGEST_FIGURES, Env, encoded
from rel.metrics import Summary, last_mean, summarise


def digest_line(transition: Transition[Any, Any]) -> str:
    """The one line of text that stands for a transition.

    This is what the digest hashes, and `rel.recording` writes a file whose
    step lines are built from the same pieces. Two ways of spelling one
    transition would be two digests, and the file would be checkable against
    only one of them.

    The action goes through `encoded` for the same reason the observations do.
    A whole number spells the same either way, so no digest of a run made
    before this moves. A point in a box does not: Python writes a tuple of
    floats to seventeen digits each, and `rel.recording` could write a file
    that it could not read back.
    """
    return (
        f"{encoded(transition.observation)}"
        f"|{encoded(transition.action)}"
        f"|{transition.reward:.{DIGEST_FIGURES}g}"
        f"|{int(transition.terminated)}{int(transition.truncated)}\n"
    )


class Digest:
    """A running hash of every transition of a run."""

    __slots__ = ("_hash", "_steps")

    def __init__(self) -> None:
        self._hash = hashlib.blake2b(digest_size=8)
        self._steps = 0

    def add(self, transition: Transition[Any, Any]) -> None:
        self._steps += 1
        self._hash.update(digest_line(transition).encode("ascii"))

    def add_line(self, line: str) -> None:
        """Take a line that was built earlier, as a recording holds it.

        Reading a file back has the pieces and not the transition they came
        from, and going through `Transition` to get the line again would be a
        second way of spelling it.
        """
        self._steps += 1
        self._hash.update(line.encode("ascii"))

    @property
    def steps(self) -> int:
        return self._steps

    def hexdigest(self) -> str:
        return self._hash.hexdigest()

    def __str__(self) -> str:
        return self.hexdigest()


def digest_of(agent: Agent[Any, Any]) -> str | None:
    """A hash of what this agent has learned, or `None` if it says nothing.

    The run digest hashes the transitions, so two agents that happen to walk
    the same path get the same one. That answers "did these two runs do the
    same thing" and not "did these two agents come to the same conclusion".

    The two are kept apart rather than merged. Merging them would change what
    the run digest means, and every number in the documentation that was
    compared against one would silently stop being comparable.

    `None` rather than the hash of nothing, because the hash of nothing is the
    same for every agent that keeps nothing and would read as a fact about the
    agent.

    ## It is not the same on every Python

    CPython 3.12 gave `sum()` compensated summation over floats. That is more
    accurate than adding them up one at a time and it is therefore a different
    answer, so an agent whose arithmetic goes through `sum` can learn a table
    that differs in its last bits between one interpreter and another.

    Measured on 3.11 against 3.12 and 3.13. Two of six move and four do not:

        tile-sarsa      cart pole, 20 episodes     moves
        reinforce       cliff walk, 50 episodes    moves
        expected-sarsa  cliff walk, 100 episodes   the same on all three
        tree-backup     cliff walk, 100 episodes   the same on all three
        off-policy-mc   cliff walk, 100 episodes   the same on all three
        q-learning      cliff walk, 100 episodes   the same on all three

    The two that move are the two whose value is a sum over many floats: eight
    tile weights, or a whole network. A tabular agent keeps one number for
    each cell and adds to it in place, so there is no long sum for the
    compensation to change. Every one of the four sums floats somewhere and
    none of them sums enough of them.

    It is a threshold rather than a property of an agent. The same
    `tile-sarsa` run is the same on all three at ten episodes and moves at
    twenty, `reinforce` is the same at twenty and moves at fifty, and
    `actor-critic` is the same at a hundred and moves at four hundred.
    `TestWhichDigestsMoveBetweenPythons` in `tests/test_training.py` holds the
    six, and the check on it is the version matrix in CI rather than anything
    the test does: the same numbers run on all three interpreters.

    **The run digest does not move on any of them.** It hashes transitions,
    and an observation written to a fixed number of figures survives a
    difference in the last bits of a value, because a policy is a comparison
    between values rather than a value. That is the other reason to keep the
    two apart. `TestTheDigestIsNotStableAcrossPythons` in
    `tests/test_linear.py` is the demonstration.
    """
    running = hashlib.blake2b(digest_size=8)
    empty = True
    for line in agent.learned():
        empty = False
        running.update(f"{line}\n".encode("ascii"))
    return None if empty else running.hexdigest()


@dataclass(frozen=True, slots=True)
class Episode:
    """What one episode did."""

    number: int
    total_reward: float
    discounted_reward: float
    length: int
    terminated: bool
    audit: Mapping[str, float]

    #: Every reward in order, kept only when the caller asked for it.
    #:
    #: A bandit run needs this. The interesting curve there is the reward at
    #: each pull averaged over problems, not the total of a whole problem, and
    #: the total hides the shape entirely: an agent that finds the best lever
    #: on pull fifty and one that finds it on pull five hundred differ by a
    #: few percent in the total and are not the same agent.
    rewards: tuple[float, ...] = ()


@dataclass
class Record:
    """Every episode of a run, and the numbers taken from them."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    discounted: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    audits: dict[str, list[float]] = field(default_factory=dict)
    step_rewards: list[tuple[float, ...]] = field(default_factory=list)
    digest: Digest = field(default_factory=Digest)

    def add(self, episode: Episode) -> None:
        self.returns.append(episode.total_reward)
        self.lengths.append(episode.length)
        self.discounted.append(episode.discounted_reward)
        self.terminated.append(episode.terminated)
        if episode.rewards:
            self.step_rewards.append(episode.rewards)
        for key, value in episode.audit.items():
            self.audits.setdefault(key, []).append(value)

    def reward_by_step(self) -> list[float]:
        """The mean reward at each step, over the episodes that reached it.

        Empty unless the run was asked to keep its steps.
        """
        if not self.step_rewards:
            return []

        longest = max(len(rewards) for rewards in self.step_rewards)
        totals = [0.0] * longest
        counts = [0] * longest
        for rewards in self.step_rewards:
            for index, reward in enumerate(rewards):
                totals[index] += reward
                counts[index] += 1
        return [total / count for total, count in zip(totals, counts, strict=True)]

    def __len__(self) -> int:
        return len(self.returns)

    @property
    def steps(self) -> int:
        return sum(self.lengths)

    def summary(self) -> Summary:
        return summarise(self.returns)

    def final(self, over: int = 100) -> float:
        """The mean return of the last `over` episodes."""
        return last_mean(self.returns, over)

    def audit_summary(self) -> dict[str, Summary]:
        return {key: summarise(values) for key, values in self.audits.items()}

    def final_audit(self, over: int = 100) -> dict[str, float]:
        return {key: last_mean(values, over) for key, values in self.audits.items()}


def run_episode(
    env: Env[Any, Any],
    agent: Agent[Any, Any],
    *,
    learn: bool = True,
    greedy: bool = False,
    number: int = 0,
    digest: Digest | None = None,
    discount: float = 1.0,
    keep_steps: bool = False,
) -> Episode:
    """One episode, from reset to an ending.

    `greedy` runs the policy with no exploration, which is what a measurement
    wants. `learn` decides whether the agent is told anything, which is a
    separate choice: an agent can be watched greedily while it still learns.
    """
    observation = env.reset()
    agent.start_episode()
    total = 0.0
    discounted = 0.0
    weight = 1.0
    length = 0
    terminated = False
    rewards: list[float] = []

    while True:
        action = agent.greedy(observation) if greedy else agent.act(observation)
        outcome = env.step(action)

        transition = Transition(
            observation=observation,
            action=action,
            reward=outcome.reward,
            next_observation=outcome.observation,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
        )
        if digest is not None:
            digest.add(transition)
        if learn:
            agent.observe(transition)

        if keep_steps:
            rewards.append(outcome.reward)

        total += outcome.reward
        discounted += weight * outcome.reward
        weight *= discount
        length += 1
        observation = outcome.observation

        if outcome.done:
            terminated = outcome.terminated
            break

    if learn:
        agent.end_episode()

    return Episode(
        number=number,
        total_reward=total,
        discounted_reward=discounted,
        length=length,
        terminated=terminated,
        audit=dict(env.audit()),
        rewards=tuple(rewards),
    )


def train(
    env: Env[Any, Any],
    agent: Agent[Any, Any],
    episodes: int,
    *,
    discount: float = 1.0,
    keep_steps: bool = False,
    on_episode: Callable[[Episode, Record], None] | None = None,
    digest: Digest | None = None,
) -> Record:
    """Run the agent for `episodes` episodes while it learns.

    `digest` lets the caller supply the thing that watches every step. A run
    that is being recorded hands it a `rel.recording.Recorder`, which is a
    digest that also keeps what it hashed.
    """
    record = Record(digest=digest) if digest is not None else Record()
    for number in range(episodes):
        episode = run_episode(
            env,
            agent,
            learn=True,
            greedy=False,
            number=number,
            digest=record.digest,
            discount=discount,
            keep_steps=keep_steps,
        )
        record.add(episode)
        if on_episode is not None:
            on_episode(episode, record)
    return record


def evaluate(
    env: Env[Any, Any],
    agent: Agent[Any, Any],
    episodes: int,
    *,
    discount: float = 1.0,
    keep_steps: bool = False,
) -> Record:
    """Run the greedy policy with the learning switched off."""
    record = Record()
    for number in range(episodes):
        episode = run_episode(
            env,
            agent,
            learn=False,
            greedy=True,
            number=number,
            digest=record.digest,
            discount=discount,
            keep_steps=keep_steps,
        )
        record.add(episode)
    return record


__all__ = [
    "DIGEST_FIGURES",
    "Digest",
    "Episode",
    "Record",
    "digest_line",
    "digest_of",
    "encoded",
    "evaluate",
    "run_episode",
    "train",
]
