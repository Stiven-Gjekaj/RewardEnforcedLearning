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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rel.agents.base import Agent, Transition
from rel.core import Env
from rel.metrics import Summary, last_mean, summarise

#: How many significant figures of a float go into the digest. Twelve is far
#: more than enough to catch a change in behaviour and short enough that the
#: text stays small.
DIGEST_FIGURES = 12


def _encode(value: Any) -> str:
    """One observation, as text that does not depend on how Python prints it."""
    if isinstance(value, tuple):
        return ",".join(_encode(item) for item in value)
    if isinstance(value, float):
        return f"{value:.{DIGEST_FIGURES}g}"
    return str(value)


class Digest:
    """A running hash of every transition of a run."""

    __slots__ = ("_hash", "_steps")

    def __init__(self) -> None:
        self._hash = hashlib.blake2b(digest_size=8)
        self._steps = 0

    def add(self, transition: Transition[Any]) -> None:
        self._steps += 1
        line = (
            f"{_encode(transition.observation)}"
            f"|{transition.action}"
            f"|{transition.reward:.{DIGEST_FIGURES}g}"
            f"|{int(transition.terminated)}{int(transition.truncated)}\n"
        )
        self._hash.update(line.encode("ascii"))

    @property
    def steps(self) -> int:
        return self._steps

    def hexdigest(self) -> str:
        return self._hash.hexdigest()

    def __str__(self) -> str:
        return self.hexdigest()


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
    env: Env[Any],
    agent: Agent[Any],
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
    env: Env[Any],
    agent: Agent[Any],
    episodes: int,
    *,
    discount: float = 1.0,
    keep_steps: bool = False,
    on_episode: Callable[[Episode, Record], None] | None = None,
) -> Record:
    """Run the agent for `episodes` episodes while it learns."""
    record = Record()
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
    env: Env[Any],
    agent: Agent[Any],
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


def returns_of(record: Record) -> Sequence[float]:
    return record.returns


__all__ = [
    "DIGEST_FIGURES",
    "Digest",
    "Episode",
    "Record",
    "evaluate",
    "run_episode",
    "train",
]
