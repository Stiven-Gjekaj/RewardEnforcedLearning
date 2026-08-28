"""How hard an agent optimises, and what that does to the number nobody pays.

The three gaming environments each report two things: what the reward paid, and
what was actually wanted. Both are measured at the best possible policy, which
is what says the gaming is the answer to the question rather than an accident an
agent stumbled into.

What that does not say is how the two numbers move apart on the way there. A
policy that optimises nothing games nothing. The question this module asks is
whether the gap opens gradually, or all at once, or not at all.

## What pressure means here

One dial. Take the best policy under the stated reward and follow it with
probability one minus epsilon, and act uniformly otherwise. At epsilon one the
agent optimises nothing. At epsilon zero it is the optimum. Pressure is one
minus epsilon, so it reads left to right as an agent that tries harder.

Nothing is learned along this ladder. That is deliberate: a learner walking
towards the optimum also gets better at the environment in ways that have
nothing to do with the reward it was given, and the two effects would arrive
together with no way to tell them apart. A fixed policy followed with noise
moves one thing only.

## Why the numbers are shares rather than returns

The three environments pay in different units, and the thing each one really
wanted is in a third set of units again. Laps, a broken vase and a share of
comfortable steps cannot be put on one axis as they are.

So each number is reported as where it sits between the worst policy and the
best one for that number: zero is what a uniform policy gets, one is the most
that is available. Both series then run from zero to one on every environment,
and the gap between them is a number that means the same thing everywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from rel.agents.base import Agent
from rel.agents.dp import value_iteration
from rel.core import TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete
from rel.training import evaluate

Builder = Callable[[Rng], TabularEnv]

#: The ladder, as epsilons. One is a uniform policy and zero is the optimum.
#: Six rungs is enough to show a shape and few enough that `rel gaming` still
#: fits on a screen.
LADDER = (1.0, 0.8, 0.5, 0.3, 0.1, 0.0)


class PressuredAgent(Agent[int]):
    """Follows a fixed policy with probability one minus epsilon.

    `greedy` is the same as `act` here, as it is for the random agent. The
    epsilon is not exploration laid over a policy that would rather do
    something else. It is the policy being measured, so an evaluation that
    dropped it would measure a different agent at every rung.
    """

    def __init__(
        self, rng: Rng, actions: Discrete, policy: Sequence[int], epsilon: float
    ) -> None:
        super().__init__(rng, actions)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon is a probability. {epsilon} is not one.")
        self.policy = tuple(policy)
        self.epsilon = epsilon

    def act(self, observation: int) -> int:
        if self.rng.chance(self.epsilon):
            return self.actions.start + self.rng.below(self.actions.n)
        return self.policy[observation]

    def greedy(self, observation: int) -> int:
        return self.act(observation)

    def __repr__(self) -> str:
        return f"PressuredAgent(epsilon={self.epsilon:g}, states={len(self.policy)})"


@dataclass(frozen=True)
class Rung:
    """One step of the ladder, and what the environment paid for it."""

    epsilon: float
    paid: float
    audit: Mapping[str, float]

    @property
    def pressure(self) -> float:
        """One minus epsilon. Zero optimises nothing, one is the optimum."""
        return 1.0 - self.epsilon


def ladder(
    build: Builder,
    discount: float,
    *,
    epsilons: Sequence[float] = LADDER,
    episodes: int = 20,
    seed: int = 1,
) -> list[Rung]:
    """Run the best policy at every rung of the ladder, and report both numbers.

    The policy is solved once. Every rung then follows that same policy with a
    different amount of noise, so the only thing that changes along the ladder
    is how much of the time the agent does the optimal thing.

    Several episodes are averaged at each rung because a noisy policy is noisy,
    and one episode of it says very little.
    """
    solved = value_iteration(build(Rng(seed).stream("env")), discount=discount)

    rungs: list[Rung] = []
    for index, epsilon in enumerate(epsilons):
        rng = Rng(seed + index)
        env = build(rng.stream("env"))
        agent = PressuredAgent(
            rng.stream("agent"), env.action_space, solved.policy, epsilon
        )
        record = evaluate(env, agent, episodes, discount=discount)
        rungs.append(
            Rung(epsilon, record.final(episodes), record.final_audit(episodes))
        )

    return rungs


def share(value: float, worst: float, best: float) -> float:
    """Where `value` sits between `worst` and `best`, from zero to one.

    Clamped, because a noisy policy can beat the worst case or fall below it,
    and a share outside zero to one would draw a chart that goes off its own
    axis. When the two ends are the same number there is nothing to be a share
    of, and the answer is zero rather than a division by it.
    """
    if best == worst:
        return 0.0
    return max(0.0, min(1.0, (value - worst) / (best - worst)))


__all__ = ["LADDER", "Builder", "PressuredAgent", "Rung", "ladder", "share"]
