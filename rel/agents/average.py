"""Learning without a discount, by subtracting the average reward instead.

Every other learning agent here maximises a discounted return. That works
because every grid in this project has a goal, so an episode ends and the
return is a finite number whatever the discount is.

On a task that never ends the discount stops being a convenience and becomes
part of the question. `rel/envs/continuing.py` has an environment where two
policies collect 1 and 2 reward per step, and where any discount below 0.7394
makes the first of them the better answer. The agent that picks it is not going
wrong. It is answering what it was asked.

## The differential value

The average reward formulation asks a different question. Let `r` be the reward
per step the policy collects for ever. Then a differential value is

    Q(s, a) = the reward from here, with r subtracted from every step

which is finite with no discount anywhere, because a policy already collecting
`r` a step adds nothing to that in the long run. The update is the one step
version of it:

    error = reward - average + max Q(s', a) - Q(s, a)
    Q(s, a) += step_size * error
    average += average_step * step_size * error

Two things follow from writing it that way and both are worth knowing.

**The values are relative, and only differences mean anything.** Adding the
same number to every cell changes nothing, so a value map of this agent is a
picture of what is better than what rather than of what anything is worth. The
policy read off it is the same policy, which is all that is asked of it.

**The average is learned from the same error.** It is not a running mean of the
rewards that arrived. A running mean would be the rate of the behaviour policy,
exploration included, and the rate this update needs is the one of the policy
being learned.

## Only the off-policy one is here

The on-policy version needs the action the policy really takes next, which
means holding the last transition and waiting a step, exactly as `Sarsa` does
and for exactly the reasons written there. That is worth doing when the answer
differs, and on the environment this was built for it does not: the finding is
about the discount and not about which of the two updates carries it.

## It assumes the task does not end

That is what the derivation rests on: one long run, with a rate. Given a
terminated step this drops the bootstrap, which is the only sensible thing to
do with it, and the result on an episodic task is an agent maximising reward
per step over the whole run rather than per episode. That is sometimes the
question and usually not.
"""

from __future__ import annotations

from collections.abc import Iterator

from rel.agents.base import TabularAgent, Transition
from rel.agents.explore import Rule
from rel.core import DIGEST_FIGURES, ObsT
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete


class DifferentialQ(TabularAgent[ObsT]):
    """Q-learning with the average reward subtracted and no discount.

    `discount` is not a setting on this agent. It is fixed at one and never
    read, because a differential agent has no discount, and offering one that
    did nothing would be a setting that quietly means nothing.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        step_size: float | Schedule = 0.1,
        average_step: float = 0.1,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=step_size,
            discount=1.0,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )

        if average_step < 0.0:
            raise ValueError("The rate of the average must not be below zero.")

        #: How fast the rate follows the error, as a fraction of `step_size`.
        #:
        #: Written relative to the step size rather than on its own, because
        #: what matters is that the rate moves more slowly than the values it
        #: is subtracted from. A rate that chased every error would be a
        #: second copy of the noise, subtracted from the whole table.
        self.average_step = average_step
        #: The reward per step this agent believes its policy collects.
        self.average = 0.0

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)

        row = self.values(transition.observation)
        offset = transition.action - self.actions.start

        ahead = (
            0.0
            if transition.terminated
            else self.best_value(transition.next_observation)
        )
        error = transition.reward - self.average + ahead - row[offset]

        step = self.current_step_size()
        row[offset] += step * error
        self.average += self.average_step * step * error

    def learned(self) -> Iterator[str]:
        """The table, and the rate.

        The rate belongs in the digest. Two of these agents holding the same
        table and different rates make different updates on the very next
        step, so they have not come to the same conclusion.
        """
        yield from super().learned()
        yield f"average|{self.average:.{DIGEST_FIGURES}g}"

    def __repr__(self) -> str:
        return (
            f"DifferentialQ(step_size={self.current_step_size():g}, "
            f"average_step={self.average_step:g}, "
            f"average={self.average:g}, {self.explore!r})"
        )


__all__ = ["DifferentialQ"]
