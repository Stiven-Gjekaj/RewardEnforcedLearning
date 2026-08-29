"""A task with no ending, where the discount decides which policy is best.

Every grid in this project has a goal, so an episode ends and the return of a
policy is a finite number whatever the discount is. A task that never ends has
no such number. The reward keeps arriving for ever, and the only ways to
compare two policies are to discount the future or to take the average.

Those two are not the same comparison, and this environment is where they
disagree. It has one decision in it, made over and over.

## Two ways to be paid

From the junction, one action takes a short loop that pays a little and comes
straight back. The other takes a long loop that pays a lot and takes several
steps to come round.

    short   one step,  pays 1     ->  1 per step
    long    five steps, pays 10   ->  2 per step

**By reward per step the long loop is twice as good.** That is the answer any
person would give, and it is the answer the average reward formulation gives.

A discounted agent gives a different answer, and which one depends on the
discount. The value of the junction under each policy is

    short   p / (1 - d)
    long    d^(n-1) q / (1 - d^n)

so the two are equal when `1 + d + ... + d^(n-1)` equals `q / p` times
`d^(n-1)`. At the defaults that is a discount of about **0.74**. Below it a
discounted agent prefers the loop that pays half as much per step, and it is
right to: it is answering the question it was asked.

## The discount is part of the specification

That is the point of this environment being here rather than in a chapter about
average reward on its own. `docs/specification-gaming.md` is about a reward
that says something other than what was meant. A discount is the other half of
the same object, and it can say something other than what was meant in exactly
the same way, without anybody writing down a reward they would disown.

Nobody chooses 0.7 to mean "prefer the worse loop". They choose it because it
converges quickly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rel.core import EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete

SHORT, LONG = 0, 1


class TwoLoops(TabularEnv):
    """A junction between a short loop that pays a little and a long one.

    State 0 is the junction. States 1 to `length - 1` are the long way round,
    and the pay for it arrives on the step back into the junction.

    Nothing here ever ends. Every episode of it runs to the step limit, and the
    number to read is the reward per step rather than the return.
    """

    def __init__(
        self,
        rng: Rng,
        length: int = 5,
        short_pay: float = 1.0,
        long_pay: float = 10.0,
        steps: int = 500,
    ) -> None:
        super().__init__(rng)

        if length < 2:
            raise ValueError("The long loop takes at least two steps.")
        if short_pay <= 0.0 or long_pay <= 0.0:
            raise ValueError("Both loops pay something above zero.")

        self.length = length
        self.short_pay = short_pay
        self.long_pay = long_pay

        self.observation_space = Discrete(length)
        self.action_space = Discrete(2)
        self.spec = EnvSpec(
            name="loops",
            summary=(
                "A junction between a short loop that pays a little often and "
                "a long one that pays a lot rarely."
            ),
            max_episode_steps=steps,
            ends=False,
            suggested_discount=0.9,
        )

        self.at = 0
        self.long_laps = 0
        self.short_laps = 0

    # -- What each loop is worth --------------------------------------------

    def per_step(self, loop: int) -> float:
        """The reward per step of going round one loop for ever.

        This is what the two policies are worth, and neither number depends on
        a discount. It is the comparison an average reward agent makes.
        """
        if loop == SHORT:
            return self.short_pay
        return self.long_pay / self.length

    def crossover(self) -> float:
        """The discount at which a discounted agent changes its mind.

        Below this the short loop has the higher discounted value and above it
        the long one does. Found by bisection rather than in closed form: the
        equation is a polynomial of degree `length - 1` and its root is not
        worth a derivation nobody would check.
        """
        low, high = 0.0, 1.0 - 1e-12
        for _ in range(200):
            middle = (low + high) / 2.0
            if self._long_leads(middle):
                high = middle
            else:
                low = middle
        return (low + high) / 2.0

    def _long_leads(self, discount: float) -> bool:
        """Whether the long loop is worth more at this discount."""
        short = self.short_pay / (1.0 - discount)
        long_ = discount ** (self.length - 1) * self.long_pay
        long_ /= 1.0 - discount**self.length
        return long_ > short

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        self.at = 0
        self.long_laps = 0
        self.short_laps = 0
        return 0

    def _step(self, action: int) -> Step[int]:
        outcome = self.transitions(self.at, action)[0]
        if self.at == 0 and action == SHORT:
            self.short_laps += 1
        elif self.at == self.length - 1:
            self.long_laps += 1

        self.at = outcome.observation
        return Step(outcome.observation, outcome.reward, False, False)

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if state == 0:
            if action == SHORT:
                return (Outcome(1.0, 0, self.short_pay, False),)
            return (Outcome(1.0, 1, 0.0, False),)

        if state == self.length - 1:
            # The last step of the long way round, and the one that pays.
            return (Outcome(1.0, 0, self.long_pay, False),)

        return (Outcome(1.0, state + 1, 0.0, False),)

    def start_states(self) -> Sequence[tuple[float, int]]:
        return ((1.0, 0),)

    def audit(self) -> Mapping[str, float]:
        """How many times each loop was gone round.

        The reward already says which loop was taken, so this is not a hidden
        objective the way the audits of the gaming environments are. It is here
        because a count of laps reads more directly than a rate does.
        """
        return {
            "short_loops": float(self.short_laps),
            "long_loops": float(self.long_laps),
        }

    def render(self) -> str:
        cells = ["o"] * self.length
        cells[self.at] = "@"
        return "".join(cells)


def two_loops(rng: Rng, length: int = 5) -> TwoLoops:
    """The junction, with the long way round `length` steps long."""
    return TwoLoops(rng, length=length)


__all__ = ["LONG", "SHORT", "TwoLoops", "two_loops"]
