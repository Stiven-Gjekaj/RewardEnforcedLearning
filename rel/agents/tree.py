"""n-step tree backup, and the family that has it at one end.

`off-policy-mc` corrects for the difference between two policies by
multiplying by a ratio at every step, and `docs/algorithms.md` measures what
that costs: over ten seeds of the Dyna maze its worst estimate reads about four
trillion on a problem whose best possible value is 0.513.

This is the answer to that. It reaches back n steps, it learns about a policy
it is not following, and it never multiplies by a ratio.

## How it avoids one

An importance ratio is a correction applied after the fact to a sample of the
wrong thing. Tree backup never takes that sample. At every step of the window
it takes the expectation over what the target policy would have done, weighting
each action by its probability, and only the action that was really taken
carries the recursion any further.

So the actions the target policy might have taken and the behaviour policy did
not are still accounted for. They are accounted for by their current estimated
value rather than by a correction, which is why nothing has to be divided.

The cost is that the recursion is multiplied by the target policy's probability
of the taken action at every step. Where the two policies disagree often, that
probability is small and the reach is short. Tree backup does not escape the
problem so much as pay for it in a currency that cannot explode.

## Two collapses, both exact

At n of one the window is a single step and the recursion never runs, so the
target is the reward plus the discounted expectation over the target policy at
the next state.

With a greedy target that expectation is the largest value in the row, which is
Q-learning. With the agent's own exploring policy it is the average over that
policy, which is expected SARSA. Both are checked cell for cell rather than
approximately, because a method that did not collapse to its one step form
would be a different algorithm wearing the name.
"""

from __future__ import annotations

from typing import Literal

from rel.agents.explore import Rule
from rel.agents.td import NStepSarsa
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule
from rel.spaces import Discrete

Target = Literal["greedy", "policy"]
TARGETS: tuple[Target, ...] = ("greedy", "policy")


class TreeBackup(NStepSarsa[ObsT]):
    """n-step tree backup over a target policy that need not be the behaviour.

    The buffering is inherited. Deciding when an update can be made is the same
    question here as for any n step method: it waits until the window has
    arrived, and `NStepSarsa` answers it already.

    What is replaced is the target. Where that class sums n rewards and
    bootstraps once at the end, this takes an expectation at every step of the
    window.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        n: int = 3,
        target: Target = "greedy",
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            n=n,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        if target not in TARGETS:
            raise ValueError(f"target is one of {TARGETS}. {target!r} is not.")
        self.target = target

    def target_probabilities(self, observation: ObsT) -> list[float]:
        """What the policy being learned about would do here.

        `greedy` is the off-policy case: the agent explores and learns about
        the policy that does not. `policy` learns about the exploring policy
        itself, which is the on-policy case and the one that makes the collapse
        at n of one expected SARSA rather than Q-learning.
        """
        if self.target == "policy":
            return self.policy_probabilities(observation)

        best = self.greedy(observation) - self.actions.start
        shares = [0.0] * self.actions.n
        shares[best] = 1.0
        return shares

    def _expected(self, index: int) -> float:
        """The value of a state, averaged over what the target policy does."""
        observation = self._states[index]
        shares = self.target_probabilities(observation)
        row = self.peek(observation)
        return sum(share * value for share, value in zip(shares, row, strict=True))

    def _update(self, tau: int, end: int | None) -> None:
        horizon = tau + self.n if end is None else min(tau + self.n, end)

        # The base of the recursion is the last reward of the window, plus the
        # expectation at the state it landed in. A terminated episode has no
        # state to land in, and `horizon` is then past the end of the buffer.
        total = self._rewards[horizon - 1]
        if horizon < len(self._states):
            total += self.discount * self._expected(horizon)

        # Walk back through the window. At each step the actions the target
        # policy might have taken and this one did not contribute their value,
        # and the action really taken carries the rest of the return.
        for step in range(horizon - 2, tau - 1, -1):
            ahead = step + 1
            taken = self._actions[ahead] - self.actions.start
            shares = self.target_probabilities(self._states[ahead])
            row = self.peek(self._states[ahead])

            others = sum(
                share * value
                for offset, (share, value) in enumerate(zip(shares, row, strict=True))
                if offset != taken
            )
            total = self._rewards[step] + self.discount * (
                others + shares[taken] * total
            )

        row_now = self.values(self._states[tau])
        index = self._actions[tau] - self.actions.start
        row_now[index] += self.current_step_size() * (total - row_now[index])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={self.n}, target={self.target!r}, "
            f"step_size={self.current_step_size():g})"
        )


class QSigma(TreeBackup[ObsT]):
    """The family with tree backup at one end and sampling at the other.

    Sutton and Barto, section 7.5. Tree backup takes the expectation over the
    target policy at every step of the window, and n step SARSA takes the one
    action that was really taken and corrects for having taken it. Neither is
    obviously right, and `sigma` says how much of each to use:

        sigma of 0   the expectation, which is tree backup
        sigma of 1   the sample and its correction, which is n step SARSA
        in between   part of each, which is neither and is the point

    ## What sigma multiplies

    Every step of the window contributes

        reward + discount * (coefficient * (rest - Q(state, action)) + expected)

    where `expected` is the value of the state averaged over the target policy
    and `rest` is what the window says from there on. The coefficient is

        sigma * ratio + (1 - sigma) * share

    with `ratio` the importance ratio of the action taken and `share` the
    target policy's probability of it. At sigma of nothing the coefficient is
    the share and the terms rearrange into tree backup's.

    **The collapse is exact against a greedy target and the last bits against
    an averaged one.** Tree backup sums the actions that were not taken and
    adds the taken one separately; this sums all of them and subtracts the
    taken one back out. The two are the same expression rearranged, and float
    addition is not associative, so an averaged target leaves a gap of about
    3e-15 that does not grow with the run. A greedy target has shares of one
    and nothing, so nothing is rearranged and the two agree bit for bit.
    `tests/test_tree.py` checks both rather than describing them.

    ## Why the correction is subtracted rather than multiplied

    The form above is the one with a control variate: the ratio multiplies
    `rest - Q` rather than the whole return. Both are unbiased and this one
    has less variance, because the part of the return that the estimate
    already explains is not multiplied by a ratio that can be large.

    It also makes the family collapse cleanly. Without the control variate,
    sigma of nothing would be tree backup and sigma of one would be a
    different agent than the one this project already has, so neither end
    would be a check on the middle.

    ## Sigma can change

    A schedule is allowed and the book suggests one: start at one and fall
    towards nothing, so the early updates sample and the late ones average.
    Whether that is better than either end is measured in
    `scripts/measure_sigma.py` rather than assumed here.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        n: int = 3,
        sigma: float | Schedule = 0.5,
        target: Target = "greedy",
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            n=n,
            target=target,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        self.sigma = as_schedule(sigma)

    def current_sigma(self) -> float:
        """How much of the sample to use on this step, from nothing to all."""
        got = self.sigma(self.steps)
        if not 0.0 <= got <= 1.0:
            raise ValueError(f"sigma is a share of a sample. {got} is not one.")
        return got

    def _coefficient(self, observation: ObsT, taken: int, share: float) -> float:
        """What the gap between the window and the estimate is multiplied by.

        `sigma * ratio + (1 - sigma) * share`, where the ratio is how much
        more likely the target policy was to take this action than the
        behaviour policy was. The behaviour policy is asked only where sigma
        is above nothing, so a run at sigma of nothing does not depend on it
        at all.

        The ratio is nothing where the behaviour policy would never take the
        action. That cannot happen on an action it really took, and answering
        it beats dividing by nothing.
        """
        sigma = self.current_sigma()
        if sigma == 0.0:
            return share

        behaviour = self.policy_probabilities(observation)[taken]
        ratio = 0.0 if behaviour <= 0.0 else share / behaviour
        return sigma * ratio + (1.0 - sigma) * share

    def _update(self, tau: int, end: int | None) -> None:
        horizon = tau + self.n if end is None else min(tau + self.n, end)

        # The same base as tree backup, and for the same reason: the window
        # ends with one reward and the expectation at the state it landed in.
        # Every sigma shares it, because what sigma weights is the gap between
        # the rest of the window and an estimate that is the rest of the
        # window there.
        total = self._rewards[horizon - 1]
        if horizon < len(self._states):
            total += self.discount * self._expected(horizon)

        for step in range(horizon - 2, tau - 1, -1):
            ahead = step + 1
            observation = self._states[ahead]
            taken = self._actions[ahead] - self.actions.start

            # Read once and used three times. A greedy target breaks ties by
            # drawing, so asking twice both spends randomness and can answer
            # differently inside one update. The first version of this asked
            # three times and did not reproduce tree backup at sigma of
            # nothing, which is how the fault was found.
            shares = self.target_probabilities(observation)
            row = self.peek(observation)
            expected = sum(
                share * value for share, value in zip(shares, row, strict=True)
            )

            coefficient = self._coefficient(observation, taken, shares[taken])
            total = self._rewards[step] + self.discount * (
                coefficient * (total - row[taken]) + expected
            )

        row_now = self.values(self._states[tau])
        index = self._actions[tau] - self.actions.start
        row_now[index] += self.current_step_size() * (total - row_now[index])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={self.n}, sigma={self.current_sigma():g}, "
            f"target={self.target!r}, step_size={self.current_step_size():g})"
        )


__all__ = ["TARGETS", "QSigma", "Target", "TreeBackup"]
