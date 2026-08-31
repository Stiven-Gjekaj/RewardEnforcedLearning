"""Seven states, no reward anywhere, and an agent whose weights run away.

This is the smallest known demonstration that function approximation,
bootstrapping and off-policy learning are safe in any two together and not in
all three. The three together are called the deadly triad, and everything else
about this environment is arranged so that nothing else can be blamed.

    every reward is zero
    every state is worth zero under either policy
    the approximation can represent that exactly, with every weight zero

So the answer is nothing, the approximation can say nothing, and a method that
diverges here is not failing at a hard problem. It is failing at the easiest
problem there is.

## The shape

Six upper states and one lower state. Two actions, and neither one depends on
where the agent is standing.

    dashed  goes to one of the six upper states, evenly
    solid   goes to the lower state

The target policy takes solid every time, so under it the agent sits in the
lower state for ever. The behaviour policy takes dashed six times out of seven,
so the data is nearly all about the upper states, and that mismatch is the
off-policy leg of the triad.

## The features are the counterexample

Eight weights for seven states, laid out so the rows overlap:

    upper state i   worth 2 * w[i] + w[7]
    lower state     worth w[6] + 2 * w[7]

Every state shares `w[7]` with every other, and the lower state leans on it
twice as hard as any upper one. An update that lowers an upper state's estimate
pulls `w[7]` down, which lowers the lower state's estimate by twice as much,
which raises the error that the next update is trying to fix.

That is the whole mechanism. On-policy it does not happen, because the states
an update reaches are the states the data comes from and the pull comes back.
Off-policy the data comes from the upper states and the target is about the
lower one, and nothing comes back.

## Why the environment carries the features and the policies

An agent never sees an environment, and this file is the only place in the
project where that rule is bent, through the door `tiling_space` already
opened: a builder reads `feature_rows`, `behaviour_shares` and `target_shares`
off the environment for one turn, and hands the agent a coder and two policies.

The reason is that the counterexample is not the transitions. Those are four
lines and nothing about them is remarkable. The counterexample is the table of
features and the pair of policies, and if those lived in the agent then the
agent would be the one that knew which environment it was in.

## The starting weights

`STARTING_WEIGHTS` is the vector the figure in the literature starts from: ones
everywhere except a ten on the lower state's own feature. It is not the
environment's business and nothing here uses it, so it is a constant that
`scripts/measure_triad.py` and the tests read.

**The divergence does not need it.** It is a starting point that makes the run
away fast enough to read in a few hundred steps, and `docs/algorithms.md`
measures the same divergence from a start of all zeros but one.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.core import EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete

DASHED, SOLID = 0, 1

#: How often the behaviour policy takes each action. Six sevenths dashed, which
#: is what makes almost every step a step the target policy would not have
#: taken.
BEHAVIOUR: tuple[float, ...] = (6.0 / 7.0, 1.0 / 7.0)

#: The policy being evaluated. Solid every time, so its importance ratio is
#: seven on a solid step and zero on a dashed one.
TARGET: tuple[float, ...] = (0.0, 1.0)

#: The weights the well known figure starts from. See the module docstring:
#: nothing here uses it, and the divergence does not need it.
STARTING_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0)


class Baird(TabularEnv):
    """The counterexample, with the number of upper states left as a setting.

    Six is the number in the literature. Nothing about the mechanism needs it
    to be six, and a run at another number is a check that the divergence is
    the triad rather than an arithmetic coincidence of one layout.
    """

    def __init__(self, rng: Rng, upper: int = 6, steps: int = 1000) -> None:
        super().__init__(rng)

        if upper < 2:
            raise ValueError("The counterexample needs at least two upper states.")

        self.upper = upper
        self.lower = upper

        self.observation_space = Discrete(upper + 1)
        self.action_space = Discrete(2)
        self.spec = EnvSpec(
            name="baird",
            summary=(
                "Seven states that pay nothing, and features laid out so that "
                "off-policy bootstrapping over them diverges."
            ),
            max_episode_steps=steps,
            ends=False,
            suggested_discount=0.99,
        )

        self.at = 0

    # -- What makes it a counterexample -------------------------------------

    @property
    def feature_rows(self) -> tuple[tuple[float, ...], ...]:
        """One row of features for each state, which is the counterexample.

        Two features are on in every row and every row adds up to three, so one
        weight everywhere makes every state worth three times that weight.
        """
        width = self.upper + 2
        shared = width - 1

        rows: list[tuple[float, ...]] = []
        for state in range(self.upper):
            row = [0.0] * width
            row[state] = 2.0
            row[shared] = 1.0
            rows.append(tuple(row))

        below = [0.0] * width
        below[self.lower] = 1.0
        below[shared] = 2.0
        rows.append(tuple(below))

        return tuple(rows)

    def runs_away_above(self) -> float:
        """The discount above which semi-gradient TD diverges here.

        `(9 + n) / (5 + 2n)`, for `n` upper states. Six of them gives 15 over
        17, which is 0.882, and it is why the discount in the literature is
        0.99 rather than a round number: below the crossing the same three
        ingredients over the same features are stable.

        A number at or above one means no discount below one diverges, which
        is what four upper states or fewer gives.

        ## Where it comes from

        The expected update is `w <- w - a A w` with `A` the average over
        states of `x(s)` times `(x(s) - discount * x(lower))` transposed, since
        the target policy always goes to the lower state. It runs away when an
        eigenvalue of `A` has a negative real part.

        The six upper states are alike, so `A` splits. On the directions where
        their weights differ it is four sevenths times the identity, which is
        positive and is never the problem. What is left is three by three, in
        the weight every upper state shares, the lower state's own weight, and
        the weight all of them share. Its determinant is zero at every discount
        and that is the direction in which no state's value changes, so the
        other two eigenvalues decide, and they have positive real parts exactly
        when the trace and the sum of the principal minors are both positive.

            minors  (1 - discount) * (20 + n) / (1 + n) squared, always above
                    zero below a discount of one
            trace   (9 + n - (5 + 2n) * discount) / (1 + n)

        So the trace is the whole of it. `scripts/measure_triad.py` finds the
        same crossing by running the update rather than by this argument, and a
        test holds the two together.
        """
        return (9.0 + self.upper) / (5.0 + 2.0 * self.upper)

    @property
    def behaviour_shares(self) -> tuple[float, ...]:
        """How often the policy that collects the data takes each action."""
        return BEHAVIOUR

    @property
    def target_shares(self) -> tuple[float, ...]:
        """How often the policy being evaluated takes each action."""
        return TARGET

    # -- The contract -------------------------------------------------------

    def _reset(self) -> int:
        self.at = self.rng.below(self.upper + 1)
        return self.at

    def _step(self, action: int) -> Step[int]:
        if action == SOLID:
            self.at = self.lower
        else:
            self.at = self.rng.below(self.upper)
        return Step(self.at, 0.0, False, False)

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        """Where each action goes, which does not depend on where it is from.

        That is not a simplification of the counterexample. It is the
        counterexample: the transitions carry no structure at all, so nothing
        about them can be the reason a method fails here.
        """
        if action == SOLID:
            return (Outcome(1.0, self.lower, 0.0, False),)

        share = 1.0 / self.upper
        return tuple(Outcome(share, ahead, 0.0, False) for ahead in range(self.upper))

    def start_states(self) -> Sequence[tuple[float, int]]:
        """Evenly over all of them, upper and lower alike."""
        share = 1.0 / (self.upper + 1)
        return tuple((share, state) for state in range(self.upper + 1))

    def render(self) -> str:
        row = " ".join("@" if state == self.at else "o" for state in range(self.upper))
        stem = " " * (self.upper - 1)
        return f"{row}\n{stem}{'@' if self.at == self.lower else 'o'}"


def baird(rng: Rng, upper: int = 6) -> Baird:
    """The counterexample, with `upper` states above the one below."""
    return Baird(rng, upper=upper)


__all__ = [
    "BEHAVIOUR",
    "DASHED",
    "SOLID",
    "STARTING_WEIGHTS",
    "TARGET",
    "Baird",
    "baird",
]
