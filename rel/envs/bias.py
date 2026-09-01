"""Two states, one wrong turn, and a maximum over noise that makes it look
right.

Sutton and Barto, example 6.7. This is the smallest problem that separates
Q-learning from double Q-learning, and everything about it is arranged so that
nothing else can be blamed.

## The shape

    A   the start. One action goes left to B. Every other goes right and ends.
    B   every action ends, and pays a reward whose mean is below zero.

Nothing pays anything at A. Going right ends the episode with nothing, so it is
worth exactly zero. Going left is worth the mean of B's reward, which is -0.1.

**So going right is better, by a tenth, and the answer is not in doubt.** There
is no long horizon here, no approximation and no exploration problem. An agent
that goes left is wrong about arithmetic it has seen a hundred samples of.

## Why an agent goes left anyway

Q-learning backs up `max` over its own estimates of B's actions. Each of those
estimates is a sample mean of a noisy reward, so each carries error in both
directions, and the largest of ten such numbers is above the true largest
almost every time. The bias does not average out with more samples of one
action: it comes from taking the maximum, and taking the maximum of `n`
estimates is not an estimate of the maximum.

So `Q(A, left)` bootstraps a positive number out of a negative one, and left
looks better than right until the estimates at B tighten. That takes a great
many visits, and every one of those visits is an episode spent going the wrong
way.

Double Q-learning splits the choosing from the valuing: one table names B's
best action and the other says what it is worth. Neither grades its own choice,
so the error in the naming and the error in the valuing are independent and the
bias goes.

## The reward at B, and why it is not a normal

The literature draws B's reward from a normal with a mean of -0.1 and a spread
of 1. A normal has no exact model, and every environment in this project can be
solved by `rel.agents.dp` and is checked against its own model by a test that
drives it for thousands of steps.

So the reward here is `mean - spread` or `mean + spread`, evenly. **That is the
same mean and the same standard deviation as the normal in the book**, on two
points rather than on a line, and the model is four numbers. What the shape of
the distribution costs is the fine detail of how fast the bias fades, which is
not what the example is about.

## Why A has ten actions and the book gives it two

An action space is one shape for the whole environment, and `Discrete` cannot
say that A has two actions while B has ten. So A has as many as B, action 0
goes left, and every other action goes right.

That moves one reference number and no mechanism. Under an epsilon-greedy
policy that has learned the answer, the share of episodes that go left is
`epsilon / actions` rather than the book's `epsilon / 2`. The measurement reads
that share, so it works the number out rather than quoting 5 percent.

The nine right actions are worth exactly zero each, are deterministic, and tie.
An agent breaking that tie is choosing between nine ways of being right.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rel.core import NO_INFO, EnvSpec, Outcome, Step, TabularEnv
from rel.rng import Rng
from rel.spaces import Discrete

#: The state numbers. `END` is terminal and no action leaves it.
START, MIDDLE, END = 0, 1, 2

#: The one action at the start that goes the wrong way.
LEFT = 0


class MaximisationBias(TabularEnv):
    """The two state problem where a maximum over noise points the wrong way."""

    def __init__(
        self,
        rng: Rng,
        actions: int = 10,
        mean: float = -0.1,
        spread: float = 1.0,
    ) -> None:
        super().__init__(rng)

        if actions < 2:
            raise ValueError("The start needs a left and at least one right.")
        if spread <= 0.0:
            raise ValueError("With no spread there is no noise to maximise over.")
        if mean >= 0.0:
            raise ValueError("Going left has to be the wrong answer.")

        self.mean = mean
        self.spread = spread

        self.observation_space = Discrete(3)
        self.action_space = Discrete(actions)
        self.action_names = (
            "left",
            *(f"right {number}" for number in range(1, actions)),
        )
        self.spec = EnvSpec(
            name="bias",
            summary=(
                "Two states. Going left pays -0.1 on average and a maximum "
                "over noise says it pays more."
            ),
            max_episode_steps=10,
            suggested_discount=1.0,
        )

        self.at = START
        self._went_left = 0.0

    # -- What the reward does not say ---------------------------------------

    def audit(self) -> Mapping[str, float]:
        """Whether this episode went the wrong way, as a one or a nothing.

        Averaged over episodes this is the share of runs that went left, which
        is the number the whole example is about. The return says far less: an
        episode that went left is worth -1.1 or +0.9 and an episode that went
        right is worth 0, so a mean return mixes the mistake with the noise
        that caused it.
        """
        if not self._started and self.at == START:
            return NO_INFO
        return {"went_left": self._went_left}

    # -- The model ----------------------------------------------------------

    def transitions(self, state: int, action: int) -> Sequence[Outcome]:
        if state == MIDDLE:
            return (
                Outcome(0.5, END, self.mean - self.spread, True),
                Outcome(0.5, END, self.mean + self.spread, True),
            )
        if state == START and action == LEFT:
            return (Outcome(1.0, MIDDLE, 0.0, False),)
        return (Outcome(1.0, END, 0.0, True),)

    def start_states(self) -> Sequence[tuple[float, int]]:
        return ((1.0, START),)

    def terminal_states(self) -> frozenset[int]:
        """The end, and not the middle.

        The default reads the model and calls a state terminal when every
        branch out of it says so. That is right almost everywhere and wrong
        here: every action at B ends the episode, so the default would call B
        terminal, and a terminal state is worth nothing by definition. B is
        worth -0.1, and the whole example is about an agent that thinks
        otherwise. Value iteration would have agreed with the agent.
        """
        return frozenset({END})

    # -- Acting -------------------------------------------------------------

    def _reset(self) -> int:
        self.at = START
        self._went_left = 0.0
        return self.at

    def _step(self, action: int) -> Step[int]:
        if self.at == MIDDLE:
            # Every action here is the same gamble. That is the point: the ten
            # of them differ in the agent's estimates and in nothing else.
            reward = self.mean + (self.spread if self.rng.chance(0.5) else -self.spread)
            self.at = END
            return Step(END, reward, True, False)

        if action == LEFT:
            self._went_left = 1.0
            self.at = MIDDLE
            return Step(MIDDLE, 0.0, False, False)

        self.at = END
        return Step(END, 0.0, True, False)

    def render(self) -> str:
        where = {START: "A", MIDDLE: "B", END: "end"}[self.at]
        return f"at {where}, went left" if self._went_left else f"at {where}"

    def __repr__(self) -> str:
        return (
            f"MaximisationBias(actions={self.action_space.n}, "
            f"mean={self.mean:g}, spread={self.spread:g})"
        )


def maximisation_bias(rng: Rng, actions: int = 10) -> MaximisationBias:
    """The example as the book states it, with ten actions at B."""
    return MaximisationBias(rng, actions=actions)


__all__ = ["END", "LEFT", "MIDDLE", "START", "MaximisationBias", "maximisation_bias"]
