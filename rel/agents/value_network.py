"""Q-learning with a network in place of the table, and the two pieces that
make that work.

A table has one number per state and action, and moving one never moves
another. A network has one set of weights for all of them, so every update
moves every estimate, and two things that are harmless with a table become the
reason a value network diverges.

## What goes wrong, and the two answers

**The steps come in correlated.** Twenty steps of a cart pole falling to the
left are twenty samples of nearly the same thing, and a network fitted to them
in a row forgets what it knew about falling to the right. `replay` is the
answer: keep the last few thousand steps and learn from a batch drawn out of
them, so one batch mixes experience from far apart in time.

**The target moves with the estimate.** Q-learning aims at
`r + discount * max Q(s')`, and with a network the same weights produce both
sides. Every step moves the thing being fitted and the thing it is fitted to,
which is a feedback loop rather than a regression. `target_refresh` is the
answer: keep a second copy of the weights, compute the target from that, and
refresh it every so often. The target then holds still between refreshes.

Both are settings rather than separate agents, so the four combinations are one
class and the difference between them is a number.
`scripts/measure_value_network.py` runs all four.

## Why the loss is squared error

Q-learning's update moves an estimate a fraction of the way towards a target.
Fitting a network to the same target with squared error and a gradient step
does the same thing to first order, and it is the form that has a gradient. The
step size then belongs to the optimiser rather than to the rule.

## What it costs

Every step is a forward pass for each member of the batch, plus one for the
target, plus a backward pass. That is `2 * batch + 1` passes through a network
per step of the environment, against one lookup in a dictionary.
`docs/algorithms.md` says what that comes to.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rel.agents.base import Agent, Transition
from rel.agents.features import Encoder
from rel.agents.replay import Replay
from rel.core import DIGEST_FIGURES, ObsT
from rel.nn.autograd import Tensor, add, scale, select, square
from rel.nn.layers import QNetwork
from rel.nn.optim import Adam
from rel.rng import Rng
from rel.schedules import Schedule, as_schedule


class DeepQ(Agent[ObsT]):
    """Q-learning over a network, with replay and a target network as settings."""

    def __init__(
        self,
        rng: Rng,
        actions: Any,
        encoder: Encoder,
        features: int,
        *,
        hidden: int = 16,
        step_size: float = 0.005,
        discount: float = 0.99,
        epsilon: float | Schedule = 0.1,
        replay: int = 2000,
        batch: int = 8,
        target_refresh: int = 200,
        clip: float = 1.0,
    ) -> None:
        super().__init__(rng, actions)

        if batch < 1:
            raise ValueError("A batch is at least one step.")
        if replay < 0:
            raise ValueError("A buffer holds no fewer than no steps.")
        if target_refresh < 0:
            raise ValueError("A refresh interval is not negative.")

        self.encoder = encoder
        self.discount = discount
        self.epsilon = as_schedule(epsilon)
        self.batch = batch
        self.target_refresh = target_refresh

        self.live = QNetwork(rng, features, actions.n, hidden)
        self.optimiser = Adam(self.live.parameters(), step_size, clip=clip)

        #: The second copy of the weights, or `None` when the target is taken
        #: from the live network. `target_refresh=0` is the ablation.
        self.target: QNetwork | None = None
        if target_refresh > 0:
            self.target = QNetwork(rng, features, actions.n, hidden)
            self.target.copy_from(self.live)

        #: The buffer, or `None` when the agent learns from the step it just
        #: took and nothing else. `replay=0` is the ablation.
        self.memory: Replay[ObsT] | None = None
        if replay > 0:
            self.memory = Replay(rng, replay)

        #: How many times the target network has been refreshed.
        self.refreshes = 0

    # -- Acting --------------------------------------------------------------

    def current_epsilon(self) -> float:
        return self.epsilon(self.episodes)

    def action_values(self, observation: ObsT) -> Sequence[float]:
        return self.live.values(self.encoder(observation))

    def act(self, observation: ObsT) -> int:
        if self.rng.chance(self.current_epsilon()):
            return self.actions.start + self.rng.below(self.actions.n)
        return self.greedy(observation)

    def greedy(self, observation: ObsT) -> int:
        row = self.action_values(observation)
        best = max(row)
        tied = [index for index, value in enumerate(row) if value == best]
        if len(tied) == 1:
            return self.actions.start + tied[0]
        return self.actions.start + tied[self.rng.below(len(tied))]

    def knows(self, observation: ObsT) -> bool:
        # A network has an opinion everywhere, whether or not it has been
        # anywhere near. Saying otherwise would draw a value map with holes in
        # it that the agent does not have.
        return True

    def learned(self) -> Any:
        for index, tensor in enumerate(self.live.parameters()):
            row = ",".join(f"{value:.{DIGEST_FIGURES}g}" for value in tensor.data)
            yield f"live.{index}|{row}"

    # -- Learning ------------------------------------------------------------

    def observe(self, transition: Transition[ObsT, int]) -> None:
        super().observe(transition)

        if self.memory is not None:
            self.memory.add(transition)
            batch = self.memory.sample(self.batch)
        else:
            # No buffer, so the only step to learn from is the one just taken.
            # Learning from it `batch` times would be the same gradient added
            # up, which is a larger step and not a larger batch.
            batch = [transition]

        if batch:
            self._fit(batch)

        if self.target is not None and self.steps % self.target_refresh == 0:
            self.target.copy_from(self.live)
            self.refreshes += 1

    def _fit(self, batch: Sequence[Transition[ObsT, int]]) -> None:
        loss: Tensor | None = None
        for transition in batch:
            predicted = select(
                self.live(self.encoder(transition.observation)),
                transition.action - self.actions.start,
            )
            wanted = Tensor([self._target_for(transition)])
            error = square(add(predicted, scale(wanted, -1.0)))
            loss = error if loss is None else add(loss, error)

        assert loss is not None
        self.optimiser.zero_grad()
        scale(loss, 1.0 / len(batch)).backward()
        self.optimiser.step()

    def _target_for(self, transition: Transition[ObsT, int]) -> float:
        """What this step says the action taken was worth.

        A terminated episode has no future. A step limit is not termination,
        so a truncated step keeps the value of where it stopped, which is the
        fault the whole project is careful about.

        The value of where it landed comes from the target network when there
        is one. That is the point of having one: between refreshes it does not
        move, so the thing being fitted is not chasing itself.
        """
        if transition.terminated:
            return transition.reward

        ahead = self.target if self.target is not None else self.live
        landed = ahead.values(self.encoder(transition.next_observation))
        return transition.reward + self.discount * max(landed)

    def __repr__(self) -> str:
        return (
            f"DeepQ(replay={len(self.memory) if self.memory else 0}, "
            f"batch={self.batch}, target_refresh={self.target_refresh}, "
            f"epsilon={self.current_epsilon():g})"
        )


__all__ = ["DeepQ"]
