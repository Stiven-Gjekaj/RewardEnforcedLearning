"""Learning from one step, before the episode is over.

Every agent here updates a table towards a target built from the reward it just
received and what it already believes about where it landed. They differ in one
line: which value of the next state goes into the target.

    SARSA           the action the policy will really take
    Q-learning      the best action, whether the policy takes it or not
    Expected SARSA  the average over the policy, weighted by how likely each is
    Double Q        the best action by one table, valued by the other

That one line is the whole of the on-policy against off-policy difference, and
the cliff walk shows what it costs. SARSA is told what its own exploration will
do to it, so it learns the path along the top. Q-learning learns the path along
the cliff edge, which is the better path for a policy that never explores and
the worse one for the policy actually being run.

Neither is wrong. They answer different questions, and a run that does not say
which question it asked cannot be read.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from rel.agents.base import TabularAgent, Transition, rows_of
from rel.agents.explore import Rule
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete


class QLearning(TabularAgent[ObsT]):
    """Updates towards the best action available in the next state.

    Off-policy: the target ignores the exploration that the agent will really
    do. It learns the value of the greedy policy while following a policy that
    sometimes wanders, which is what makes it learn the cliff edge.
    """

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)

        row = self.values(transition.observation)
        index = transition.action - self.actions.start
        target = self.bootstrap(
            transition, self.best_value(transition.next_observation)
        )
        row[index] += self.current_step_size() * (target - row[index])


class Sarsa(TabularAgent[ObsT]):
    """Updates towards the action the policy really takes next.

    On-policy. The target needs the next action, and the next action is not
    known until the agent is asked for it.

    The obvious way round that is to choose it early, during the update, and
    hand the same one back when `act` is called. This agent does not do that,
    and the reason is worth writing down. A promise couples the agent to the
    loop: it is only correct while the loop takes the action it was given. A
    loop that evaluates greedily while learning, or that clips an action, or
    that has any policy of its own, quietly makes the agent update a cell that
    nothing visited. The first version of this file did promise, and an n step
    agent built the same way updated the wrong cell in a test that fed it a
    fixed sequence of actions.

    So the update waits one step instead. The agent holds the last transition,
    and when the next one arrives it carries the action that was really taken.
    Nothing has to be true about the loop for this to be right.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        self._held: Transition[ObsT] | None = None

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)

        held = self._held
        if held is not None:
            self._learn(held, transition.action)

        if transition.terminated:
            # There is no action after this one, and no value either.
            self._learn(transition, None)
            self._held = None
        elif transition.truncated:
            # The episode stops here and the state still has a future. Its
            # value is the value of what the policy would have done next, so
            # one action is drawn to stand for that and never taken.
            self._learn(transition, super().act(transition.next_observation))
            self._held = None
        else:
            self._held = transition

    def end_episode(self) -> None:
        held = self._held
        if held is not None:
            # The loop stopped without saying how. Treat the last state as
            # having no future, which is what a caller that stops early means.
            self._learn(held, None)
            self._held = None
        super().end_episode()

    def _learn(self, transition: Transition[ObsT], next_action: int | None) -> None:
        row = self.values(transition.observation)
        index = transition.action - self.actions.start

        if next_action is None:
            target = transition.reward
        else:
            offset = next_action - self.actions.start
            target = self.bootstrap(
                transition, self.peek(transition.next_observation)[offset]
            )

        row[index] += self.current_step_size() * (target - row[index])


class ExpectedSarsa(TabularAgent[ObsT]):
    """Updates towards the average over the policy, not a sample from it.

    This is SARSA with the sampling taken out of the target. The variance of
    the update falls, so a larger step size becomes usable, and on the cliff
    walk it reaches a good policy in fewer episodes than either neighbour.
    """

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)

        row = self.values(transition.observation)
        index = transition.action - self.actions.start

        probabilities = self.policy_probabilities(transition.next_observation)
        next_row = self.peek(transition.next_observation)
        expected = sum(
            share * value for share, value in zip(probabilities, next_row, strict=True)
        )

        target = self.bootstrap(transition, expected)
        row[index] += self.current_step_size() * (target - row[index])


class DoubleQ(TabularAgent[ObsT]):
    """Two tables. One picks the best action, the other says what it is worth.

    Q-learning takes a maximum over numbers that carry noise, and a maximum
    over noisy numbers is above the true maximum. The bias never averages out,
    because the same table both picks and values.

    Splitting the two removes it. A coin decides which table is updated, the
    other one supplies the value, and neither ever grades its own choice.

    The behaviour policy ranks actions by the sum of the two tables, which is
    the only place the split is put back together.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        self.q_other: dict[ObsT, list[float]] = {}

    def learned(self) -> Iterator[str]:
        # Both tables, and each says which it is. Run together, an agent that
        # had learned a thing in the first would match one that had learned it
        # in the second.
        yield from rows_of(self.q, "chooser.")
        yield from rows_of(self.q_other, "valuer.")

    def other_values(self, observation: ObsT) -> list[float]:
        row = self.q_other.get(observation)
        if row is None:
            row = [self.optimism] * self.actions.n
            self.q_other[observation] = row
        return row

    def peek_other(self, observation: ObsT) -> Sequence[float]:
        row = self.q_other.get(observation)
        if row is None:
            return [self.optimism] * self.actions.n
        return row

    def knows(self, observation: ObsT) -> bool:
        return observation in self.q or observation in self.q_other

    def action_scores(self, observation: ObsT) -> Sequence[float]:
        return [
            first + second
            for first, second in zip(
                self.peek(observation), self.peek_other(observation), strict=True
            )
        ]

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)

        # The row being written needs `values`, which makes one. The two rows
        # being read do not, and behind a writer they left a row for the goal
        # cell of the grid in a table the agent never acts from.
        if self.rng.chance(0.5):
            write_to = self.values
            read_chooser, read_valuer = self.peek, self.peek_other
        else:
            write_to = self.other_values
            read_chooser, read_valuer = self.peek_other, self.peek

        row = write_to(transition.observation)
        index = transition.action - self.actions.start

        best = self._argmax(read_chooser(transition.next_observation))
        target = self.bootstrap(
            transition, read_valuer(transition.next_observation)[best]
        )
        row[index] += self.current_step_size() * (target - row[index])


class NStepSarsa(TabularAgent[ObsT]):
    """SARSA that waits `n` steps before it decides what a step was worth.

    One step of SARSA carries the reward of one step and a guess about the
    rest. `n` steps carry `n` rewards and a guess about less. The guess is
    where the bias lives and the rewards are where the noise lives, so `n` is
    the dial between them, and the best setting is neither end.

    On a grid where only the goal pays, this is the difference between one
    episode teaching one cell and one episode teaching `n` of them.

    Like `Sarsa`, this waits for an action to be taken rather than choosing it
    early. The update for step `t` is made when the transition at step `t + n`
    arrives, which is the first moment the action at that step is a fact.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        n: int = 4,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        if n < 1:
            raise ValueError("An n step method needs n of one or more.")
        self.n = n

        self._states: list[ObsT] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._end: int | None = None
        self._next = 0

    def observe(self, transition: Transition[ObsT]) -> None:
        super().observe(transition)

        self._states.append(transition.observation)
        self._actions.append(transition.action)
        self._rewards.append(transition.reward)

        if transition.terminated:
            # Nothing follows, so no return has a tail to bootstrap from.
            self._end = len(self._rewards)
        elif transition.truncated:
            # The state at the limit has a future. One action is drawn from
            # the policy to value it, and never taken.
            self._states.append(transition.next_observation)
            self._actions.append(super().act(transition.next_observation))
            self._end = len(self._rewards)

        self._catch_up()

    def end_episode(self) -> None:
        if self._end is None and self._rewards:
            # The loop stopped without saying how. Close the episode off with
            # no tail, which is what a caller that stops early means.
            self._end = len(self._rewards)
        self._catch_up()
        self._reset_buffer()
        super().end_episode()

    def _reset_buffer(self) -> None:
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._end = None
        self._next = 0

    def _catch_up(self) -> None:
        """Make every update whose rewards and tail have now arrived."""
        while True:
            tau = self._next
            end = self._end

            if end is None:
                # The tail needs the action taken at step tau + n, which is
                # known only once that transition has arrived.
                if tau + self.n >= len(self._states):
                    return
            elif tau >= end:
                return

            self._update(tau, end)
            self._next += 1

    def _update(self, tau: int, end: int | None) -> None:
        last = tau + self.n if end is None else min(tau + self.n, end)

        total = 0.0
        for offset, index in enumerate(range(tau, last)):
            total += (self.discount**offset) * self._rewards[index]

        tail = tau + self.n
        if tail < len(self._states):
            total += (self.discount ** (tail - tau)) * self._value_at(tail)
        elif last < len(self._states):
            # Truncated inside this window. The tail is shorter than n and it
            # ends on the state the limit stopped in.
            total += (self.discount ** (last - tau)) * self._value_at(last)

        row = self.values(self._states[tau])
        index = self._actions[tau] - self.actions.start
        row[index] += self.current_step_size() * (total - row[index])

    def _value_at(self, index: int) -> float:
        # `peek` rather than `values`. The last entry of the buffer can be the
        # state a step limit stopped in, which the agent arrives at and never
        # chooses an action from, and a writer here would give it a row.
        offset = self._actions[index] - self.actions.start
        return self.peek(self._states[index])[offset]


__all__ = ["DoubleQ", "ExpectedSarsa", "NStepSarsa", "QLearning", "Sarsa"]
