"""Learning from whole episodes, once they are over.

A Monte Carlo agent waits. It plays the episode out, and then it knows what
every state it visited was really worth, because it saw the rest of the episode
happen. Nothing is bootstrapped and nothing is guessed, so nothing is biased.

The cost is variance and patience. One episode gives one sample of the return,
and on the cliff walk one unlucky fall makes that sample a hundred worse.
"""

from __future__ import annotations

from rel.agents.base import TabularAgent, Transition
from rel.agents.explore import Rule
from rel.core import ObsT
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete


class MonteCarloControl(TabularAgent[ObsT]):
    """First visit Monte Carlo control on an epsilon greedy policy.

    Each state and action pair is credited once per episode, with the return
    from the first time it was reached. Every visit credit is the alternative
    and it is not obviously worse, and first visit is what the chapter uses.

    ## The step size

    With `step_size` left as `None` the update is a running average: the tenth
    return counts for a tenth. That is the textbook rule and it is right for an
    environment that stays still.

    With a number, the update keeps a fixed weight on the newest return. That
    forgets, which is what an environment that changes needs.

    **It is not what control needs in a fixed environment, which this used to
    say.** The argument was that the policy keeps changing, so a return from
    three hundred episodes ago was collected by a policy that no longer exists.
    `scripts/measure_blackjack.py` scores what is learned against an exact
    answer at three budgets, and the argument holds only early:

        hands       running average   step 0.01   step 0.05
        50,000             -0.06258    -0.06557    -0.05994
        200,000            -0.05210    -0.05341    -0.05865
        500,000            -0.04841    -0.05227    -0.06007

    Perfect play is worth -0.04656. A fixed step wins at fifty thousand hands
    and loses at five hundred thousand, and the larger it is the worse it gets
    with more data, which is what a weight that never falls does. The default
    here stays a fixed 0.1 because the grids this project measures on are run
    for hundreds of episodes rather than hundreds of thousands, which is the
    end of that table where forgetting pays.

    ## A truncated episode has no return

    Monte Carlo needs the rest of the episode, and a step limit takes it away.
    Three things could be done and only one of them is honest about it.

    Dropping the episode loses everything on an environment that usually hits
    its limit. Treating the limit as an ending says the future is worth
    nothing, which is the same fault the whole project warns about. So the tail
    is filled in with what the agent currently believes the last state is
    worth, and this stops being pure Monte Carlo at that one point. The
    docstring says so rather than the code hiding it.
    """

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        *,
        step_size: float | Schedule | None = None,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
        explore: Rule | None = None,
    ) -> None:
        super().__init__(
            rng,
            actions,
            step_size=0.0 if step_size is None else step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
            explore=explore,
        )
        self.averaging = step_size is None
        self.visits: dict[tuple[ObsT, int], int] = {}
        self._episode: list[Transition[ObsT, int]] = []

    def observe(self, transition: Transition[ObsT, int]) -> None:
        super().observe(transition)
        self._episode.append(transition)

    def end_episode(self) -> None:
        if self._episode:
            self._learn()
            self._episode.clear()
        super().end_episode()

    def _learn(self) -> None:
        last = self._episode[-1]

        if last.terminated:
            total = 0.0
        else:
            # The tail the step limit took away, filled in with what the agent
            # believes. See the note in the class docstring.
            shares = self.policy_probabilities(last.next_observation)
            row = self.peek(last.next_observation)
            total = sum(share * value for share, value in zip(shares, row, strict=True))

        first_seen: dict[tuple[ObsT, int], int] = {}
        for index, transition in enumerate(self._episode):
            key = (transition.observation, transition.action)
            first_seen.setdefault(key, index)

        for index in range(len(self._episode) - 1, -1, -1):
            transition = self._episode[index]
            total = transition.reward + self.discount * total

            key = (transition.observation, transition.action)
            if first_seen[key] != index:
                continue

            row = self.values(transition.observation)
            offset = transition.action - self.actions.start

            if self.averaging:
                count = self.visits.get(key, 0) + 1
                self.visits[key] = count
                row[offset] += (total - row[offset]) / count
            else:
                row[offset] += self.current_step_size() * (total - row[offset])


__all__ = ["MonteCarloControl"]
