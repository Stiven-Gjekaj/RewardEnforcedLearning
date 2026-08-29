"""Q-learning whose choices are options rather than primitive actions.

An option lasts several steps, so the return collected while one runs covers
several rewards and the state it hands back in is several steps away. The
update is the same shape as Q-learning with that substituted in:

    Q(s, o) += step * (R + discount^k * max Q(s', o') - Q(s, o))

`R` is the discounted reward collected over the k steps the option ran, and
`s'` is where it stopped. At k of one this is Q-learning exactly, which is what
makes an agent holding only primitive options the agent it came from.

## The table is over options, not actions

`self.actions` here is the space of options and `self.moves` is the space of
primitive actions the environment takes. Everything inherited from
`TabularAgent` then works on the option table without changing: a row is one
number per option, and the exploring policy explores over options.

What is not inherited is `act` and `greedy`, because the environment takes a
primitive action. `act` runs the option it chose until the option stops.
`greedy` does not: it reports the action of the best option available here and
keeps no running state at all, because the renderer asks it about every cell of
a grid in any order, and an evaluation run that carried a half finished option
into an unrelated cell would report a policy nobody followed.

## Not every option can start everywhere

A hallway option covers one room. So the best value of a state is the best over
the options that can start there, and an option that cannot is not a choice
that was passed over. Reading the whole row instead would let an untouched
entry for an option belonging to another room win the maximum, and on a grid
where every step costs something that entry is the largest number in the table.
"""

from __future__ import annotations

from collections.abc import Sequence

from rel.agents.base import TabularAgent, Transition
from rel.options import Option
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Discrete


class OptionsQ(TabularAgent[int]):
    """Q-learning over a fixed set of options."""

    def __init__(
        self,
        rng: Rng,
        actions: Discrete,
        options: Sequence[Option],
        *,
        step_size: float | Schedule = 0.1,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> None:
        if not options:
            raise ValueError("An agent over options needs at least one option.")

        super().__init__(
            rng,
            Discrete(len(options)),
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )
        #: The primitive actions the environment takes. `self.actions` is the
        #: space of options, which is what the table is over.
        self.moves = actions
        self.options = list(options)

        #: The option running now, where it started, what it has collected and
        #: how long it has been going.
        self.running: int | None = None
        self.began_at = 0
        self.collected = 0.0
        self.length = 0

        #: How many options have run to the end, and how many steps they took
        #: between them. Their ratio is what says whether the agent is using
        #: the long options or only the primitive ones.
        self.finished = 0
        self.steps_in_options = 0
        #: How many times the learning rule has been applied to a cell.
        self.updates = 0
        #: How many of those choices were an option that lasts. A measurement
        #: that wants to know what the abstraction cost needs this and cannot
        #: get it from the mean length alone, because a run that never chose a
        #: long option and one that chose a one step long option look the same
        #: in that number.
        self.long_chosen = 0

    # -- Which options are on offer -----------------------------------------

    def available(self, observation: int) -> list[int]:
        """The options that can start here, by their place in the list."""
        return [
            index
            for index, option in enumerate(self.options)
            if option.can_start(observation)
        ]

    def best_option(self, observation: int) -> int | None:
        """The best option that can start here, ties broken by drawing."""
        choices = self.available(observation)
        if not choices:
            return None

        row = self.peek(observation)
        best = max(row[index] for index in choices)
        tied = [index for index in choices if row[index] == best]
        if len(tied) == 1:
            return tied[0]
        return tied[self.rng.below(len(tied))]

    def best_value(self, observation: int) -> float:
        """The best value over the options that can start here.

        A state no option can start in is worth nothing, which is the right
        answer for the goal cell and the only state that happens in.
        """
        choices = self.available(observation)
        if not choices:
            return 0.0
        row = self.peek(observation)
        return max(row[index] for index in choices)

    def action_values(self, observation: int) -> Sequence[float] | None:
        choices = self.available(observation)
        if not choices:
            return None
        row = self.peek(observation)
        return [row[index] for index in choices]

    # -- Acting --------------------------------------------------------------

    def act(self, observation: int) -> int:
        if self.running is None:
            self.running = self._choose(observation)
            self.began_at = observation
            self.collected = 0.0
            self.length = 0
            if not self.options[self.running].is_primitive:
                self.long_chosen += 1
        return self.options[self.running].act(observation)

    def _choose(self, observation: int) -> int:
        choices = self.available(observation)
        if not choices:
            raise ValueError(
                f"No option of this agent can start in state {observation}."
            )

        if self.rng.chance(self.current_epsilon()):
            return choices[self.rng.below(len(choices))]

        chosen = self.best_option(observation)
        assert chosen is not None
        return chosen

    def greedy(self, observation: int) -> int:
        """The action of the best option here, with no running state.

        This is one step of the best option rather than the option run to the
        end. An evaluation asks again at every step and gets the same answer
        while the option stays the best one, and the renderer can ask about any
        cell in any order without disturbing a run.
        """
        chosen = self.best_option(observation)
        if chosen is None:
            return self.moves.start
        return self.options[chosen].act(observation)

    def choice_lasts(self, observation: int) -> bool:
        chosen = self.best_option(observation)
        return chosen is not None and not self.options[chosen].is_primitive

    # -- Learning ------------------------------------------------------------

    def start_episode(self) -> None:
        self.running = None

    def observe(self, transition: Transition[int]) -> None:
        super().observe(transition)
        if self.running is None:
            return

        option = self.options[self.running]
        self.collected += self.discount**self.length * transition.reward
        self.length += 1

        self._learn_from_step(transition)

        if transition.done or option.stops_at(transition.next_observation):
            self._learn(transition)
            self.running = None

    def _learn_from_step(self, transition: Transition[int]) -> None:
        """Learn from the step just taken, before knowing where it leads.

        Nothing here. This agent waits for the option to stop and credits the
        state it started in, which is one update for however many steps the
        option ran. `IntraOptionQ` fills this in, and that is the whole
        difference between the two.
        """

    def _learn(self, transition: Transition[int]) -> None:
        """Learn from the option that has just stopped."""
        assert self.running is not None

        # A cut off episode is not a finished one. The state the option was
        # stopped in still has a future, and the option gets credit for it.
        target = self.collected
        if not transition.terminated:
            target += self.discount**self.length * self.best_value(
                transition.next_observation
            )

        row = self.values(self.began_at)
        row[self.running] += self.current_step_size() * (target - row[self.running])
        self._bump()

        self.finished += 1
        self.steps_in_options += self.length

    def _bump(self) -> None:
        """One more application of the learning rule to one cell.

        Counted rather than worked out from the episodes, because the two
        agents here make a different number of them per step and that
        difference is what the comparison between them is about.
        """
        self.updates += 1

    def end_episode(self) -> None:
        super().end_episode()
        self.running = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({len(self.options)} options, "
            f"step_size={self.current_step_size():g}, "
            f"epsilon={self.current_epsilon():g})"
        )


class IntraOptionQ(OptionsQ):
    """Q-learning over options that teaches every option agreeing with a step.

    `options-q` waits for an option to stop and credits the state it started
    in. Three steps inside one option move one cell, and the two states it
    passed through learn nothing about it. On four rooms that costs 2.57 return
    while learning, and `docs/algorithms.md` measures it.

    This is the standard answer. One real step is evidence about every option
    that would have taken that action there, whether or not that option was the
    one running, so every one of them is updated:

        Q(s, o) += step * (r + discount * U(s') - Q(s, o))

    where `U(s')` is what the option is worth from where it landed. That is the
    option's own value if it keeps going and the best value there if it stops,
    because an option that stops hands control back and what happens next is
    somebody else's choice.

    ## The collapse

    A primitive option stops after every step, so `U(s')` is always the best
    value and the rule is Q-learning exactly. An agent holding only primitive
    options is therefore Q-learning here as well, and is the same agent
    `options-q` is.

    ## What it costs

    More updates. `options-q` makes about one update for every step it takes
    and this makes one for every option that agrees with the step, which on
    four rooms is between one and three. Whether the extra updates pay for
    themselves is a measurement rather than an argument, and
    `scripts/measure_intra_option.py` is it.

    ## Why the option that ran is not treated differently

    It is not special. It is one of the options that agreed with the step, and
    it gets the same update they do. Adding the multi step update on top would
    credit its starting state twice for the same experience.
    """

    def _learn(self, transition: Transition[int]) -> None:
        """Nothing but the counters.

        The multi step update is replaced rather than added to. The option
        that ran is one of the options that agreed with each of its steps, and
        it has already been credited for every one of them.
        """
        self.finished += 1
        self.steps_in_options += self.length

    def _learn_from_step(self, transition: Transition[int]) -> None:
        step_size = self.current_step_size()

        for index, option in enumerate(self.options):
            if not option.would_take(transition.observation, transition.action):
                continue

            row = self.values(transition.observation)
            row[index] += step_size * (
                self._target(option, index, transition) - row[index]
            )
            self._bump()

    def _target(self, option: Option, index: int, transition: Transition[int]) -> float:
        """What this step says the option is worth, from where it landed.

        A terminated episode has no future at all. Otherwise it is the option's
        own value if the option keeps going, and the best value available if it
        stops, because an option that stops hands the choice back.

        A step limit is not a stopping rule. It stops the run and it does not
        stop the option, so the target here is the one the option's own rule
        gives and not the one a forced stop would give.
        """
        landed = transition.next_observation
        if transition.terminated:
            return transition.reward

        if option.stops_at(landed):
            ahead = self.best_value(landed)
        else:
            ahead = self.peek(landed)[index]
        return transition.reward + self.discount * ahead


__all__ = ["IntraOptionQ", "OptionsQ"]
