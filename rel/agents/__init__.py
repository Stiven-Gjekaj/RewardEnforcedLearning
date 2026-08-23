"""Every agent the command line can build, declared once.

Adding an agent means one entry here and one class. From the entry come the
name that the command line accepts, the line in `--list`, the settings that
`--set` will take, and the table in the documentation.

## The defaults here are not the defaults in the classes

A class carries the default that is right in general. An entry here carries the
default that is right for the environments in this project, which is a
different question and sometimes a different number.

A step size of 0.1 is the sensible general answer and it learns the cliff walk
slowly. The entries below use 0.5, which is what the chapter uses for that
figure. Both numbers are correct answers to their own question, and putting
them in two places is how each one stays readable.

## Why a builder takes the environment

An agent never sees the environment. A builder does, for one turn, to read two
things off it: how many actions there are, and, for an agent over a tile coder,
the range each observation covers.

Handing the agent the environment instead would let it read the reward, the
model and the audit, and an agent that can read the audit optimises the audit.
The builder is where that line is drawn.
"""

from __future__ import annotations

from typing import Any

from rel.agents.bandit import (
    EpsilonGreedyBandit,
    GradientBandit,
    OptimisticBandit,
    UpperConfidenceBandit,
)
from rel.agents.base import Agent, RandomAgent, TabularAgent, Transition
from rel.agents.dp import FixedPolicyAgent, value_iteration
from rel.agents.dyna import DynaQ, DynaQPlus
from rel.agents.features import encoder_for
from rel.agents.linear import SemiGradientQ, SemiGradientSarsa
from rel.agents.monte_carlo import MonteCarloControl
from rel.agents.policy import ActorCritic, Reinforce
from rel.agents.td import DoubleQ, ExpectedSarsa, NStepSarsa, QLearning, Sarsa
from rel.agents.tiles import TileCoder
from rel.core import Env, TabularEnv
from rel.registry import Entry, Registry
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Box

AgentBuilder = Any


def _tabular(cls: type[TabularAgent[Any]]) -> AgentBuilder:
    """A builder for an agent whose settings are the usual four."""

    def build(
        rng: Rng,
        env: Env[Any],
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> Agent[Any]:
        return cls(
            rng,
            env.action_space,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )

    return build


def _random(rng: Rng, env: Env[Any]) -> Agent[Any]:
    return RandomAgent(rng, env.action_space)


def _optimal(rng: Rng, env: Env[Any], discount: float = 1.0) -> Agent[Any]:
    """The best policy there is, worked out from the model and then played.

    This is the reference every other run is measured against. It learns
    nothing and it cannot be run on an environment that keeps no model.
    """
    if not isinstance(env, TabularEnv):
        raise TypeError(
            f"{env.spec.name} keeps no model, so the best possible policy "
            f"cannot be worked out. This agent needs an environment tagged "
            f"'tabular'."
        )
    return FixedPolicyAgent(
        rng, env.action_space, value_iteration(env, discount=discount).policy
    )


def _n_step(
    rng: Rng,
    env: Env[Any],
    n: int = 4,
    step_size: float | Schedule = 0.5,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    optimism: float = 0.0,
) -> Agent[Any]:
    return NStepSarsa(
        rng,
        env.action_space,
        n=n,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        optimism=optimism,
    )


def _monte_carlo(
    rng: Rng,
    env: Env[Any],
    step_size: float | Schedule | None = 0.1,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    optimism: float = 0.0,
) -> Agent[Any]:
    return MonteCarloControl(
        rng,
        env.action_space,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        optimism=optimism,
    )


def _dyna(cls: type[DynaQ[Any]]) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any],
        planning_steps: int = 20,
        step_size: float | Schedule = 0.5,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> Agent[Any]:
        return cls(
            rng,
            env.action_space,
            planning_steps=planning_steps,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )

    return build


def _dyna_plus(
    rng: Rng,
    env: Env[Any],
    kappa: float = 0.001,
    planning_steps: int = 20,
    step_size: float | Schedule = 0.5,
    discount: float = 0.95,
    epsilon: float | Schedule = 0.1,
) -> Agent[Any]:
    return DynaQPlus(
        rng,
        env.action_space,
        kappa=kappa,
        planning_steps=planning_steps,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
    )


def _policy_gradient(cls: type[Reinforce[Any]]) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any],
        hidden: int = 16,
        step_size: float = 0.02,
        value_step_size: float = 0.05,
        discount: float = 0.99,
        entropy: float = 0.01,
        normalise: bool = True,
        clip: float = 1.0,
    ) -> Agent[Any]:
        encoder, features = encoder_for(env.observation_space)
        return cls(
            rng,
            env.action_space,
            encoder,
            features,
            hidden=hidden,
            step_size=step_size,
            value_step_size=value_step_size,
            discount=discount,
            entropy=entropy,
            normalise=normalise,
            clip=clip,
        )

    return build


def _linear(cls: type[SemiGradientSarsa] | type[SemiGradientQ]) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any],
        bins: int = 8,
        grids: int = 8,
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.0,
        optimism: float = 0.0,
    ) -> Agent[Any]:
        box = getattr(env, "tiling_space", env.observation_space)
        if not isinstance(box, Box):
            raise TypeError(
                f"{env.spec.name} has a {type(box).__name__} observation, and "
                f"a tile coder divides a Box. This agent needs an environment "
                f"tagged 'continuous'."
            )
        return cls(
            rng,
            env.action_space,
            TileCoder(box, bins=bins, grids=grids),
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )

    return build


def _epsilon_greedy_bandit(
    rng: Rng,
    env: Env[Any],
    epsilon: float | Schedule = 0.1,
    step_size: float | Schedule | None = None,
    optimism: float = 0.0,
) -> Agent[Any]:
    return EpsilonGreedyBandit(
        rng, env.action_space, epsilon=epsilon, step_size=step_size, optimism=optimism
    )


def _optimistic_bandit(
    rng: Rng, env: Env[Any], optimism: float = 5.0, step_size: float | Schedule = 0.1
) -> Agent[Any]:
    return OptimisticBandit(
        rng, env.action_space, optimism=optimism, step_size=step_size
    )


def _upper_confidence(rng: Rng, env: Env[Any], confidence: float = 2.0) -> Agent[Any]:
    return UpperConfidenceBandit(rng, env.action_space, confidence=confidence)


def _gradient_bandit(
    rng: Rng,
    env: Env[Any],
    step_size: float | Schedule = 0.1,
    baseline: bool = True,
) -> Agent[Any]:
    return GradientBandit(rng, env.action_space, step_size=step_size, baseline=baseline)


AGENTS: Registry[Agent[Any]] = Registry(
    "agent",
    fixed=2,
    entries=[
        Entry(
            "random",
            "Chooses with equal probability and learns nothing.",
            _random,
            tags=("baseline",),
        ),
        Entry(
            "optimal",
            "Plays the best possible policy, worked out from the model.",
            _optimal,
            tags=("reference", "needs-model"),
        ),
        Entry(
            "q-learning",
            "Updates towards the best action available in the next state.",
            _tabular(QLearning),
            tags=("tabular", "off-policy"),
        ),
        Entry(
            "sarsa",
            "Updates towards the action the policy really takes next.",
            _tabular(Sarsa),
            tags=("tabular", "on-policy"),
        ),
        Entry(
            "expected-sarsa",
            "Updates towards the average over the policy, not a sample from it.",
            _tabular(ExpectedSarsa),
            tags=("tabular", "on-policy"),
        ),
        Entry(
            "double-q",
            "Two tables. One picks the best action, the other says what it is worth.",
            _tabular(DoubleQ),
            tags=("tabular", "off-policy"),
        ),
        Entry(
            "n-step-sarsa",
            "SARSA that waits n steps before deciding what a step was worth.",
            _n_step,
            tags=("tabular", "on-policy"),
        ),
        Entry(
            "monte-carlo",
            "Waits for the episode to end and credits every state with the rest.",
            _monte_carlo,
            tags=("tabular", "on-policy"),
        ),
        Entry(
            "dyna-q",
            "Q-learning that replays remembered steps between real ones.",
            _dyna(DynaQ),
            tags=("tabular", "planning", "off-policy"),
        ),
        Entry(
            "dyna-q-plus",
            "Dyna-Q that pays a bonus for a step it has not tried in a while.",
            _dyna_plus,
            tags=("tabular", "planning", "off-policy"),
        ),
        Entry(
            "tile-sarsa",
            "SARSA over a tile coder, for observations that are real numbers.",
            _linear(SemiGradientSarsa),
            tags=("linear", "on-policy"),
        ),
        Entry(
            "tile-q",
            "Q-learning over a tile coder.",
            _linear(SemiGradientQ),
            tags=("linear", "off-policy"),
        ),
        Entry(
            "reinforce",
            "Waits for the episode, then pushes up whatever led to a good return.",
            _policy_gradient(Reinforce),
            tags=("network", "on-policy"),
        ),
        Entry(
            "actor-critic",
            "Replaces the tail of the return with what a value network believes.",
            _policy_gradient(ActorCritic),
            tags=("network", "on-policy"),
        ),
        Entry(
            "bandit-greedy",
            "Pulls the best lever so far, and a random one now and then.",
            _epsilon_greedy_bandit,
            tags=("bandit",),
        ),
        Entry(
            "bandit-optimistic",
            "Expects too much of every lever and is disappointed into trying all.",
            _optimistic_bandit,
            tags=("bandit",),
        ),
        Entry(
            "bandit-ucb",
            "Pulls the lever with the highest value it could plausibly have.",
            _upper_confidence,
            tags=("bandit",),
        ),
        Entry(
            "bandit-gradient",
            "Keeps preferences rather than values, and pushes them.",
            _gradient_bandit,
            tags=("bandit",),
        ),
    ],
)

__all__ = [
    "AGENTS",
    "Agent",
    "RandomAgent",
    "TabularAgent",
    "Transition",
]
