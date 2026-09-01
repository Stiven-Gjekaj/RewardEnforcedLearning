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

The exploration of the agents over a tile coder is the clearest case. The
classes default to none at all, which is what the mountain car chapter uses
and what works there. Measured over five seeds, a tile coded SARSA with no
exploration reaches 157 on the cart pole and one with epsilon at 0.05 reaches
498 out of a possible 500, and on the mountain car the two are within two
points of each other. So the entry here says 0.05: a setting that solves one
environment and costs nothing on the other is a better default than one that
is right in the chapter it came from. `docs/algorithms.md` has the sweep.

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

from rel.agents.average import DifferentialQ
from rel.agents.bandit import (
    EpsilonGreedyBandit,
    GradientBandit,
    OptimisticBandit,
    UpperConfidenceBandit,
)
from rel.agents.base import Agent, RandomAgent, TabularAgent, Transition
from rel.agents.basis import RadialBasis
from rel.agents.dp import FixedPolicyAgent, value_iteration
from rel.agents.dyna import DynaQ, DynaQPlus
from rel.agents.explore import as_rule
from rel.agents.features import encoder_for
from rel.agents.fourier import FlatSteps, FourierBasis
from rel.agents.gaussian import GaussianActorCritic
from rel.agents.linear import SemiGradientQ, SemiGradientSarsa
from rel.agents.linear_prediction import (
    GradientTD,
    Policy,
    SemiGradientTD,
    fixed,
)
from rel.agents.lookup import Lookup, aggregated
from rel.agents.monte_carlo import MonteCarloControl
from rel.agents.off_policy import Estimator, OffPolicyMonteCarlo
from rel.agents.options import IntraOptionQ, OptionsQ
from rel.agents.policy import ActorCritic, Reinforce
from rel.agents.prediction import (
    MonteCarloPrediction,
    NStepTD,
    TDLambda,
    TemporalDifference,
)
from rel.agents.search import TreeSearch
from rel.agents.sweeping import PrioritisedSweeping
from rel.agents.td import DoubleQ, ExpectedSarsa, NStepSarsa, QLearning, Sarsa
from rel.agents.tiles import TileCoder
from rel.agents.traces import Kind, SarsaLambda, WatkinsQLambda
from rel.agents.tree import QSigma, Target, TreeBackup
from rel.agents.value_network import DeepQ
from rel.core import Env, TabularEnv
from rel.options import Option, hallway_options, primitives
from rel.registry import Entry, Registry
from rel.rng import Rng
from rel.schedules import Schedule
from rel.spaces import Box, Discrete

AgentBuilder = Any


def _tabular(cls: type[TabularAgent[Any]]) -> AgentBuilder:
    """A builder for an agent whose settings are the usual four."""

    def build(
        rng: Rng,
        env: Env[Any, Any],
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        explore: str = "epsilon-greedy",
        optimism: float = 0.0,
    ) -> Agent[Any, Any]:
        return cls(
            rng,
            _whole_numbers(env),
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            explore=as_rule(explore, epsilon),
            optimism=optimism,
        )

    return build


def _random(rng: Rng, env: Env[Any, Any]) -> Agent[Any, Any]:
    return RandomAgent(rng, _whole_numbers(env))


def _optimal(rng: Rng, env: Env[Any, Any], discount: float = 1.0) -> Agent[Any, Any]:
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
        rng, _whole_numbers(env), value_iteration(env, discount=discount).policy
    )


#: Two steps and a step size of 0.2, where every other tabular agent in the
#: registry uses 0.5. The two settings trade against each other and the sweep
#: in `docs/algorithms.md` shows it: an n step return propagates a value n
#: times faster and carries n times the spread. At 0.5 the table already moves
#: fast, so the spread is all cost and one step wins on all five grids. At 0.1
#: the one step agent has not carried the value far enough in five hundred
#: episodes and four steps wins instead.
#:
#: At n=2 and 0.2 the count of seeds whose policy never reaches the goal, over
#: thirty seeds of each of the five grids, falls from 47 to 10.
def _n_step(
    rng: Rng,
    env: Env[Any, Any],
    n: int = 2,
    step_size: float | Schedule = 0.2,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any, Any]:
    return NStepSarsa(
        rng,
        _whole_numbers(env),
        n=n,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
        optimism=optimism,
    )


#: A decay of 0.6 at a step size of 0.1, where the other tabular agents use a
#: step size of 0.5 and n-step SARSA uses 0.2. The three are the same finding
#: at three points: reaching further back buys propagation and costs spread,
#: and a large step size is already propagating fast enough that the spread is
#: all cost.
#:
#: Measured over four grids and thirty seeds each. Against the 0.8 and 0.2 this
#: started at, SARSA with traces is better on all four grids and leaves 1 policy
#: stuck rather than 10, and Watkins' Q is better on two and the same on two.
#: `docs/algorithms.md` has the sweep.
def _off_policy(
    rng: Rng,
    env: Env[Any, Any],
    estimator: Estimator = "weighted",
    step_size: float | Schedule | None = None,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any, Any]:
    return OffPolicyMonteCarlo(
        rng,
        _whole_numbers(env),
        estimator=estimator,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
        optimism=optimism,
    )


def _traced(cls: type[SarsaLambda[Any]]) -> AgentBuilder:
    """A builder for the two agents that keep traces."""

    def build(
        rng: Rng,
        env: Env[Any, Any],
        trace_decay: float = 0.6,
        traces: Kind = "replacing",
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        explore: str = "epsilon-greedy",
        optimism: float = 0.0,
    ) -> Agent[Any, Any]:
        return cls(
            rng,
            _whole_numbers(env),
            trace_decay=trace_decay,
            traces=traces,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            explore=as_rule(explore, epsilon),
            optimism=optimism,
        )

    return build


def _tree_backup(
    rng: Rng,
    env: Env[Any, Any],
    n: int = 3,
    target: Target = "greedy",
    step_size: float | Schedule = 0.2,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any, Any]:
    return TreeBackup(
        rng,
        _whole_numbers(env),
        n=n,
        target=target,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
        optimism=optimism,
    )


def _q_sigma(
    rng: Rng,
    env: Env[Any, Any],
    n: int = 3,
    sigma: float | Schedule = 0.5,
    target: Target = "greedy",
    step_size: float | Schedule = 0.2,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any, Any]:
    return QSigma(
        rng,
        _whole_numbers(env),
        n=n,
        sigma=sigma,
        target=target,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
        optimism=optimism,
    )


def _monte_carlo(
    rng: Rng,
    env: Env[Any, Any],
    step_size: float | Schedule | None = 0.1,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any, Any]:
    return MonteCarloControl(
        rng,
        _whole_numbers(env),
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
        optimism=optimism,
    )


def _dyna(cls: type[DynaQ[Any]]) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any, Any],
        planning_steps: int = 20,
        step_size: float | Schedule = 0.5,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        explore: str = "epsilon-greedy",
        optimism: float = 0.0,
    ) -> Agent[Any, Any]:
        return cls(
            rng,
            _whole_numbers(env),
            planning_steps=planning_steps,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            explore=as_rule(explore, epsilon),
            optimism=optimism,
        )

    return build


def _dyna_plus(
    rng: Rng,
    env: Env[Any, Any],
    kappa: float = 0.001,
    planning_steps: int = 20,
    step_size: float | Schedule = 0.5,
    discount: float = 0.95,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
) -> Agent[Any, Any]:
    return DynaQPlus(
        rng,
        _whole_numbers(env),
        kappa=kappa,
        planning_steps=planning_steps,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
    )


def _sweeping(
    rng: Rng,
    env: Env[Any, Any],
    planning_steps: int = 5,
    threshold: float = 1e-4,
    step_size: float | Schedule = 0.5,
    discount: float = 0.95,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
) -> Agent[Any, Any]:
    #: Five planning steps where the other two planners take twenty. Ordering
    #: the replays is what buys the reach, so the quota does not have to, and
    #: `docs/algorithms.md` counts what each of them costs in updates.
    return PrioritisedSweeping(
        rng,
        _whole_numbers(env),
        planning_steps=planning_steps,
        threshold=threshold,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
    )


def _doorways(env: TabularEnv) -> list[int]:
    """The doorways of a grid, asked for without naming what a grid is.

    A builder may read an environment and this file is the only one under
    `rel/agents/` allowed to know environments exist at all. It still may not
    import one, and `tests/test_layering.py` says so, so this asks whether the
    environment can describe its own doorways rather than checking its type.
    An environment that cannot has none.
    """
    describe = getattr(env, "gaps", None)
    return list(describe()) if callable(describe) else []


def _options(cls: type[OptionsQ]) -> AgentBuilder:
    """A builder for the two agents that choose options."""

    def build(
        rng: Rng,
        env: Env[Any, Any],
        hallways: bool = True,
        step_size: float | Schedule = 0.5,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> Agent[Any, Any]:
        return _built_options(
            cls, rng, env, hallways, step_size, discount, epsilon, optimism
        )

    return build


def _built_options(
    cls: type[OptionsQ],
    rng: Rng,
    env: Env[Any, Any],
    hallways: bool,
    step_size: float | Schedule,
    discount: float,
    epsilon: float | Schedule,
    optimism: float,
) -> Agent[Any, Any]:
    #: `hallways=False` leaves the agent with its primitive actions only, which
    #: is Q-learning. That is the comparison the whole method is measured by,
    #: so it is a setting rather than a second entry: both sides of it are then
    #: the same code and any difference is the options.
    if not isinstance(env, TabularEnv):
        raise TypeError(
            f"{env.spec.name} keeps no model, so its rooms cannot be worked "
            f"out. An agent over options needs an environment tagged 'tabular'."
        )

    built: list[Option] = list(
        primitives(_whole_numbers(env), range(env.observation_space.n))
    )
    if hallways:
        built.extend(hallway_options(env, _doorways(env)))

    return cls(
        rng,
        _whole_numbers(env),
        built,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        optimism=optimism,
    )


def _followed(
    env: Env[Any, Any], policy: str, discount: float
) -> dict[int, int] | None:
    """The fixed policy a predictor is handed.

    `uniform` is `None` rather than a mapping. A policy that draws cannot be
    written as one action per state, and it is the classic setting for the
    random walk, so the two are different kinds of thing rather than one kind
    with a special case.
    """
    if policy == "uniform":
        return None
    if policy != "optimal":
        raise ValueError(f"policy is 'uniform' or 'optimal'. {policy!r} is not.")

    if not isinstance(env, TabularEnv):
        raise TypeError(
            f"{env.spec.name} keeps no model, so the best policy cannot be "
            f"worked out. Use policy=uniform, or an environment tagged "
            f"'tabular'."
        )
    return dict(enumerate(value_iteration(env, discount=discount).policy))


def _td(
    rng: Rng,
    env: Env[Any, Any],
    policy: str = "uniform",
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any, Any]:
    return TemporalDifference(
        rng,
        _whole_numbers(env),
        _followed(env, policy, discount),
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _n_step_td(
    rng: Rng,
    env: Env[Any, Any],
    policy: str = "uniform",
    n: int = 3,
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any, Any]:
    return NStepTD(
        rng,
        _whole_numbers(env),
        _followed(env, policy, discount),
        n=n,
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _td_lambda(
    rng: Rng,
    env: Env[Any, Any],
    policy: str = "uniform",
    trace_decay: float = 0.8,
    traces: Kind = "replacing",
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any, Any]:
    return TDLambda(
        rng,
        _whole_numbers(env),
        _followed(env, policy, discount),
        trace_decay=trace_decay,
        traces=traces,
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _mc_prediction(
    rng: Rng,
    env: Env[Any, Any],
    policy: str = "uniform",
    first_visit: bool = True,
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any, Any]:
    return MonteCarloPrediction(
        rng,
        _whole_numbers(env),
        _followed(env, policy, discount),
        first_visit=first_visit,
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _differential_q(
    rng: Rng,
    env: Env[Any, Any],
    step_size: float | Schedule = 0.1,
    average_step: float = 0.1,
    epsilon: float | Schedule = 0.1,
    optimism: float = 0.0,
    explore: str = "epsilon-greedy",
) -> Agent[Any, Any]:
    #: No discount setting, and that is the whole point of the agent. It
    #: subtracts the rate it is collecting instead, so there is no number here
    #: to get wrong.
    return DifferentialQ(
        rng,
        _whole_numbers(env),
        step_size=step_size,
        average_step=average_step,
        epsilon=epsilon,
        optimism=optimism,
        explore=as_rule(explore, epsilon),
    )


def _tree_search(
    rng: Rng,
    env: Env[Any, Any],
    simulations: int = 50,
    depth: int = 60,
    confidence: float = 1.0,
    discount: float = 0.95,
    reuse: bool = True,
) -> Agent[Any, Any]:
    #: `reuse=False` is the ablation, and it is the one that says what the
    #: tree is doing. Off, the agent plans from nothing at every decision and
    #: carries away nothing at all. On, the work done for one state is there
    #: when it comes back. Both are the same code with one flag changed.
    if not isinstance(env, TabularEnv):
        raise TypeError(
            f"{env.spec.name} keeps no model, so there is nothing to search. "
            f"This agent needs an environment tagged 'tabular'."
        )
    return TreeSearch(
        rng,
        _whole_numbers(env),
        env,
        simulations=simulations,
        depth=depth,
        confidence=confidence,
        discount=discount,
        reuse=reuse,
    )


def _deep_q(
    rng: Rng,
    env: Env[Any, Any],
    hidden: int = 16,
    step_size: float = 0.01,
    discount: float = 0.99,
    epsilon: float | Schedule = 0.1,
    replay: int = 2000,
    batch: int = 8,
    target_refresh: int = 200,
    clip: float = 1.0,
    priority: float = 0.0,
    weighting: float = 0.0,
    double: bool = False,
) -> Agent[Any, Any]:
    #: `replay=0` and `target_refresh=0` are the two ablations, so the four
    #: combinations are settings of one agent rather than four entries. Both
    #: sides of a comparison are then the same code.
    #:
    #: `priority` and `weighting` are the same arrangement for the buffer: at
    #: zero it draws evenly, and raising `priority` alone is the mistake worth
    #: being able to run, because it is what makes the agent settle somewhere
    #: else. `scripts/measure_prioritised.py` runs all three.
    #:
    #: `double` is the third: the live network names the best action of the
    #: next state and the target network says what it is worth, so the maximum
    #: is not taken over the numbers it is then read from.
    #: `scripts/measure_double.py` measures it.
    encoder, features = encoder_for(env.observation_space)
    return DeepQ(
        rng,
        _whole_numbers(env),
        encoder,
        features,
        hidden=hidden,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        replay=replay,
        batch=batch,
        target_refresh=target_refresh,
        clip=clip,
        priority=priority,
        weighting=weighting,
        double=double,
    )


def _policy_gradient(cls: type[Reinforce[Any]]) -> AgentBuilder:
    #: The entropy bonus here is 0.05 and the classes default to less. The
    #: registry is where a default is chosen by measurement rather than by the
    #: chapter a method came from, which is the same reason the tile agents
    #: below carry an epsilon the mountain car chapter does not.
    #:
    #: At 0.01 REINFORCE loses three of twelve cliff walk seeds and reaches
    #: 356.5 of a possible 500 on the cart pole. At 0.05 it loses one seed and
    #: reaches 500 on all five. The cost is that the policies it does find are
    #: 0.51 blunter on the cliff walk. `docs/algorithms.md` has both tables.
    def build(
        rng: Rng,
        env: Env[Any, Any],
        hidden: int = 16,
        step_size: float = 0.02,
        value_step_size: float = 0.05,
        discount: float = 0.99,
        entropy: float = 0.05,
        normalise: bool = True,
        clip: float = 1.0,
    ) -> Agent[Any, Any]:
        encoder, features = encoder_for(env.observation_space)
        return cls(
            rng,
            _whole_numbers(env),
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


def _linear(
    cls: type[SemiGradientSarsa[Any]] | type[SemiGradientQ[Any]],
) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any, Any],
        bins: int = 8,
        grids: int = 8,
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.05,
        optimism: float = 0.0,
    ) -> Agent[Any, Any]:
        box = getattr(env, "tiling_space", env.observation_space)
        if not isinstance(box, Box):
            raise TypeError(
                f"{env.spec.name} has a {type(box).__name__} observation, and "
                f"a tile coder divides a Box. This agent needs an environment "
                f"tagged 'continuous'."
            )
        return cls(
            rng,
            _whole_numbers(env),
            TileCoder(box, bins=bins, grids=grids),
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )

    return build


def _a_box_of_actions(env: Env[Any, Any]) -> Box:
    """The environment's action space, when its actions are numbers.

    The other side of `_whole_numbers`. An agent whose policy is a
    distribution over a box needs the bounds of one, and an environment that
    hands out a short list of actions instead is refused here rather than
    where the bounds would be read.
    """
    space = env.action_space
    if not isinstance(space, Box):
        raise TypeError(
            f"{env.spec.name} takes an action from {space!r}, and this agent "
            f"aims at a point in a box. An environment whose actions are a "
            f"short list needs any of the other agents, or `pendulum-levels` "
            f"is that list cut from this box."
        )
    return space


def _gaussian(
    rng: Rng,
    env: Env[Any, Any],
    hidden: int = 32,
    step_size: float = 0.05,
    value_step_size: float = 0.05,
    spread_step_size: float = 0.01,
    discount: float = 0.99,
    spread: float = 0.5,
    entropy: float = 0.0,
) -> Agent[Any, Any]:
    encoder, features = encoder_for(env.observation_space)
    return GaussianActorCritic(
        rng,
        _a_box_of_actions(env),
        encoder,
        features,
        hidden=hidden,
        step_size=step_size,
        value_step_size=value_step_size,
        spread_step_size=spread_step_size,
        discount=discount,
        spread=spread,
        entropy=entropy,
    )


def _whole_numbers(env: Env[Any, Any]) -> Discrete:
    """The environment's action space, when its actions are whole numbers.

    Every agent here but the ones over a box of actions either ranks its
    actions or counts them, and neither is possible over a box. So the
    narrowing happens once, in the builder, where the message can name the
    environment and say what kind of agent it needs.
    """
    space = env.action_space
    if not isinstance(space, Discrete):
        raise TypeError(
            f"{env.spec.name} takes an action from {space!r}, and this agent "
            f"chooses between whole numbers. An environment whose actions are "
            f"a box needs an agent tagged 'continuous'."
        )
    return space


def _handed_features(env: Env[Any, Any], groups: int = 10) -> Lookup:
    """The table of features to predict this environment over.

    Two ways to get one, and the environment decides which.

    **It carries its own.** This is the third door of its kind, after
    `tiling_space` and the model a tabular environment writes out: a builder
    reads one thing off the environment for one turn and the agent never sees
    it. Baird's counterexample is the whole reason. Its features are what makes
    it a counterexample, and an agent that carried them would be an agent that
    knew which environment it was in.

    **It does not, and its states are whole numbers.** Then they are grouped,
    `groups` of them, which is the smallest approximation there is. `groups`
    does nothing on an environment that carries a table, because a table that
    is the point of the environment is not one to override with a setting.
    """
    rows = getattr(env, "feature_rows", None)
    if rows is None:
        space = env.observation_space
        if not isinstance(space, Discrete):
            raise TypeError(
                f"{env.spec.name} carries no table of features and its "
                f"observation is a {type(space).__name__} rather than a whole "
                f"number, so there is nothing to group. This agent needs "
                f"'baird', or an environment tagged 'tabular'."
            )
        return aggregated(space.n, groups)

    space = env.observation_space
    if isinstance(space, Discrete) and len(rows) != space.n:
        # A table one row short reaches an index error on whichever step first
        # lands in the state it has no row for, which can be thousands of
        # steps in and reads as a fault in the agent.
        raise ValueError(
            f"{env.spec.name} has {space.n} states and hands out "
            f"{len(rows)} rows of features. A lookup table needs one row "
            f"for each state."
        )
    return Lookup(rows)


def _handed_policies(env: Env[Any, Any]) -> tuple[Policy[int], Policy[int]]:
    """The policy that collects the data and the policy the question is about.

    An environment that says nothing about either is predicted under a uniform
    policy, on-policy, which is the ordinary case and the one where every
    importance ratio is one.
    """
    even = [1.0 / _whole_numbers(env).n] * _whole_numbers(env).n
    behaviour = tuple(getattr(env, "behaviour_shares", even))
    target = tuple(getattr(env, "target_shares", behaviour))
    return fixed(behaviour), fixed(target)


#: What a linear predictor here starts every state at.
#:
#: The classes start at nothing, which is the right general answer and is the
#: one number that demonstrates nothing on the only environment these can run
#: on. Baird's counterexample pays nothing, so nothing is the answer, so an
#: agent that starts there has no error to move on and never takes a step.
#: `rel train linear-td --env baird` printed a run of five hundred thousand
#: steps in which no weight moved.
#:
#: One puts every state at a value of one, because every row of that table
#: adds up to three and a weight of a third therefore makes a state worth one.
#: A table whose rows do not agree cannot answer this at all, and `Lookup`
#: says so rather than sharing out a number that is right for some states.
STARTS_AT = 1.0


def _linear_prediction(cls: type[SemiGradientTD[int]]) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any, Any],
        groups: int = 10,
        step_size: float | Schedule = 0.05,
        discount: float = 0.99,
        start_value: float = STARTS_AT,
    ) -> Agent[Any, Any]:
        behaviour, target = _handed_policies(env)
        return cls(
            rng,
            _whole_numbers(env),
            _handed_features(env, groups),
            behaviour,
            target,
            step_size=step_size,
            discount=discount,
            start_value=start_value,
        )

    return build


def _gradient_td(
    rng: Rng,
    env: Env[Any, Any],
    groups: int = 10,
    helper_step: float = 0.25,
    step_size: float | Schedule = 0.05,
    discount: float = 0.99,
    start_value: float = STARTS_AT,
) -> Agent[Any, Any]:
    behaviour, target = _handed_policies(env)
    return GradientTD(
        rng,
        _whole_numbers(env),
        _handed_features(env, groups),
        behaviour,
        target,
        helper_step=helper_step,
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _fourier(
    cls: type[SemiGradientSarsa[Any]] | type[SemiGradientQ[Any]],
) -> AgentBuilder:
    """SARSA or Q-learning over cosine waves.

    There is no `optimism` here, unlike its two neighbours. A Fourier basis
    has no single weight that makes every point worth the same, so the setting
    is left off rather than offered and refused: a reader who cannot find it
    learns what it would have meant, and one who sets it to a number that is
    then rejected learns only that something went wrong.
    """

    def build(
        rng: Rng,
        env: Env[Any, Any],
        order: int = 3,
        scaled_steps: bool = True,
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.05,
    ) -> Agent[Any, Any]:
        box = getattr(env, "tiling_space", env.observation_space)
        if not isinstance(box, Box):
            raise TypeError(
                f"{env.spec.name} has a {type(box).__name__} observation, and "
                f"a Fourier basis waves over a Box. This agent needs an "
                f"environment tagged 'continuous'."
            )
        return cls(
            rng,
            _whole_numbers(env),
            FourierBasis(box, order=order)
            if scaled_steps
            else FlatSteps(box, order=order),
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
        )

    return build


def _grouped(
    cls: type[SemiGradientSarsa[Any]] | type[SemiGradientQ[Any]],
) -> AgentBuilder:
    """SARSA or Q-learning over states grouped together.

    Every other coder in this project needs a `Box`. This one needs a whole
    number, which is what the tabular environments hand out, so it is the one
    approximator that can be run on the same environments as the tables.

    `groups` is a resolution dial with a table at one end. At `groups` equal to
    the number of states each state has its own weight, and the update is then
    the tabular update exactly: a one hot row makes `w[i] += a * d * x[i]` into
    `q[s] += a * d`. Below that, states share weights and the agent cannot tell
    them apart.

    That is what makes it worth having here. `scripts/measure_resolution.py`
    runs the gaming environments down this dial, and the question it asks is
    whether an agent that cannot see a place clearly can still find the way to
    game it.
    """

    def build(
        rng: Rng,
        env: Env[Any, Any],
        groups: int = 0,
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> Agent[Any, Any]:
        space = env.observation_space
        if not isinstance(space, Discrete):
            raise TypeError(
                f"{env.spec.name} has a {type(space).__name__} observation, "
                f"and states are grouped by their number. This agent needs an "
                f"environment tagged 'tabular'."
            )
        #: Zero means one group for each state, which is a table written as a
        #: coder. A number would have to be right for every environment at
        #: once, and they run from sixteen states to several thousand.
        return cls(
            rng,
            _whole_numbers(env),
            aggregated(space.n, space.n if groups <= 0 else groups),
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )

    return build


def _radial(
    cls: type[SemiGradientSarsa[Any]] | type[SemiGradientQ[Any]],
) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any, Any],
        bins: int = 6,
        width: float = 0.75,
        kept: int | None = None,
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.05,
        optimism: float = 0.0,
    ) -> Agent[Any, Any]:
        box = getattr(env, "tiling_space", env.observation_space)
        if not isinstance(box, Box):
            raise TypeError(
                f"{env.spec.name} has a {type(box).__name__} observation, and "
                f"a radial basis divides a Box. This agent needs an "
                f"environment tagged 'continuous'."
            )
        return cls(
            rng,
            _whole_numbers(env),
            RadialBasis(box, bins=bins, width=width, kept=kept),
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            optimism=optimism,
        )

    return build


def _epsilon_greedy_bandit(
    rng: Rng,
    env: Env[Any, Any],
    epsilon: float | Schedule = 0.1,
    step_size: float | Schedule | None = None,
    optimism: float = 0.0,
) -> Agent[Any, Any]:
    return EpsilonGreedyBandit(
        rng,
        _whole_numbers(env),
        epsilon=epsilon,
        step_size=step_size,
        optimism=optimism,
    )


def _optimistic_bandit(
    rng: Rng,
    env: Env[Any, Any],
    optimism: float = 5.0,
    step_size: float | Schedule = 0.1,
) -> Agent[Any, Any]:
    return OptimisticBandit(
        rng, _whole_numbers(env), optimism=optimism, step_size=step_size
    )


def _upper_confidence(
    rng: Rng, env: Env[Any, Any], confidence: float = 2.0
) -> Agent[Any, Any]:
    return UpperConfidenceBandit(rng, _whole_numbers(env), confidence=confidence)


def _gradient_bandit(
    rng: Rng,
    env: Env[Any, Any],
    step_size: float | Schedule = 0.1,
    baseline: bool = True,
) -> Agent[Any, Any]:
    return GradientBandit(
        rng, _whole_numbers(env), step_size=step_size, baseline=baseline
    )


AGENTS: Registry[Agent[Any, Any]] = Registry(
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
            "sarsa-lambda",
            "SARSA with a trace on every cell, which fades rather than stops.",
            _traced(SarsaLambda),
            tags=("tabular", "on-policy"),
        ),
        Entry(
            "q-lambda",
            "Q-learning with traces, cut whenever the agent leaves its policy.",
            _traced(WatkinsQLambda),
            tags=("tabular", "off-policy"),
        ),
        Entry(
            "tree-backup",
            "Off-policy n-step learning that never divides by a probability.",
            _tree_backup,
            tags=("tabular", "off-policy"),
        ),
        Entry(
            "off-policy-mc",
            "Learns the greedy policy from episodes an exploring one collected.",
            _off_policy,
            tags=("tabular", "off-policy"),
        ),
        Entry(
            "q-sigma",
            "Part sample and part expectation, with tree backup at one end.",
            _q_sigma,
            tags=("tabular", "off-policy"),
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
            "prioritised-sweeping",
            "Dyna that replays the step that matters rather than a random one.",
            _sweeping,
            tags=("tabular", "planning", "off-policy"),
        ),
        Entry(
            "differential-q",
            "Q-learning with no discount, which subtracts the rate it collects.",
            _differential_q,
            tags=("tabular", "off-policy", "average-reward"),
        ),
        Entry(
            "mcts",
            "Runs simulations from where it stands, then acts on what they said.",
            _tree_search,
            tags=("planning", "needs-model"),
        ),
        Entry(
            "options-q",
            "Q-learning that can choose to walk to a doorway and stop there.",
            _options(OptionsQ),
            tags=("tabular", "planning", "off-policy"),
        ),
        Entry(
            "intra-option-q",
            "Options, with every state an option passed through credited too.",
            _options(IntraOptionQ),
            tags=("tabular", "planning", "off-policy"),
        ),
        Entry(
            "td",
            "Estimates what a fixed policy is worth, one step at a time.",
            _td,
            tags=("tabular", "prediction", "on-policy"),
        ),
        Entry(
            "n-step-td",
            "Prediction that waits n steps before falling back on an estimate.",
            _n_step_td,
            tags=("tabular", "prediction", "on-policy"),
        ),
        Entry(
            "td-lambda",
            "Prediction with a trace on every state, which fades rather than stops.",
            _td_lambda,
            tags=("tabular", "prediction", "on-policy"),
        ),
        Entry(
            "mc-prediction",
            "Prediction that waits for the episode and uses the return itself.",
            _mc_prediction,
            tags=("tabular", "prediction", "on-policy"),
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
            "grouped-sarsa",
            "SARSA over states grouped together, which is a dial from a table "
            "to one number.",
            _grouped(SemiGradientSarsa),
            tags=("linear", "on-policy"),
        ),
        Entry(
            "grouped-q",
            "Q-learning over states grouped together.",
            _grouped(SemiGradientQ),
            tags=("linear", "off-policy"),
        ),
        Entry(
            "rbf-sarsa",
            "SARSA over radial basis features, which have no boundaries.",
            _radial(SemiGradientSarsa),
            tags=("linear", "on-policy"),
        ),
        Entry(
            "rbf-q",
            "Q-learning over radial basis features.",
            _radial(SemiGradientQ),
            tags=("linear", "off-policy"),
        ),
        Entry(
            "fourier-sarsa",
            "SARSA over cosine waves, which need an order and nothing else.",
            _fourier(SemiGradientSarsa),
            tags=("linear", "on-policy"),
        ),
        Entry(
            "fourier-q",
            "Q-learning over cosine waves.",
            _fourier(SemiGradientQ),
            tags=("linear", "off-policy"),
        ),
        Entry(
            "linear-td",
            "Prediction over features handed in, which off-policy can diverge.",
            _linear_prediction(SemiGradientTD),
            tags=("linear", "prediction", "off-policy"),
        ),
        Entry(
            "gradient-td",
            "The same, with the term that stops it diverging put back.",
            _gradient_td,
            tags=("linear", "prediction", "off-policy"),
        ),
        Entry(
            "gaussian-actor-critic",
            "Aims at a point in a box and learns how far to wander from it.",
            _gaussian,
            tags=("network", "on-policy", "continuous-actions"),
        ),
        Entry(
            "deep-q",
            "Q-learning over a network, with a replay buffer and a target copy.",
            _deep_q,
            tags=("network", "off-policy"),
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
