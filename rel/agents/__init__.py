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
from rel.agents.linear import SemiGradientQ, SemiGradientSarsa
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
from rel.agents.tree import Target, TreeBackup
from rel.agents.value_network import DeepQ
from rel.core import Env, TabularEnv
from rel.options import Option, hallway_options, primitives
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
        explore: str = "epsilon-greedy",
        optimism: float = 0.0,
    ) -> Agent[Any]:
        return cls(
            rng,
            env.action_space,
            step_size=step_size,
            discount=discount,
            epsilon=epsilon,
            explore=as_rule(explore, epsilon),
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
    env: Env[Any],
    n: int = 2,
    step_size: float | Schedule = 0.2,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any]:
    return NStepSarsa(
        rng,
        env.action_space,
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
    env: Env[Any],
    estimator: Estimator = "weighted",
    step_size: float | Schedule | None = None,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any]:
    return OffPolicyMonteCarlo(
        rng,
        env.action_space,
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
        env: Env[Any],
        trace_decay: float = 0.6,
        traces: Kind = "replacing",
        step_size: float | Schedule = 0.1,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.1,
        explore: str = "epsilon-greedy",
        optimism: float = 0.0,
    ) -> Agent[Any]:
        return cls(
            rng,
            env.action_space,
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
    env: Env[Any],
    n: int = 3,
    target: Target = "greedy",
    step_size: float | Schedule = 0.2,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any]:
    return TreeBackup(
        rng,
        env.action_space,
        n=n,
        target=target,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
        optimism=optimism,
    )


def _monte_carlo(
    rng: Rng,
    env: Env[Any],
    step_size: float | Schedule | None = 0.1,
    discount: float = 1.0,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
    optimism: float = 0.0,
) -> Agent[Any]:
    return MonteCarloControl(
        rng,
        env.action_space,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
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
        explore: str = "epsilon-greedy",
        optimism: float = 0.0,
    ) -> Agent[Any]:
        return cls(
            rng,
            env.action_space,
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
    env: Env[Any],
    kappa: float = 0.001,
    planning_steps: int = 20,
    step_size: float | Schedule = 0.5,
    discount: float = 0.95,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
) -> Agent[Any]:
    return DynaQPlus(
        rng,
        env.action_space,
        kappa=kappa,
        planning_steps=planning_steps,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        explore=as_rule(explore, epsilon),
    )


def _sweeping(
    rng: Rng,
    env: Env[Any],
    planning_steps: int = 5,
    threshold: float = 1e-4,
    step_size: float | Schedule = 0.5,
    discount: float = 0.95,
    epsilon: float | Schedule = 0.1,
    explore: str = "epsilon-greedy",
) -> Agent[Any]:
    #: Five planning steps where the other two planners take twenty. Ordering
    #: the replays is what buys the reach, so the quota does not have to, and
    #: `docs/algorithms.md` counts what each of them costs in updates.
    return PrioritisedSweeping(
        rng,
        env.action_space,
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
        env: Env[Any],
        hallways: bool = True,
        step_size: float | Schedule = 0.5,
        discount: float = 0.95,
        epsilon: float | Schedule = 0.1,
        optimism: float = 0.0,
    ) -> Agent[Any]:
        return _built_options(
            cls, rng, env, hallways, step_size, discount, epsilon, optimism
        )

    return build


def _built_options(
    cls: type[OptionsQ],
    rng: Rng,
    env: Env[Any],
    hallways: bool,
    step_size: float | Schedule,
    discount: float,
    epsilon: float | Schedule,
    optimism: float,
) -> Agent[Any]:
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
        primitives(env.action_space, range(env.observation_space.n))
    )
    if hallways:
        built.extend(hallway_options(env, _doorways(env)))

    return cls(
        rng,
        env.action_space,
        built,
        step_size=step_size,
        discount=discount,
        epsilon=epsilon,
        optimism=optimism,
    )


def _followed(env: Env[Any], policy: str, discount: float) -> dict[int, int] | None:
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
    env: Env[Any],
    policy: str = "uniform",
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any]:
    return TemporalDifference(
        rng,
        env.action_space,
        _followed(env, policy, discount),
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _n_step_td(
    rng: Rng,
    env: Env[Any],
    policy: str = "uniform",
    n: int = 3,
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any]:
    return NStepTD(
        rng,
        env.action_space,
        _followed(env, policy, discount),
        n=n,
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _td_lambda(
    rng: Rng,
    env: Env[Any],
    policy: str = "uniform",
    trace_decay: float = 0.8,
    traces: Kind = "replacing",
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any]:
    return TDLambda(
        rng,
        env.action_space,
        _followed(env, policy, discount),
        trace_decay=trace_decay,
        traces=traces,
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _mc_prediction(
    rng: Rng,
    env: Env[Any],
    policy: str = "uniform",
    first_visit: bool = True,
    step_size: float | Schedule = 0.1,
    discount: float = 1.0,
    start_value: float = 0.0,
) -> Agent[Any]:
    return MonteCarloPrediction(
        rng,
        env.action_space,
        _followed(env, policy, discount),
        first_visit=first_visit,
        step_size=step_size,
        discount=discount,
        start_value=start_value,
    )


def _differential_q(
    rng: Rng,
    env: Env[Any],
    step_size: float | Schedule = 0.1,
    average_step: float = 0.1,
    epsilon: float | Schedule = 0.1,
    optimism: float = 0.0,
    explore: str = "epsilon-greedy",
) -> Agent[Any]:
    #: No discount setting, and that is the whole point of the agent. It
    #: subtracts the rate it is collecting instead, so there is no number here
    #: to get wrong.
    return DifferentialQ(
        rng,
        env.action_space,
        step_size=step_size,
        average_step=average_step,
        epsilon=epsilon,
        optimism=optimism,
        explore=as_rule(explore, epsilon),
    )


def _tree_search(
    rng: Rng,
    env: Env[Any],
    simulations: int = 50,
    depth: int = 60,
    confidence: float = 1.0,
    discount: float = 0.95,
    reuse: bool = True,
) -> Agent[Any]:
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
        env.action_space,
        env,
        simulations=simulations,
        depth=depth,
        confidence=confidence,
        discount=discount,
        reuse=reuse,
    )


def _deep_q(
    rng: Rng,
    env: Env[Any],
    hidden: int = 16,
    step_size: float = 0.01,
    discount: float = 0.99,
    epsilon: float | Schedule = 0.1,
    replay: int = 2000,
    batch: int = 8,
    target_refresh: int = 200,
    clip: float = 1.0,
) -> Agent[Any]:
    #: `replay=0` and `target_refresh=0` are the two ablations, so the four
    #: combinations are settings of one agent rather than four entries. Both
    #: sides of a comparison are then the same code.
    encoder, features = encoder_for(env.observation_space)
    return DeepQ(
        rng,
        env.action_space,
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
        env: Env[Any],
        hidden: int = 16,
        step_size: float = 0.02,
        value_step_size: float = 0.05,
        discount: float = 0.99,
        entropy: float = 0.05,
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
        epsilon: float | Schedule = 0.05,
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


def _radial(cls: type[SemiGradientSarsa] | type[SemiGradientQ]) -> AgentBuilder:
    def build(
        rng: Rng,
        env: Env[Any],
        bins: int = 6,
        width: float = 1.0,
        kept: int | None = None,
        step_size: float | Schedule = 0.5,
        discount: float = 1.0,
        epsilon: float | Schedule = 0.05,
        optimism: float = 0.0,
    ) -> Agent[Any]:
        box = getattr(env, "tiling_space", env.observation_space)
        if not isinstance(box, Box):
            raise TypeError(
                f"{env.spec.name} has a {type(box).__name__} observation, and "
                f"a radial basis divides a Box. This agent needs an "
                f"environment tagged 'continuous'."
            )
        return cls(
            rng,
            env.action_space,
            RadialBasis(box, bins=bins, width=width, kept=kept),
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
