<div align="center">
  <a href="../README.md"><b>Reward Enforced Learning</b></a>
</div>

# Architecture

How the package is put together, and which decisions the rest of it rests on.

---

## The four layers

```
rel/core.py       the contract: what an environment is, what a step is
rel/envs/         environments. Never import an agent.
rel/agents/       agents. Never import an environment.
rel/nn/           a gradient engine and two networks. Knows nothing of rewards.
rel/training.py   the loop that puts an agent in an environment
rel/ui/           drawing. Nothing that decides anything imports this.
rel/cli.py        the command line
```

`tests/test_layering.py` reads the import statements of every module and fails
if any of those rules is broken. It parses the source rather than importing it,
so a module that would fail to import is still checked.

**Why an agent must not import an environment.** An agent that could tell a
cliff from a lake would be tuned for each, and every number in the
documentation would stop meaning what it says. The rule is not about tidiness.
It is what makes a comparison between two agents a comparison of two agents.

**Why nothing that decides imports the drawing.** A whole experiment runs with
no terminal attached. That is what lets continuous integration run the suite,
and it is what lets a result be produced by a machine with no display.

---

## One seed, many streams

`rel/rng.py` is PCG-XSH-RR, written out. `Rng(7).stream("env")` gives an
independent source from the same seed.

This is not a convenience. An environment and an agent both draw chance. If
they shared one source, the number of draws the agent makes would decide which
numbers the environment gets, and raising epsilon from 0.05 to 0.10 would
change the wind on the grid as well as the exploration. The comparison between
two settings would no longer be a comparison of one thing.

Each stream carries a different odd increment, so two sequences never meet, and
a stream depends on the seed and the name only. `Rng(7).stream("env")` is the
same source whatever the agent did.

The generator is written out rather than taken from `random`. Python promises
the same sequence from the same seed for `random()` and `getrandbits()`, and
not for the methods built on them: `sample` changed its draw pattern in 3.11.
A result that only reproduces on one minor version is not a reproducible
result.

`tests/test_rng.py` holds the generator to the published output of the
reference PCG32 implementation, so the claim that this is really PCG32 does not
rest on the code agreeing with itself.

---

## Terminated is not truncated

A step reports two different endings and an agent has to treat them
differently.

| | what it means | what the value of the next state is |
| --- | --- | --- |
| `terminated` | the episode ended inside the rules | zero. There is no future |
| `truncated` | a step limit stopped it from outside | whatever the agent believes. There is a future |

An agent that treats a step limit as an ending teaches itself that the state at
the limit is worthless, and then avoids the long path that would have won. On
the cart pole, where the limit is the best outcome the environment has, that
turns the correct answer into the one the agent avoids.

Every learning rule in this project goes through `TabularAgent.bootstrap` or
writes the same two lines out, and every one has a test that puts a truncation
in and checks that the future survived it.

---

## The base class does the stepping

`Env.step` is never overridden. It validates the action, calls `_step`, counts
the step and decides truncation. A subclass writes `_step` and never sees the
counter.

Two faults go away. An environment cannot forget to truncate, and an
environment cannot accept an action outside its space and do something quiet
with it. The second has a habit of turning into an agent that looks broken.

---

## An environment can describe itself

`TabularEnv` adds one method: `transitions(state, action)`, giving every branch
the action can take with its probability. That is enough for
`rel/agents/dp.py` to work out the best possible policy exactly.

That number is what makes every other claim checkable. "The agent reached -17"
says nothing on its own. "The agent reached -17 and the best possible is -13"
says what is left to gain, and it comes from the environment rather than from a
paper.

Writing the model out is extra work for an environment and it earns its keep
twice. Dynamic programming solves the environment, and
`tests/test_gridworld.py` drives the environment for a hundred and twenty
thousand steps and checks that what it did matches what it said it would do.
A model that has drifted from the code is worse than no model.

That test has a limit worth knowing: it cannot catch a fault inside a helper
that both the stepping and the model call, because a change there moves both
answers together. Its docstring says which faults it can catch and which it
cannot, and each of those was checked by putting the fault in.

---

## The audit

`Env.audit()` reports what an episode really did, whether or not the reward
paid for it: laps completed, vases broken, the share of the time the room was
comfortable.

An agent never sees it. That is the point rather than an oversight. An agent
that could read the audit would optimise it, and the gap this project is about
would close for the wrong reason.

The channel carries numbers only. Everything on it is averaged over episodes by
the recorder, and a string cannot be averaged.

---

## Two registries

`rel/envs/__init__.py` and `rel/agents/__init__.py` each hold one table.
Adding an environment means one entry and one module. From the entry come the
name the command line accepts, the line in `rel list`, the settings `--set`
takes, the group a benchmark runs, and the table in this documentation.

The settings are read off the builder's signature, so an environment that gains
a setting gains a command line option with no second place to edit.

The alternative is a decorator on each class. That reads well at the class and
badly everywhere else: nothing then holds the whole list, so the list has to be
found by importing every module and hoping none was missed.

**The defaults in a registry are not the defaults in the class.** A class
carries the default that is right in general. An entry carries the default that
is right for the environments in this project. A step size of 0.1 is the
sensible general answer and it learns the cliff walk slowly; the entries use
0.5, which is what the chapter uses for that figure.

---

## Nothing is imported

The package uses the standard library and nothing else. `pyproject.toml`
declares no dependencies and `tests/test_layering.py` reads every import
statement to hold it that way. A job in continuous integration builds an
interpreter holding the package and nothing else, and trains an agent in it,
because a test runs inside an interpreter that pytest has already filled.

What that buys: there is nothing to install, nothing to pin, and no version of
anything that a result depends on. What it costs: everything is written out, so
everything can be wrong here rather than somewhere else, and it runs at the
speed of Python.

Both halves of that are real. The tile coder had a fault that threw away most
of the resolution of most of its grids, and no library would have had it. It
was found by a test that removed a line and still passed. See
[algorithms.md](algorithms.md#the-tile-coder).

---

## What a run writes down

`rel/training.py` returns a `Record` holding the return, the length, the audit
and a digest of every episode.

The digest is a running hash of every transition. Two runs with the same digest
took the same path through the environment, step for step. It is exact for a
tabular environment and platform bound for a continuous one: the cart pole
calls `cos` and `sin`, a C library may return a result one bit different from
another, and five hundred steps of a swinging pole turn that bit into a visible
difference. So a digest from the cliff walk can be compared between two
machines, and a digest from the cart pole between two runs on one machine.

Saying which is which is worth more than either.

---

## Reading and writing are not the same

`TabularAgent.values` makes a row if there is not one. `TabularAgent.peek`
does not.

The renderer asks for the greedy action of every cell of a grid. With the
writing accessor behind that, drawing the picture put a row in the table for
every cell, including the eleven cliff cells the agent is never in, and the
value map then reported that the agent had stood in all of them and that they
were the best cells on the map.

`Agent.knows` is the other half. A value map has to tell a state worth nothing
from a state nothing is known about, and on a grid where every reward is
negative, an untouched entry of zero is the highest number in the table.

---

## Where to look

| For | Look at |
| --- | --- |
| The generator and its streams | [`rel/rng.py`](../rel/rng.py) |
| The contract | [`rel/core.py`](../rel/core.py) |
| A grid, and the presets built from one | [`rel/envs/gridworld.py`](../rel/envs/gridworld.py) |
| Solving exactly | [`rel/agents/dp.py`](../rel/agents/dp.py) |
| The one step learners | [`rel/agents/td.py`](../rel/agents/td.py) |
| The gradient engine | [`rel/nn/autograd.py`](../rel/nn/autograd.py) |
| The rules about layering | [`tests/test_layering.py`](../tests/test_layering.py) |
