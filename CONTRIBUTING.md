# Contributing to RewardEnforcedLearning

Thank you for looking. This file says how to set the project up and what is
expected of a change.

## Set up

You need Python 3.11 or later. Nothing else.

```sh
git clone https://github.com/Stiven-Gjekaj/RewardEnforcedLearning
cd RewardEnforcedLearning
python -m rel train q-learning --env cliff --seed 7
```

The package imports nothing outside the standard library, so there is no
install step and no environment to make. The development tools are the only
things to fetch, and only if you want to run the gate below:

```sh
pip install -r requirements-dev.txt
```

## The gate

Run all of these before every commit. Not a selection.

```sh
ruff check .
ruff format --check .
mypy rel
pytest
```

`python scripts/verify.py` runs the four in that order and stops at the first
failure. Continuous integration runs the same four. A change that fails any of
them fails there as well.

## What a change looks like

- **One logical change per commit.** Code and its tests go in together.
  Documentation goes in its own commit. A wide rename goes in its own commit
  with no change of behaviour inside it.
- **Write the subject line in the present tense**, saying what the change does.
  No version numbers in a subject line.
- **No em-dashes, and no emoji**, in code, comments, documentation or commit
  messages.
- **Do not open a pull request unless somebody asks for one.**

## What earns trust here

- **Run it.** Do not conclude that it will learn.
- **Give the seed with the number.** Every result in the documentation names
  the command that produced it. A number with no command behind it is removed
  rather than checked.
- **When you add a test, break the thing it covers on purpose and watch it
  fail.** A test that cannot fail is worse than none, because it makes
  something look guarded while it drifts.
- **Compare against the random policy.** An agent that learns nothing still
  returns a number, and a learning curve that goes up can come from a change
  in the environment rather than from the agent.
- **Say plainly when a measurement does not support the conclusion.**

## Where the rules live

`rel/envs/` holds the environments. An environment never imports an agent.

`rel/agents/` holds the agents. An agent never imports an environment, and
never reads a field that the observation does not carry. An agent that knows
which environment it is in can cheat, and the cheat is invisible in the
returns.

`rel/ui/` draws. Nothing in `rel/envs/` or `rel/agents/` imports it. A whole
experiment runs with no terminal attached, which is what lets continuous
integration run the suite.

A test enforces each of these three rules by reading the imports of every
module. See `tests/test_layering.py`.
