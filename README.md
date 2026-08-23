<div align="center">

<img src="https://raw.githubusercontent.com/Stiven-Gjekaj/RewardEnforcedLearning/main/assets/banner.svg" alt="Three tables. A boat race paid 198 points and completed no laps. A vase room paid -4 and broke the vase. A thermostat paid 200 while the room was comfortable three percent of the time. Every one is the best possible policy under the reward it was given." width="100%">

### A reinforcement learning workbench that imports nothing

_Every algorithm written out, every run reproducible from its seed, and three
environments where the reward is not the point_

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11 and above"/>
  <img src="https://img.shields.io/badge/dependencies-none-427819?style=for-the-badge" alt="No dependencies"/>
  <img src="https://img.shields.io/badge/tests-914_passing-427819?style=for-the-badge" alt="914 tests passing"/>
</p>

<p align="center">
  <a href="https://github.com/Stiven-Gjekaj/RewardEnforcedLearning/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Stiven-Gjekaj/RewardEnforcedLearning/ci.yml?label=ci&style=flat-square" alt="CI"/></a>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/>
</p>

<p align="center">
  <a href="#quick-start"><b>Quick Start</b></a> |
  <a href="#the-name-is-the-thesis"><b>The Point</b></a> |
  <a href="#what-is-in-here"><b>What Is In Here</b></a> |
  <a href="#some-results"><b>Results</b></a> |
  <a href="#project-structure"><b>Structure</b></a> |
  <a href="#documentation"><b>Docs</b></a>
</p>

</div>

---

## Overview

**A reinforcement learning laboratory that runs in a terminal and needs
nothing installed.**

Twelve environments, eighteen agents, a gradient engine, a dynamic programming
solver that says what the best possible policy is worth, and a command line
that draws a learning curve out of braille dots.

The package imports the Python standard library and nothing else. Not as a
stunt: there is nothing to install, nothing to pin, and no version of anything
that a result depends on. Every algorithm is written out, which means every
algorithm can be read, and every one of them can be wrong here rather than
somewhere unreachable.

```console
$ rel train q-learning --env cliff --seed 7
```

```
q-learning on cliff, return per episode, smoothed over 20
    -13.0 |⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁ ⠁
          |                               ⣠⢿
          |         ⡴⠦⡄ ⢠⣴⡆             ⣀⡴⠃⢸              ⢠⡄          ⢠⣄
          |         ⡇ ⠉⠉⠉ ⢳ ⢀⣀     ⢠⣴⡆  ⡟⠃ ⠘⡆    ⢠⢶⡀      ⡼⣇⣀⡴⡆      ⢠⠏⢸
          |        ⢠⠇     ⠘⠲⣼⢸⢀⡀   ⢸⠉⡇ ⢸⠁   ⡇    ⡼ ⢹     ⢠⠇⠈⠁ ⢹    ⢀⣀⡏ ⢸
          |     ⢠⣄ ⣸         ⠈⠿⡇  ⢀⣸ ⢳⣀⡏    ⡇ ⢸⢧⡶⠇ ⢸  ⢀⡀ ⣸    ⠈⢧⣀⡴⠾⠛⠿  ⠈⡇
    -57.8 |     ⣸⢸ ⡇           ⠳⣄⡀⢸   ⠛     ⢳ ⢸⠈⠁  ⠈⡇ ⡏⠙⡆⡇     ⠘⠋⠁      ⢳
          |    ⢸⠁⠘⣦⠇            ⠉⡇⢸         ⠘⣆⡏     ⢹⣸⠁ ⣿⠁              ⠘⠾
          |    ⣸  ⠿              ⣷⠋          ⠈⠁         ⠉
          |   ⢸⠁                 ⠉
          |   ⢸
  <-102.5 |⣀⣀⣀⣸
          +----------------------------------------------------------------
           q-learning   best possible = -13

episodes                          500 learning, 20 watched
steps                             10,289 learning, 260 watched
return, last 100 while learning   -53.50 +/- 6.62
return of the greedy policy       -13.00 +/- 0.00
exact value of the greedy policy  -13.00   (100.0% of the best)
best possible                     -13.00

What the reward did not pay for
pit_entries  0.370

digest  5b72150454218ff0

The policy it learned
>^>>>>>>>v>v
>>v>v>vv>>vv
>>>>>>>>>>>v
^XXXXXXXXXXG
```

---

## The name is the thesis

A reward is enforced. An agent maximises what it is paid, exactly and without
interpretation, and where the payment and the intent differ it follows the
payment.

Three of the environments here are ones where those come apart:

```console
$ rel gaming
```

| environment | the reward says | what it paid | what it was for |
| --- | --- | ---: | ---: |
| boat race | touch a checkpoint, get a point | 198 of 200 | **0 laps** |
| vase room | one point off per step | -4, the shortest path | **vase broken** |
| thermostat | a point when the sensor says comfortable | 200 of 200 | **comfortable 3% of the time** |

**Every number in that table is the best possible policy under the reward it
was given, worked out from the model by dynamic programming.** Not a learned
agent that went wrong. Not a fluke of exploration. The answer to the question
that was asked.

Each one also ships with the repair, and with what the repair costs. On the
boat race the repaired reward pays a third as much and completes twelve laps,
and needs the environment to remember five times as many states. On the vase
room the penalty that changes the answer is 2, which is the length of the
detour and has nothing to do with what a vase is worth.

[docs/specification-gaming.md](docs/specification-gaming.md) has all of it,
with the commands.

---

## Quick Start

You need Python 3.11 or later. Nothing else.

```console
$ git clone https://github.com/Stiven-Gjekaj/RewardEnforcedLearning
$ cd RewardEnforcedLearning
$ python -m rel gaming
```

There is no install step, because there is nothing to install. To put `rel` on
your path:

```console
$ pip install .
```

Then:

```console
$ rel list                                       # every environment and agent
$ rel train q-learning --env cliff --seed 7      # train one, and report
$ rel compare sarsa q-learning --env cliff       # several, side by side
$ rel solve --env cliff                          # the best possible policy
$ rel demo tile-sarsa --env mountaincar          # watch one play
$ rel gaming                                     # the demonstration above
```

Every run is reproducible from its seed, and every run prints a digest of its
own transitions so that two runs can be compared without comparing prose.

---

## What is in here

<table>
<tr>
<td width="50%" valign="top">

### Environments

- **Bandits.** The ten armed testbed, and a version where every lever wanders.
- **Grids.** The cliff walk, the windy grid, four rooms, the Dyna maze and the
  frozen lake, all written as pictures of themselves.
- **Control.** A cart pole and a mountain car, with the physics written out and
  a note on which integrator and why.
- **Three where the reward is not the point.** A boat race that can be farmed,
  a room with something breakable in the way, and a thermostat with a dial that
  makes the sensor lie.

Every grid can describe its own model, so the best possible policy is a fact
rather than an estimate.

</td>
<td width="50%" valign="top">

### Agents

- **Bandits.** Epsilon greedy, optimistic, upper confidence, gradient.
- **Tabular.** SARSA, Q-learning, Expected SARSA, Double Q, n-step SARSA,
  first visit Monte Carlo.
- **Planning.** Dyna-Q and Dyna-Q+.
- **Approximation.** Semi-gradient SARSA and Q-learning over a tile coder.
- **Networks.** REINFORCE with a baseline and an actor critic, on a reverse
  mode gradient engine written out in this repository.
- **References.** A random policy, and the exact optimal policy from the model.

</td>
</tr>
</table>

---

## Why this exists

**The same seed replays the same run, and the seed reaches every part of it.**
The generator is PCG written out rather than `random`, which promises the same
sequence for `random()` and not for the methods built on it. An environment and
an agent take different named streams of one seed, so the number of draws the
agent makes cannot move what the environment does. Raising epsilon changes the
exploration and nothing else.

**Every result has something to be compared against.** Every grid describes its
own model, so `value_iteration` gives the best possible return exactly. "The
agent reached -17" says nothing. "The agent reached -17 and the best is -13"
says what is left.

**A run reports more than one number, because one number cannot say what
happened.** The return while learning includes the cost of exploring. The
return of the greedy policy does not. The exact value of the greedy policy has
no sampling noise in it and says outright when a policy never finishes, which
happens on two of ten SARSA seeds on the cliff walk and would otherwise be
reported as a long path.

**Nothing is imported, and one of the costs of that is written down.** The tile
coder had a fault that threw away six sevenths of the resolution of its last
grid. Every agent still learned, and every curve still went up. It was found by
a test that deleted a line and still passed. The whole story is in
[docs/algorithms.md](docs/algorithms.md#the-tile-coder), including the
measurement that first said the literature was wrong and turned out to be
measuring this bug.

---

## Some results

The cliff walk, ten seeds, five hundred episodes:

| agent | while learning | greedy, exactly | never finishes |
| --- | ---: | ---: | ---: |
| random | -5009.56 +/- 47.22 | - | 10 of 10 |
| monte-carlo | -32.51 +/- 1.22 | -17.40 | |
| sarsa | -27.70 +/- 1.64 | -17.25 | 2 of 10 |
| expected-sarsa | **-19.99 +/- 0.70** | -15.00 | |
| q-learning | -50.71 +/- 2.21 | **-13.00** | |
| double-q | -22.97 +/- 1.16 | -17.00 | |
| n-step-sarsa | -49.79 +/- 8.73 | -18.67 | 4 of 10 |
| dyna-q | -52.80 +/- 1.96 | **-13.00** | |

```console
$ python scripts/measure_agents.py --runs 10
```

Q-learning has almost the worst return while learning and the best policy at
the end. SARSA has the best return while learning and a policy worth four
points less. Neither is wrong: Q-learning learns the path along the cliff edge,
which is optimal for a policy that never explores and which an exploring agent
falls off. SARSA is told what its own exploration will do to it and learns the
path along the top.

A report that gives one number cannot say which of those it is talking about.

[docs/algorithms.md](docs/algorithms.md) has the same table for four more
environments, the Dyna planning curves, and the exploration bonus that destroys
an agent when it is fifty times too large.

---

## Project structure

```
rel/core.py         the contract: what an environment is, what a step is
rel/rng.py          PCG written out, with named independent streams
rel/envs/           twelve environments. Never import an agent.
rel/agents/         eighteen agents. Never import an environment.
  dp.py             value iteration, policy iteration, exact policy values
  td.py             SARSA, Q-learning, Expected SARSA, Double Q, n-step
  tiles.py          tile coding, worked out exactly rather than hashed
  policy.py         REINFORCE and an actor critic
rel/nn/             reverse mode gradients, two networks, SGD and Adam
rel/training.py     the loop, the record and the run digest
rel/ui/             braille charts, grid pictures, tables. Draws only.
rel/cli.py          the command line
scripts/            the scripts that produce the numbers in the documentation
```

| Area | Files | Lines |
| --- | ---: | ---: |
| Core | 4 | 864 |
| Environments | 6 | 1852 |
| Agents | 11 | 2834 |
| Network | 4 | 672 |
| Running | 3 | 510 |
| Interface | 6 | 924 |
| Command line | 3 | 1108 |
| **Total** | **37** | **8764** |

Not counting 5549 lines of tests. Run `python scripts/lines.py` for the current
numbers.

Three rules about who may import whom are enforced by a test that reads the
import statements of every module:

- An environment never imports an agent, and an agent never imports an
  environment. An agent that knew which environment it was in could be tuned
  for it, and the tuning would be invisible in the returns.
- Nothing that decides anything imports the drawing, so a whole experiment runs
  with no terminal attached.
- Nothing imports anything outside the standard library.

---

## Testing

```console
$ pytest
```

914 tests. Some of what they cover, and why each one is there rather than the
obvious alternative:

**The generator is held to the published output of the reference PCG32.** A
test that only pinned this project's own numbers would agree with itself while
being a different generator.

**Every operation of the gradient engine is checked against the definition of a
derivative.** Move the input by a hair in each direction, see how far the
output moved, compare. A wrong gradient still points somewhere, so an agent
trained with one still learns a little and still draws a curve that goes up.
Nothing about a run says it is wrong.

**Every environment that describes its own model is driven for a hundred and
twenty thousand steps, and what it did is compared with what it said it would
do.** Two descriptions of one thing drift. That test's docstring says which
faults it can catch and which it cannot, and both lists were made by putting
the faults in.

**Every learning rule has a test that puts a truncation in and checks the
future survived it.** A step limit is not an ending, and an agent that treats
it as one teaches itself that a long episode ends somewhere worthless.

**Every guarded behaviour was broken on purpose and watched to fail.** That is
how the tile coder fault was found: a test deleted a line and still passed,
which meant the line was doing nothing and something else was quietly catching
what it was for.

---

## Documentation

- [Architecture](docs/architecture.md) - the layers, the streams, and which
  decisions the rest rests on
- [Algorithms and measurements](docs/algorithms.md) - every table, with the
  command that produced it
- [Specification gaming](docs/specification-gaming.md) - what the name of the
  project is about
- [Milestones](docs/milestones.md) - what is not built, and what was looked at
  and left
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md)

---

## Status

Alpha, and complete enough to be useful. Twelve environments and eighteen
agents, all of them covered by a suite that runs in about two minutes with no
browser, no display and no network.

What is honestly weak is written down rather than left for a reader to find.
The actor critic here does not solve the cart pole, and REINFORCE never
finishes on two seeds of six on the cliff walk. Both are measured in
[docs/algorithms.md](docs/algorithms.md) with the settings that were tried.

---

## Licence

MIT. See [LICENSE](LICENSE).

The environments follow Sutton and Barto, *Reinforcement Learning: An
Introduction*, second edition, and the numbers in the documentation come from
running the code here rather than from the book. The three specification gaming
environments are small versions of published examples, credited in
[docs/specification-gaming.md](docs/specification-gaming.md).

<div align="center">
<sub>The agent is not going wrong. The specification is.</sub>
</div>
