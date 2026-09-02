<div align="center">
  <a href="../README.md"><b>Reward Enforced Learning</b></a>
</div>

# Algorithms, and what they actually do here

Every number on this page comes from a command in this repository, with the
seed next to it. Nothing here is remembered from a paper. Where a measurement
disagreed with what the literature says, it is written down that way and the
disagreement is investigated rather than rounded off.

---

## The cliff walk, ten seeds

```console
$ python scripts/measure_agents.py --runs 10
```

Four by twelve. Every step pays -1 and the cliff pays -100 and puts the agent
back at the start. The best possible return is **-13**: one step up, eleven
right, one down.

| agent | while learning | greedy, exactly | stuck |
| --- | ---: | ---: | ---: |
| random | -5009.56 +/- 47.22 | - | 10 |
| monte-carlo | -32.51 +/- 1.22 | -17.40 | |
| sarsa | -27.70 +/- 1.64 | -17.25 | 2 |
| expected-sarsa | **-19.99 +/- 0.70** | -15.00 | |
| q-learning | -50.71 +/- 2.21 | **-13.00** | |
| double-q | -22.97 +/- 1.16 | -17.00 | |
| n-step-sarsa | -23.13 +/- 1.35 | -16.80 | |
| dyna-q | -52.80 +/- 1.96 | **-13.00** | |
| dyna-q-plus | -47.56 +/- 1.91 | **-13.00** | |

*500 episodes, seeds 1 to 10, epsilon 0.1, and a step size of 0.5 for
every agent except `n-step-sarsa`, which the registry builds at 0.2. The
reason is measured in [how many steps, and how big a
step](#how-many-steps-and-how-big-a-step).*

Three numbers, because one of them cannot say what happened.

**While learning** is the mean return of the last hundred episodes, including
the cost of exploring. **Greedy, exactly** is the value of the greedy policy,
worked out from the model with no sampling noise in it. **Stuck** counts the
seeds whose greedy policy never reaches an ending at all.

### The two columns disagree, and that is the result

Q-learning has almost the worst return while learning and the best policy at
the end. SARSA and Expected SARSA have the best returns while learning and
policies worth four points less.

Neither is wrong. They answer different questions.

Q-learning's target is the best action available next, whether or not the
policy takes it. So it learns the value of the greedy policy while following
one that explores, and the greedy policy is the path along the cliff edge,
which is optimal and which an exploring agent falls off.

SARSA's target is the action the policy really takes next. It is therefore told
what its own exploration will do to it, and it learns the path along the top
row, which is two steps longer and survivable.

An agent that will be run greedily afterwards wants Q-learning. An agent whose
learning run is the run wants SARSA. A report that gives one number cannot say
which one it is talking about, which is why this project prints both.

### Expected SARSA is the best of both here

It is SARSA with the sampling taken out of the target: the average over the
policy instead of a draw from it. The variance of the update falls, and on this
grid it gets the best learning return of anything and a policy worth -15.

### The stuck column

Two of ten SARSA seeds produce a greedy policy that never finishes.

The mechanism is worth knowing because it looks like a bug and is not. On the
top row, moving up runs into the edge and the agent stays where it is. The four
actions at a cell there are within a point of each other, because a wasted step
costs one out of about eighteen, and the noise in the estimates is the same
size. When the wall bump wins by a hundredth, a greedy policy takes it, arrives
where it started, and takes it again for ever.

Q-learning never has this. Its policy runs along the cliff edge, where a wrong
action costs a hundred, and no amount of noise reorders that.

**So the agents that learn the safe path are the agents whose safe path can trap
them.** The step limit is what makes it visible instead of infinite, and a
report that averaged the truncated episodes in would call it a long path.

Four of ten n-step seeds sat in this column as well until its step size was
measured. This is why that setting mattered so much: the trap is the noise in
an estimate against a difference of one point, and a larger step size keeps
more noise in the estimate. Dropping it from 0.5 to 0.2 empties the column for
that agent on this grid, and [how many steps, and how big a
step](#how-many-steps-and-how-big-a-step) has the sweep.

---

## The same table, four more environments

```console
$ python scripts/measure_agents.py --env windy --runs 10 --episodes 600
$ python scripts/measure_agents.py --env rooms --runs 10 --episodes 600
$ python scripts/measure_agents.py --env maze --runs 10 --episodes 600
$ python scripts/measure_agents.py --env lake --runs 10 --episodes 600
```

One command for each of the four tables. This block showed only the first of
them for nineteen tracks, so three quarters of what follows named no command at
all, and `python scripts/check_numbers.py` reports a table nothing on the page
accounts for as exactly that. All four still print what is written under them.

**Windy grid**, best possible **-15**:

| agent | while learning | greedy, exactly | stuck |
| --- | ---: | ---: | ---: |
| random | -929.04 +/- 4.17 | - | 10 |
| monte-carlo | -26.28 +/- 1.60 | -18.60 | |
| sarsa | -19.69 +/- 0.24 | -16.17 | 4 |
| expected-sarsa | -17.26 +/- 0.07 | **-15.00** | |
| q-learning | -17.14 +/- 0.07 | **-15.00** | |
| double-q | -18.55 +/- 0.27 | -16.20 | |
| n-step-sarsa | -17.55 +/- 0.09 | **-15.00** | |
| dyna-q | **-17.12 +/- 0.05** | **-15.00** | |
| dyna-q-plus | -17.15 +/- 0.06 | **-15.00** | |

**Four rooms**, best possible **-20**:

| agent | while learning | greedy, exactly | stuck |
| --- | ---: | ---: | ---: |
| random | -724.13 +/- 5.93 | - | 10 |
| monte-carlo | -36.57 +/- 1.82 | -26.80 | |
| sarsa | -23.80 +/- 0.16 | -20.20 | |
| expected-sarsa | -22.73 +/- 0.09 | **-20.00** | |
| q-learning | -22.60 +/- 0.09 | **-20.00** | |
| double-q | -22.59 +/- 0.10 | **-20.00** | |
| n-step-sarsa | -22.84 +/- 0.11 | **-20.00** | |
| dyna-q | **-22.40 +/- 0.07** | **-20.00** | |
| dyna-q-plus | -22.41 +/- 0.07 | **-20.00** | |

**Dyna maze**, discount 0.95, best possible **0.51**:

| agent | while learning | greedy, exactly | stuck |
| --- | ---: | ---: | ---: |
| random | 0.92 +/- 0.01 | - | 10 |
| monte-carlo | 1.00 +/- 0.00 | 0.43 | 3 |
| sarsa | 0.99 +/- 0.00 | 0.48 | 3 |
| expected-sarsa | 1.00 +/- 0.00 | 0.49 | |
| q-learning | 1.00 +/- 0.00 | 0.48 | |
| double-q | 1.00 +/- 0.00 | **0.51** | |
| n-step-sarsa | 1.00 +/- 0.00 | 0.49 | 3 |
| dyna-q | 1.00 +/- 0.00 | 0.50 | |
| dyna-q-plus | 1.00 +/- 0.00 | 0.49 | 2 |

The two columns measure different things here and it is worth saying why. The
return is undiscounted, so on a maze where only the goal pays it is 1 if the
agent arrives at all. The exact column is discounted, so it says how quickly.
"Everything reached 1.00" and "the best is 0.51 and the field is between 0.43
and 0.51" are both true and only the second is informative.

**Frozen lake**, slippery, best possible **0.82**:

| agent | while learning | greedy, exactly | stuck |
| --- | ---: | ---: | ---: |
| random | 0.01 +/- 0.00 | 0.02 | |
| monte-carlo | 0.07 +/- 0.01 | 0.05 | |
| sarsa | 0.18 +/- 0.01 | 0.36 | |
| expected-sarsa | **0.26 +/- 0.01** | **0.60** | |
| q-learning | 0.26 +/- 0.03 | 0.57 | 2 |
| double-q | 0.08 +/- 0.02 | 0.15 | |
| n-step-sarsa | 0.20 +/- 0.03 | 0.40 | |
| dyna-q | 0.14 +/- 0.01 | 0.36 | |
| dyna-q-plus | 0.12 +/- 0.01 | 0.29 | 1 |

Nothing here gets near 0.82 in six hundred episodes, and the ones that do worst
are the ones whose assumptions the lake breaks. Dyna keeps one result for each
state and action, and on a slippery lake it remembers the last slip and replays
it as if it were certain. n-step SARSA carries several steps of a trajectory
that the slipping made unrepresentative, which is why it sat at 0.14 here
before its default was measured.

Neither half of that default explains the 0.40 on its own. Over thirty seeds,
dropping n from four to two is worth 0.061 and dropping the step size from 0.5
to 0.2 is worth 0.043, and the two together are worth 0.218. That is the
interaction in [how many steps, and how big a
step](#how-many-steps-and-how-big-a-step), seen from a third grid.

---

## How many steps, and how big a step

`n-step-sarsa` used to take four steps at a step size of 0.5. In the tables
above it got stuck more often than any other agent on four grids of five, and
on the cliff walk it also had the worst policy of anything that learned.
Nobody had swept it. Swept, on thirty seeds of each of the five grids:

```console
$ python scripts/measure_agents.py --env cliff --agents n-step-sarsa \
    --runs 30 --set n=4 step_size=0.5
```

<!-- not checked: a sweep of the command above over five grids and five
pairs of n and step size, so twenty five runs rather than one, and each
cell pairs a mean with a stuck count from the same run -->
| grid | n=4, 0.5 | n=2, 0.5 | n=1, 0.5 | n=4, 0.2 | n=2, 0.2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cliff, best -13 | -18.24 (9) | -17.84 (11) | -17.18 (8) | -17.71 (2) | **-16.80 (0)** |
| windy, best -15 | -17.86 (8) | -16.41 (8) | -15.96 (3) | -16.07 (1) | **-15.21 (1)** |
| rooms, best -20 | -21.14 (9) | -20.08 (6) | -20.00 (1) | -20.07 (1) | **-20.00 (0)** |
| lake, best 0.82 | 0.076 (1) | 0.137 (0) | **0.335** (0) | 0.119 (0) | 0.294 (0) |
| maze, best 0.51 | 0.460 (20) | 0.460 (18) | **0.486** (4) | 0.459 (13) | 0.473 (9) |

The bracket is how many of the thirty seeds ended with a policy that never
reaches the goal. Every other grid and setting is the same command with a
different `--env` and a different `--set`.

The two tables after this one are differences, and each cell is the value at
n=4 minus the value at n=1 from two runs of that command.

**At a step size of 0.5, one step wins on all five grids.** One step is plain
SARSA, so at the setting every other tabular agent here uses, the n step
return earns nothing at all.

### Two explanations, and the measurement refused both

The first guess was exploration. An n step return of an on-policy agent folds
n exploratory actions into one target, and with epsilon 0.1 and n=4 about a
third of the targets carry at least one. If that were the cause, less
exploration would shrink the cost of n. It grows:

<!-- not checked: every cell is one run minus another, and no command
prints a difference. The prose above says which two runs. -->
| epsilon | cliff | windy | rooms |
| ---: | ---: | ---: | ---: |
| 0.20 | -0.51 | -1.52 | -1.17 |
| 0.10 | -1.06 | -1.90 | -1.14 |
| 0.05 | -1.31 | -1.89 | -0.43 |
| 0.02 | **-1.54** | **-2.52** | -0.64 |

The second guess was the step size, on the ground that a four step target sums
four rewards and carries more spread, which a step size of 0.5 then writes
straight into the table. That one does not survive either, and the way it fails
is the answer:

<!-- not checked: every cell is one run minus another, and no command
prints a difference. The prose above says which two runs. -->
| step size | cliff | windy | rooms |
| ---: | ---: | ---: | ---: |
| 0.50 | -1.06 | -1.90 | -1.14 |
| 0.20 | -1.18 | -0.42 | -0.07 |
| 0.10 | -2.06 | **+2.75** | **+0.67** |
| 0.05 | -1.07 | **+3.04** | one step reaches nothing |

**The sign flips.** At a step size of 0.1 on the windy grid, four steps beats
one step by 2.75, and one step gets stuck on 23 of 30 seeds where four steps
gets stuck on none. On four rooms at 0.05, one step gets stuck on all thirty.

So the two settings are not independent and neither is a fault of the other.
An n step return carries a value n cells further per update and n times the
spread. A large step size is already moving the table fast, so the spread is
all cost. A small step size is not, and then the reach is worth the spread.

The registry default is n=2 at a step size of 0.2, which is the best of the
settings above on every grid where the n step return is used at all. Over the
five grids and thirty seeds each, the count of policies that never reach the
goal falls from 47 to 10.

### What that cost

The comparison tables above are a comparison of learning rules at one shared
setting, and this row is now the exception: every other agent runs at 0.5 and
this one at 0.2. That is stated in the caption rather than hidden, and the
like-for-like row is in the table above under `n=2, 0.5`.

The alternative was to leave the agent at a setting where it is beaten by
one-step SARSA on every grid in the project, so that one row of one table
stayed tidy. The measurement is more use than the tidiness.

---

## A trace is the same dial, done smoothly

`sarsa-lambda` and `q-lambda` keep a number on every cell they have visited
that says how much of the current error belongs to it. It falls by the discount
times the decay at every step, so credit reaches back for ever and fades rather
than stopping at a whole number of steps.

At a decay of zero the agent is the one step agent it was built on. That is
checked exactly rather than approximately: `SarsaLambda` at zero and `Sarsa`
fed the same transitions hold the same table cell for cell, and the same for
Watkins' Q against Q-learning.

### The same interaction, found from the other side

The section above says n and the step size trade against each other, and that
the sign of the trade flips. If a trace is that same idea done smoothly, the
flip has to be here too. It is.

```console
$ python scripts/measure_agents.py --env windy --agents sarsa-lambda \
    --runs 30 --episodes 600 --set trace_decay=0.8 step_size=0.1
```

What a decay of 0.8 is worth against a decay of zero, over thirty seeds:

<!-- not checked: every cell is one run minus another, and no command
prints a difference. The prose above says which two runs. -->
| step size | cliff | windy | rooms |
| ---: | ---: | ---: | ---: |
| 0.50 | -0.49 | -0.56 | -0.18 |
| 0.20 | -1.08 | -0.12 | 0.00 |
| 0.10 | -1.84 | **+1.97** | **+1.33** |
| 0.05 | -1.46 | see below | see below |

At a step size of 0.1 on the windy grid, a decay of zero leaves twenty seeds of
thirty with a policy that never reaches the goal, and a decay of 0.8 leaves
none. On four rooms the same two numbers are twenty seven and none. At a step
size of 0.05 a decay of zero never reaches the goal on **any** of the thirty
seeds of either grid, and a decay of 0.8 reaches the exact optimum on four
rooms with nothing stuck.

The cliff walk is the exception at every step size, and it was the exception in
the n step sweep as well.

**Two different ways of reaching further back, measured separately, give the
same interaction and the same exception.** That is worth more than either
measurement on its own. What reaching back buys is propagation, what it costs
is spread, and a large step size is already propagating fast enough that the
spread is all cost.

### The default, and the third point on one line

Four settings, four grids, thirty seeds each:

<!-- not checked: four settings over four grids at thirty seeds each, and
the stuck column adds the four grids together, so no single run prints a
number in this table -->
| decay | step size | sarsa-lambda, stuck of 120 | q-lambda, stuck of 120 |
| ---: | ---: | ---: | ---: |
| 0.8 | 0.2 | 10 | 0 |
| 0.8 | 0.1 | **0** | 0 |
| 0.0 | 0.2 | 9 | 1 |
| **0.6** | **0.1** | 1 | **0** |

The registry takes 0.6 at a step size of 0.1. It is better than the 0.8 and 0.2
this started at on all four grids for SARSA with traces, and on two of four for
Watkins' Q with the other two unchanged. The one row with nothing stuck at all
is 0.8 at 0.1, and it is not the pick: one stuck policy in 240 runs is not a
difference worth 0.87 of return on the cliff walk.

That makes three settings in this project on one line:

<!-- not checked: these are the registry defaults written side by side,
which is a statement about this project rather than a measurement -->
| method | how far back it reaches | step size |
| --- | --- | ---: |
| SARSA, Q-learning, the rest | one step | 0.5 |
| n-step SARSA | two steps | 0.2 |
| SARSA and Q with traces | a decay of 0.6 | 0.1 |

**The further back a method reaches, the smaller the step size it wants.** That
was not designed. Each of the three was swept on its own and the pattern is
what the three answers turned out to have in common.

### Watkins' Q almost never leaves a policy stuck

Over six decays, four grids and thirty seeds, which is 720 runs of each agent:

<!-- not checked: 720 runs of each agent added together, over six decays,
four grids and thirty seeds -->
| agent | policies that never reach the goal |
| --- | ---: |
| `sarsa-lambda` | 59 of 720 |
| `q-lambda` | **1 of 720** |

That is the stuck column above, seen from a new direction. The trap is a cell
where the four actions sit within a point of each other and the noise decides.
Q-learning's policy runs where a wrong action costs a hundred, and no amount of
noise reorders that. Cutting the trace on every exploratory step keeps that
property rather than blurring it away.

---

## Four ways of exploring, and the one thing that separates them

```console
$ python scripts/measure_exploration.py
$ python scripts/measure_exploration.py --env maze
$ python scripts/measure_exploration.py --rules count-bonus:0.1,count-bonus:2
$ rel sweep q-learning --env corridor --set discount=0.95 --over explore=epsilon-greedy,count-bonus:0.5
```

Every agent above explores the same way. It takes the best action, and with
probability epsilon it takes any action instead. That rule ignores everything
the agent knows: a move it has made a thousand times and a move it has never
tried are equally likely to be the one it tries.

`explore` is a setting on every tabular agent, and three rules answer it.

<!-- not checked: a table of what the three rules do. The only number in
it is part of a setting's name -->
| `--set explore=` | ranks actions by |
| --- | --- |
| `epsilon-greedy` | what they are worth, with a fixed chance of ignoring that |
| `softmax:0.02` | what they are worth, smoothly |
| `count-bonus:0.5` | what they are worth, plus a term that shrinks as an action is taken |

There is a fourth way and it is not a rule. **Optimistic initialisation** is a
starting value. `--set optimism=1` makes every action of a state nobody has
visited worth the best outcome there is, so a greedy agent walks towards
whatever it has not tried and stops as soon as the numbers come down to the
truth. It answers the same question in the other place, and a comparison that
left it out would be a comparison of three answers to a question that has four.

### On the grids this project already had, all four are the same

```console
$ python scripts/measure_exploration.py --env cliff
$ python scripts/measure_exploration.py --env rooms
$ python scripts/measure_exploration.py --env maze
```

Ten seeds and three hundred episodes of `q-learning`, each grid at its own
suggested discount. The four numbers in each cell are the four rules in the
order above, with optimism last. The three runs are one table here and three
tables on a terminal.

| grid | first episode that reached the goal | settled | value of the policy found |
| --- | --- | ---: | --- |
| cliff | 2, 1, 1, 1 | 17, 13, 13, 13 | -13.0000 by all four |
| rooms | 1, 1, 1, 1 | 23, 20, 20, 20 | -20.0000 by all four |
| maze | 1, 1, 1, 1 | 16, 15, 16, 14 | 0.5133, 0.4883, 0.4883, 0.5133 |

This section had no command at all above it for nineteen tracks, and the value
column was rounded to two places where the command prints four.

**Every rule finds the goal on the first episode of nearly every seed.** The
settled column, which is the mean episode length over the last fifty, only says
how much exploring each is still doing at the end: epsilon-greedy is still
taking a random action one step in ten and the other three have stopped.

That is not a result about exploring. It is a result about these grids. The
goal of the cliff walk is thirteen steps from the start and the goal of the
Dyna maze is fourteen, so an agent that has learned nothing arrives anyway and
there is nothing for a better rule to be better at.

### The corridor was built to have something to be better at

One folded path, nine by nine, no branch anywhere in it. The route is forty
eight steps and nothing pays until the end of it. A random policy covers a line
of length n in about n squared steps, so an agent that has learned nothing
spends a whole episode not finding out that there is anything to find.

```console
$ python scripts/measure_exploration.py
```

That is the first of the commands at the top of this section, repeated here
because the corridor is its default environment and this is the table it
makes. The block above the table before this one is three other grids.

<!-- not checked, column rule, time: the rule column holds the settings
these rows were run at, and the time column belongs to the machine -->
| rule | first end | steps to it | settled | value found | time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `epsilon-greedy` | 18 | 17,500 | 53 | 0.0897 | 24s |
| `softmax:0.02` | 21 | 20,000 | 104 | 0.0897 | 31s |
| `count-bonus:0.5` | 6 | 5,000 | 49 | 0.0897 | 23s |
| optimism 1 | **1** | **0** | 48 | 0.0897 | 2s |

*Ten seeds, three hundred episodes, discount 0.95. The best possible return
from the start is 0.0897, which is 0.95 raised to the forty seventh: the reward
arrives on the forty eighth step and is discounted once for every step before
it.*

The episode each seed first reached the goal on:

```
epsilon-greedy   13   2  36  16  24  18  35  13  19  22
softmax:0.02     12  29  13  19  16  29  97  27  23   2
count-bonus:0.5   6   4   9  29   3   3   7   3   8   6
optimism 1        1   1   1   1   1   1   1   1   1   1
```

**All four end with the same policy.** The value column is what each learned,
worked out exactly from the model, and it is the optimum in every row. What
differs is the seventeen thousand five hundred steps epsilon-greedy spent
before there was anything to learn from.

### Neither dial matters, and the digest says so exactly

The obvious reply to that table is that softmax was set wrongly. It was not.
Three episodes of the corridor on seed 1, and the digest of the path each one
walked:

```
epsilon-greedy:0.0   8ecde0e097a33e50
epsilon-greedy:0.1   a9e79fa6af4c3715
epsilon-greedy:0.5   a9e79fa6af4c3715
epsilon-greedy:0.9   a9e79fa6af4c3715
epsilon-greedy:1.0   8ecde0e097a33e50
softmax:0.02         83a3e33beaed3144
softmax:0.001        83a3e33beaed3144
```

Not similar runs. The same run, step for step.

**Before anything pays, every value in the table is the starting number.** A
rule that ranks actions by value is ranking a row of equal numbers, so it takes
a uniform action whatever its dial says. At an epsilon of 0.1 the agent takes a
random action one time in ten and breaks a four way tie the other nine, and
both of those are one uniform draw over four actions, so the run does not
depend on which of them happened.

The two ends of that ladder have an identity of their own for a smaller reason.
`Rng.chance` answers 0 and 1 without drawing, so those two runs spend one draw
fewer per step. They are still uniform and they are still identical to each
other.

The count bonus does the same thing from the other side. A confidence of 0.1,
0.5 and 2 gives the same ten seeds:

```
count-bonus:0.1   6   4   9  29   3   3   7   3   8   6
count-bonus:0.5   6   4   9  29   3   3   7   3   8   6
count-bonus:2     6   4   9  29   3   3   7   3   8   6
```

Multiplying every count-based bonus by a constant does not reorder them, and
with the values all equal the bonus is the whole score.

**The dial of a rule cannot help find a first reward. Only what the rule ranks
by can.** That is why the four rows above split two and two rather than four
ways.

The dials do their work afterwards, once the values are real, and what they
decide is how much exploring carries on. That is the settled column: 48, 49 and
77 steps at a confidence of 0.1, 0.5 and 2, against a route that is 48.

### What the alternatives cost

`count-bonus` keeps a second dictionary the size of the table and writes to it
on every step. Nothing else here keeps one, so the agent asks the rule whether
to count rather than counting always.

`count-bonus` also cannot be the behaviour policy of an importance sampling
method. It explores by ranking rather than by chance, so every action but its
choice has a probability of zero, and `off-policy-mc` divides by exactly that
number. Nothing is wrong with either piece and they cannot be used together.
The error says so rather than learning from a correction that is not defined.
Softmax gives every action a share and works.

### Where each one is weak

**A temperature is in the units of the value.** Epsilon means the same thing on
every problem because it is a probability. A temperature divides a difference
between two values, so what counts as hot depends on how far apart the values
of that environment are. On the cliff walk neighbouring actions differ by ones
and a temperature of one already follows the ordering closely. On the corridor
they differ by thousandths and the same setting is uniform. A setting carried
from one environment to another is not the same setting.

**Optimism has to know the scale of the answer.** An optimism of 1 is right for
a grid that pays 1 at the goal and nothing else. On the cliff walk, where every
reward is negative, a table of zeros is already above every true value, so an
agent there is optimistic whether or not anybody asked for it. Ten seeds and
three hundred episodes read -50.25 at an optimism of 0 and -49.78 at 1, which
is the same number.

**The count bonus is the bandit rule once per state, and the guarantee does not
come with it.** A bandit's arms pay out on their own and a grid's actions pay
out through the states they lead to. So it drives the agent to try every action
of the states it stands in, and nothing in it drives the agent towards a state
it has never reached. Once every action of a state has been taken a few times
the bonus there is small and the agent is greedy again, whatever is still
unexplored two rooms away.

---

## Learning about one policy while following another

Every agent above learns about the policy it is following, or about the greedy
policy behind it. `off-policy-mc` learns about the greedy policy from episodes
an exploring one collected, which is the question worth asking whenever the
data already exists and was not collected by whoever now wants an answer.

The correction is the ratio of the two policies at each step, multiplied along
the episode. The target policy here is greedy, so the ratio is zero the moment
the behaviour policy explored, and the walk backwards stops there. That is the
algorithm and not an optimisation.

### The two estimators, and what the second one is for

Both divide the same product. `ordinary` divides by a count and is unbiased
with unbounded variance. `weighted` divides by the sum of the ratios it really
saw, and is biased with far less.

"Unbounded" is doing a great deal of work in that sentence, so here it is as a
number.

```console
$ python scripts/measure_importance.py --episodes 1200
```

Ten seeds, 1200 episodes, a behaviour policy that explores a fifth of the time.
`spread` is how far apart the ten seeds' estimates of one cell are, averaged
over every cell all ten of them credited.

The episode count is on the command line because it is not the default. This
block said `python scripts/measure_importance.py` for five tracks, which runs
1500 episodes and prints a different table, and the prose beside it said 1200
the whole time. `python scripts/check_numbers.py --only importance` is what
noticed.

| grid | estimator | cells | spread | widest | policy |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen lake, best 0.824 | ordinary | 10 | 4.194 | 13.204 | 0.237 |
| frozen lake | **weighted** | 10 | **0.710** | **0.913** | **0.369** |
| maze, best 0.513 | ordinary | 2 | 1,970,224,597,202.702 | 3,928,201,830,698.047 | never finishes |
| maze | **weighted** | 34 | **0.058** | **0.136** | **0.498** |

**The ordinary estimator's worst cell on the maze reads about four trillion,
on a problem whose best possible value is 0.513.**

That is not a fault in the arithmetic. The behaviour policy takes the greedy
action about 85% of the time, so every greedy step multiplies the correction by
about 1.18. A long run of them multiplies it by 1.18 raised to the length of
the run, and a maze episode early in a run is long.

The unbiasedness is real and it is not worth anything here. An estimator whose
average over infinitely many runs is right, and whose ten runs disagree by
twelve orders of magnitude, has not told anybody what the maze is worth. The
registry default is `weighted`.

Two other things in that table are worth reading. The ordinary estimator
credited **2** cells that all ten seeds agreed on, against 34 for the weighted
one, so the huge spread is measured over almost nothing. And it never reached
the goal on the maze on any of the ten seeds, where the weighted one reaches
0.498 against a best possible of 0.513.

### Tree backup asks the same question and never divides

`tree-backup` is off-policy for the same reason and by a different route. It
reaches back n steps, learns about a policy it is not following, and multiplies
by no ratio at all.

An importance ratio is a correction applied to a sample of the wrong thing.
Tree backup never takes that sample: at every step of the window it takes the
expectation over what the target policy would have done, and only the action
really taken carries the recursion any further. The actions not taken are
accounted for by their current estimated value rather than by a correction, so
there is nothing to divide by. A test reads the source of the update and
asserts there is no division in it.

```console
$ python scripts/measure_agents.py --env cliff --runs 10 \
    --agents tree-backup off-policy-mc q-learning n-step-sarsa
```

| agent | greedy, exactly | stuck of 10 |
| --- | ---: | ---: |
| q-learning | **-13.00** | |
| tree-backup | -15.00 | |
| n-step-sarsa | -16.80 | |
| off-policy-mc | - | **10** |

**The method that exists to avoid importance sampling works on the grid where
importance sampling cannot.** Both are asking what the greedy policy is worth
from data an exploring one collected. One of them answers.

It is not free. The recursion is multiplied by the target policy's probability
of the action taken, at every step, so where the two policies disagree often
that probability is small and the reach is short. Tree backup does not escape
the problem of two policies differing. It pays for it in a currency that cannot
explode.

### Where it cannot learn at all

Not on the cliff walk, and the reason is worth writing down rather than leaving
for somebody to rediscover.

An episode teaches only the tail after the last step the behaviour policy
explored. On the cliff walk, episodes run to the five hundred step limit far
more often than they reach the goal: over two hundred episodes, ten of them
finished and the agent credited **about one cell per episode**. The cell next
to the goal was still untouched.

Two tests hold that shape, because a weakness that is measured is worth more
than a weakness that is described.

---

## Planning, on the maze

```console
$ rel train dyna-q --env maze --set planning_steps=0
$ rel train dyna-q --env maze --set planning_steps=5
$ rel train dyna-q --env maze --set planning_steps=50
```

Seed 3, step size 0.1, epsilon 0.1, discount 0.95. Steps taken in each of the
first ten episodes:

<!-- not checked: the length of each of the first ten episodes of seed 3,
read off the three runs above rather than printed by any of them. `rel
train` draws the returns and prints a summary, not a list of step counts -->
| planning steps | first episode | episodes 2 to 10 |
| ---: | ---: | --- |
| 0 | 518 | 321, 1236, 178, 126, 366, 611, 269, 96, 232 |
| 5 | 966 | 179, 23, 33, 18, 19, 18, 20, 16, 16 |
| 50 | 44 | 42, 16, 17, 21, 16, 16, 22, 20, 20 |

The shortest path is 14 steps. With no planning, the tenth episode still takes
232 steps. With fifty, the third is already near the shortest path.

Nothing about a replayed step is imagined. It is a step that really happened,
applied again. What it buys is that one real step can move many table entries,
so a single walk to the goal teaches a path rather than a cell.

### The exploration bonus, and how large it can be

Dyna-Q+ adds `kappa` times the square root of how long it has been since a
state and action were tried. On a maze where a wall moves after the agent has
learned a route, that is the difference between finding the new short way and
keeping the long one:

```console
$ python scripts/measure_shortcut.py
```

| agent | mean episode length after a shortcut opens |
| --- | ---: |
| dyna-q | 17.9 |
| dyna-q-plus, kappa 0.001 | **12.5** |
| dyna-q-plus, kappa 0.01 | 927.9 |
| dyna-q-plus, kappa 0.05 | 926.7 |

*Five seeds, 60 episodes before the shortcut opens and 120 after.*

The last two rows are the point. The bonus is added to a remembered reward, and
on this maze the goal pays 1 and nothing else pays anything. With kappa at 0.01
the bonus passes 1 after ten thousand steps, the planning stops being about the
environment, and the agent wanders.

**There is no safe default for kappa.** It has to be small against the rewards
the environment really pays, and nothing about the agent knows what those are.
That is why it is an argument with 0.001 behind it and not a constant.

### Ordering the replays, counted in updates

```console
$ python scripts/measure_sweeping.py --episodes 400
```

Comparing two planners by episode hides the question. Both of them replay
remembered steps, and the question is how many replays it takes. An update here
is one application of the learning rule to one cell: `dyna-q` makes one for the
real step and its full quota after it, and `prioritised-sweeping` makes one for
every entry it takes off the queue.

| planner | solved | updates | fewest | most | idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| dyna-q | 9 of 10 | 8520 | 2532 | 19296 | 6.000 |
| prioritised-sweeping | 9 of 10 | **880** | 705 | 6590 | **0.001** |

*The maze, ten seeds, 400 episodes, five planning steps, step size 0.5.
`updates` is the median number of updates before the greedy policy was exactly
optimal, over the seeds where it was. `idle` is the updates per real step over
the last twenty episodes.*

The median is a factor of nearly ten, and the worst seed the sweeper
solves takes fewer updates than the median seed the uniform planner
solves. Both see the
same steps and hold the same model. The only difference is which remembered
step is replayed next.

```console
$ python scripts/measure_sweeping.py --episodes 400 --each-seed
```

Per seed, in updates. The command above had no way to print this until the
check went looking for the command behind the table and found none:

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dyna-q | 3534 | 5412 | 11958 | 8322 | 12246 | - | 10950 | 8520 | 2532 | 19296 |
| prioritised-sweeping | 1060 | 880 | 965 | 840 | 6590 | 705 | 730 | 920 | - | 725 |

### The idle column is the other half of it

`dyna-q` makes six updates for every step for as long as it runs, whatever it
has left to learn. Once the table has settled, all six teach nothing.

`prioritised-sweeping` stops. A step whose replay would move a value by less
than `threshold` never goes on the queue, so when nothing in the model would
move, the queue empties and stays empty. Over the last twenty episodes of these
runs it makes about one update every thousand steps.

**It stops because it believes it is finished, and it can be wrong about that.**
Seed 9 is the case. It settles on a route two steps longer than the shortest
one, nothing in the model changes any more, so nothing is queued and it never
looks again. Two hundred further episodes move it nowhere. That is the seed
`dyna-q` solves in 2532 updates, which is the fewest it needs anywhere.

The two planners lose one seed each and they are not the same seed. Uniform
replay keeps grinding at a settled model and sometimes that is what is needed.
Ordered replay stops, and stopping is what makes the answer final.

---

## Planning at the moment of choosing

```console
$ rel train mcts --env maze
$ rel train mcts --env maze --set reuse=off --episodes 40
$ python scripts/measure_search.py --episodes 40 --runs 3
```

The two planners above spend a model in the background. They improve a table,
and when the moment comes to act they read it. The work is done before the
question is asked, and it is done for every state whether or not the agent will
ever stand there.

`mcts` does the opposite. It spends the model on the state it is standing in,
right now, and on the futures that follow from there. Four steps, repeated
`simulations` times:

    descend    from the state, take the action the tree ranks highest, until
               a state the tree has not seen
    expand     put that state in the tree
    roll out   act at random from there until an ending or a depth limit
    back up    credit every state and action on the way down with the return

Then take the action with the most visits at the root. The most visits rather
than the best mean: a mean over three samples is high by luck often enough to
matter, and the count is what the descent has already agreed with.

The rule that picks an action inside the tree is the `count-bonus` of the
exploration section, applied to the means in a node rather than to a learned
table. Upper confidence selection inside a tree and upper confidence
exploration in a grid are the same rule, and this project writes it once so the
two cannot drift apart.

**It is handed the model rather than learning one.** `dyna-q` builds its model
out of the steps it takes. `mcts` here is given the environment's own. That is
not a fair fight, and the direction it is unfair in is the interesting one: the
agent that was handed the answer is the one to beat.

### Counted in model steps, because that is what all three spend

<!-- not checked, column time: seconds belong to the machine, and the
model steps beside it are the currency this section counts in -->
| agent | settled | policy found | stuck | model steps | time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mcts`, 10 simulations | 16 | 0.4633 | | 94,296 | 1s |
| `mcts`, 25 simulations | 16 | 0.4633 | | 180,736 | 2s |
| `mcts`, 50 simulations | 16 | 0.4633 | | 309,069 | 3s |
| `mcts`, 50, `reuse=off` | 330 | - | 3 | 42,345,425 | 288s |
| `dyna-q`, 5 planning steps | 15 | **0.5133** | | **7,345** | 0s |
| `dyna-q`, 20 planning steps | 16 | **0.5133** | | 26,480 | 0s |
| `prioritised-sweeping` | 17 | **0.5133** | | **3,575** | 0s |

*The Dyna maze, three seeds, 40 episodes, discount 0.95. `settled` is the mean
episode length over the last ten and the shortest route is 14 steps. `policy`
is what the greedy policy is worth, worked out from the model.*

An episode against a table of returns would let an agent that asks a model a
thousand times a step look free, and against wall clock it would compare the
speed of the code. Model steps is what each of these is actually spending: a
replay for the two planners, a simulated step for the search.

**Every row but one settles in about the same place, and they are two orders of
magnitude apart in what it cost.** The search reaches 16 steps for 94,296 model
steps at its cheapest. Prioritised sweeping reaches 17 for 3,575, and it had to
learn its model on the way.

### More simulations buy nothing here

10, 25 and 50 simulations all settle at 16 steps. Three times the budget per
decision, and the same answer.

The reason is the row underneath. What is carrying this agent is not the search
from the state it is in, it is the tree left over from the last time it was
there. Turn that off and there is nothing left.

### Without the tree it is hopeless

`reuse=off` clears the tree before every decision, which is decision-time
planning with nothing else in it. On this maze that agent never settles: 330
steps against a shortest route of 14, no seed of three ending with a policy
that reaches the goal at all, and forty two million model steps to fail.

**The tree is doing the learning.** With it kept, the agent is a planner with a
memory and it behaves like the other planners. Without it, the same code with
the same model is worse than Q-learning with no model at all.

The console block above says `--episodes 40` on that row alone, and the reason
is the row itself. `rel train` runs 500 episodes by default, and 500 episodes
of an agent that cannot learn are 500 walks of a few hundred steps each with a
fresh search at every one of them. That run was still going at fifty minutes
when the number checker gave up on it, which made this the one command on the
page nothing had ever run to the end. Forty is what the table beside it uses,
and it takes six minutes and shows the same failure.

### Why the rollout cannot carry it

A simulation that stops at the depth limit is worth what it collected and
nothing after. There is no estimate of the rest, because there is no value
function to ask.

That is affordable when a random rollout reaches endings. It does not. Over two
hundred tries, a random policy of thirty steps reaches an ending **none** of
those times on the Dyna maze and **none** on the cliff walk. Both goals are
about fourteen steps away and a random walk does not go in a straight line.
`tests/test_search.py` measures it.

So the tail of a simulation is nearly always the same number, and on the cliff
walk it is exactly the same number: every step there pays -1 and the discount
is one, so a rollout that stops at the depth limit is worth minus the depth
wherever it went. Nothing separates two branches.

```console
$ python scripts/measure_search.py --env cliff --episodes 10 --runs 3 \
    --only "mcts 50,dyna-q 5,prioritised-sweeping"
```

<!-- not checked, column time: seconds belong to the machine, and the
model steps beside it are the currency this section counts in -->
| agent | settled | policy found | stuck | model steps | time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mcts`, 50 simulations | **500** | - | 3 | **15,000,000** | 129s |
| `dyna-q`, 5 planning steps | 56 | - | 3 | 2,815 | 0s |
| `prioritised-sweeping` | 44 | **-13.0000** | | 2,201 | 0s |

*The cliff walk, three seeds, ten episodes. The shortest route is 13 steps and
the step limit is 500.*

**Every episode of every seed runs to the step limit.** The search never
reaches the goal at all, and it spends fifteen million model steps not reaching
it. Ten episodes is too few for `dyna-q` to have finished either, which is why
its policy column is empty as well, but it is walking 56 steps where the search
is walking 500.

That is why the test suite runs this agent on the maze rather than on the cliff
walk: not to be kind to it, but because there is nothing to test in a run that
cannot start.

### Where it is weak

**It has no policy until it searches.** Every other agent here answers "what
would you do in that cell" by reading a row. This one answers by running a
search, and a search of fifty simulations from a cold state is a noisy answer.
That is the `policy` column: 0.4633 is a sixteen step route and 0.5133 is the
shortest one at fourteen, so the search behaves as well as the planners and
describes itself two steps worse. The picture `rel train` draws of it is that
description rather than a table.

**The standard repair is not here.** What fixes a rollout that reaches no
ending is a value at the leaf instead of a rollout, learned or given. Adding
one would make this the shape that plays board games, and it is not built.

---

## An action that lasts several steps

```console
$ rel train options-q --env rooms
$ rel train options-q --env rooms --set hallways=off
$ python scripts/measure_options.py
$ python scripts/measure_options.py --runs 20
```

An option is a policy, a rule for when it stops, and the states it can start
from. "Cross this room and stop at the doorway on your left" is one, and
choosing it covers the seven steps that follow.

`options-q` is Q-learning whose choices are options. The update is the same
shape with the multi step return substituted in: the discounted reward
collected while the option ran, bootstrapped by the discount raised to how many
steps it took. At one step that is Q-learning exactly.

### Where the options come from

Nothing here names the four rooms grid. A gap in a wall is a cell you can only
pass straight through, which is a shape the layout can be read for. Take those
cells out of the model and what is left falls into rooms. A gap that touches
two of them is a doorway, and for each room and each doorway on its edge there
is one option: cross this room, stop there.

On the four rooms grid that finds the eight options of Sutton, Precup and
Singh, 1999. On the Dyna maze it finds four gaps and no options, because those
gaps are passages inside one room rather than doors between two, and on the
cliff walk, the windy grid and the frozen lake it finds nothing at all because
none of them has an interior wall.

### The collapse, and then the cost

| agent | while learning | greedy, exactly | stuck |
| --- | ---: | ---: | ---: |
| q-learning | -22.44 | -20.00 | |
| options-q, `hallways=off` | -22.44 | -20.00 | |
| options-q | -25.01 | -20.00 | 1 |

*Four rooms, ten seeds, 500 episodes, at the default epsilon of 0.1. The best
possible return is -20. The third command above prints this table, and the row
that says `q-learning` is why it prints it: that row is a different class
reaching the same number, and without it the first two rows are one agent
compared with itself.*

The first two rows are the collapse. An option that stops after every step is a
primitive action, so an agent holding only those is Q-learning, and it is
Q-learning to the last two decimal places on a whole grid rather than on one
update.

The third row is the result. **The eight hallway options cost 2.57 return while
learning and leave one seed of ten with a policy that never reaches the goal.**
What they were expected to do was the opposite.

### Where the cost is

The cost is the price of exploring, and the way to show that is a ladder rather
than an argument. An exploratory choice that lands on a long option commits
several steps in one direction, so it is paid for several times over. A cost
that comes from that has to fall with epsilon. A cost that came from what was
learned would not.

| epsilon | actions only | with options | cost | cost / epsilon | long | length |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | -25.75 | -31.95 | 6.20 | 31.0 | 11% | 1.34 |
| 0.1 | -22.56 | -25.00 | 2.43 | 24.3 | 10% | 1.28 |
| 0.05 | -21.31 | -22.40 | 1.09 | 21.9 | 9% | 1.25 |
| 0.02 | -20.54 | -21.00 | 0.46 | 23.2 | 9% | 1.24 |

*Twenty seeds, which is the second of the two commands above. `long` is the
share of choices that were an option lasting more than a step, and `length` is
the mean number of steps an option ran for.*

Epsilon changes by a factor of ten and the fourth column sits between 22 and 31
with no trend in it. The cost is the exploring.

That is not an argument against options. It is a measurement of what this
particular pairing costs on this particular grid, where the reward is dense and
one step of feedback arrives after every step. An option is a commitment, and a
commitment is worth what it saves minus what it costs to be wrong about. Here
there is nothing to save: the value of a cell is already carried back one cell
per step by every reward, and the doorways are not where the agent wants to be.

### Crediting the states an option passed through

```console
$ rel train intra-option-q --env rooms
$ python scripts/measure_intra_option.py --episodes 800 --block 100
```

`options-q` waits for an option to stop and credits the state it started in.
Three steps inside one option move one cell, and the two states it passed
through learn nothing about it. That is the usual complaint about the method,
and intra-option learning is the usual answer: one real step is evidence about
every option that would have taken that action there, so every one of them is
updated whether or not it was the one running.

The section above reads the cost of having options as the price of exploring,
and it reads it off a ladder rather than off a mechanism. That reading makes a
prediction. If the cost is the exploring, then fixing the credit assignment
will move the early episodes and leave the late ones roughly alone.

| agent | 1-100 | 101-200 | 201-300 | 301-400 | 701-800 | updates/step | last 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| options-q | 123.9 | 41.2 | 30.5 | 26.0 | 24.5 | 0.78 | -24.50 |
| intra-option-q | **105.3** | **31.2** | **25.8** | 25.1 | **24.0** | 1.64 | **-23.96** |
| options-q, no options | 95.4 | 28.6 | 22.9 | 22.7 | 22.5 | 1.00 | -22.48 |
| intra-option-q, no options | 95.4 | 28.6 | 22.9 | 22.7 | 22.5 | 1.00 | -22.48 |

*Four rooms, ten seeds, 800 episodes. The numbered columns are the mean
episode length over that block. `updates/step` is how many times the learning
rule was applied to a cell for each step taken.*

**The prediction holds.** Intra-option learning is 15% shorter in the first
hundred episodes and 24% shorter in the second, and by the fourth block the two
are within noise of each other. It recovers 0.54 of the 2.02 that having the
options costs at all, and it does that work early.

So two independent measurements now say the same thing. The ladder says the
cost scales with epsilon, and fixing the credit assignment leaves most of it
standing. **The cost of an option here is the cost of committing to it while
exploring, and not the cost of learning about it slowly.**

What it buys is not free: 1.64 applications of the learning rule for every step
taken, against 0.78. Slightly over twice the work for a quarter of the gap.

### The last two rows are the collapse

A primitive option stops after every step, so what it is worth from where it
landed is the best value there, and the rule is Q-learning exactly. Both agents
holding only those are the same agent, and the table says so in every column
of every block: 95.4, 28.6, 22.9, 22.7, 22.5, and -22.48 on the last hundred.

That is a stronger check than the unit test beside it. The test feeds three
transitions and compares two tables. This runs eight thousand episodes on ten
seeds and gets the same number to two decimal places.

### The picture is not the policy

There is a second thing here, and it is about measurement rather than about
options.

Every report in this project reads a policy off an agent one cell at a time:
`greedy` is asked about each state and the answers are laid out as a map. An
agent that chooses options does not follow one action per cell. It commits to
an option and runs it, and asking it about a cell it is not standing in cannot
be allowed to move that option along, because the renderer asks about every
cell of a grid in any order.

So `greedy` here reports the first step of the best option available at that
cell, with no running state at all. That is a real policy and it is not the one
the agent follows. On the four rooms grid, after 500 episodes on seed 1, 18 of
the 103 cells carry a choice that would have run on. The policy picture marks
them, because a picture that did not would be a picture of something nobody
does.

The stuck seed in the table above is this. What the agent learned is fine, and
the map read off it walks in a circle.

---

## Prediction, where the answer is known

Every agent above controls. It keeps a number for each action, ranks them, and
acts on the ranking. Estimating what a policy you are not trying to improve is
worth is the other half of the subject, and it is worth separating because
control mixes two questions: when a policy improves, its estimates chase a
moving target, and a measurement of how good the estimates are cannot say
whether the answer moved or the estimate did.

```console
$ rel train td --env walk --set start_value=0.5
$ python scripts/measure_prediction.py
```

### The one problem whose answer is arithmetic

The random walk is five cells in a line between two endings. The left one pays
nothing, the right one pays 1, and a policy that goes each way half the time
reaches the right one from the k-th cell exactly k times in six.

```
|--o--|          0.167  0.333  0.500  0.667  0.833
```

Every other table on this page compares an agent against dynamic programming
over the same model. That is a strong check and it is still a check against
another computation. This one is a check against arithmetic, and a test holds
the closed form against a sweep over the model so that a fault in either shows
up as a disagreement rather than as a plausible number.

### TD against Monte Carlo

| method | step 0.01 | step 0.02 | step 0.05 | step 0.1 | step 0.2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| td | 0.1283 | 0.0689 | **0.0343** | 0.0543 | 0.0822 |
| 3-step td | 0.0880 | **0.0455** | 0.0571 | 0.0820 | 0.1219 |
| td-lambda 0.8 | 0.1124 | 0.0592 | 0.0470 | 0.0685 | 0.1017 |
| mc, first visit | **0.1278** | 0.0761 | 0.0590 | 0.0854 | 0.1313 |
| mc, every visit | 0.0942 | 0.0768 | 0.1189 | 0.1739 | 0.2427 |

*Fifty runs of a hundred episodes, estimates starting at 0.5. Each cell is the
root mean square error against the true values, over the five states an agent
can be in. An untrained table scores 0.2357.*

The rows are `td`, `n-step-td`, `td-lambda` and `mc-prediction` twice, the last
of them with `--set first_visit=off`. The first four are the registry defaults:
`n-step-td` takes three steps and `td-lambda` decays at 0.8 without being told
to. Every row here sets `start_value` to 0.5.

**TD's best is 0.0343 and Monte Carlo's best is 0.0590.** Both are estimating
the same thing from the same episodes, and the one that uses what it already
believes about where it ended up gets 42% closer than the one that waits to
find out what the return was.

The two are level at the smallest step size and TD is ahead everywhere else. At
0.01 neither has arrived after a hundred episodes, so that column is a
measurement of how far they have got rather than of where they settle.

### Every row is a U

There is no best step size in that table, only a best step size for a method
and a budget. Too small and a hundred episodes is not enough to arrive. Too
large and the estimate never settles.

**A constant step size does not converge at all.** It tracks: the estimate ends
up in a band around the answer whose width is proportional to the step size.
That is the right behaviour for a problem that moves and the wrong behaviour
for one that does not, and the random walk does not move. It is also why the
test that says these methods reach the answer decays the step size, and why
this table is a ladder rather than a row.

### Every visit is the odd one out

Crediting a state once for every time an episode passed through it, rather than
once for the first, is worse everywhere but the smallest step size, and it gets
worse fastest as the step size grows.

The random walk is why. It doubles back constantly, so one episode can pass
through a cell four or five times, and every visit then moves that cell four or
five times in the same direction from the same return. At a step size of 0.2 it
scores 0.2427, which is worse than not learning at all.

---

## The discount is part of the question

```console
$ python scripts/measure_average_reward.py
$ python scripts/measure_average_reward.py --length 8 --episodes 30
$ rel train differential-q --env loops
$ rel solve --env loops --discount 0.7
```

Every environment above has a goal, so an episode ends and the return of a
policy is a finite number whatever the discount is. The discount changes how
much a run is worth and not which policy is best.

A task that never ends has no such number. The reward keeps arriving for ever,
and the two ways to compare policies are to discount the future or to take the
average per step. **Those are different comparisons and they can disagree.**

### One decision, made over and over

`loops` is a junction. One action takes a short loop that pays 1 and comes
straight back. The other takes a long loop that pays 10 and takes five steps to
come round.

    short   one step,  pays 1     ->  1.00 per step
    long    five steps, pays 10   ->  2.00 per step

By reward per step the long loop is twice as good. The discounted value of the
junction under each is

    short   p / (1 - d)
    long    d^(n-1) q / (1 - d^n)

and those are equal at a discount of **0.7394**. Below it the short loop has
the higher discounted value.

<!-- not checked: worked out from the two closed forms above it rather
than run -->
| discount | best policy | what it collects per step |
| ---: | ---: | ---: |
| 0.5 | short | 1.00 |
| 0.7 | short | 1.00 |
| 0.735 | short | 1.00 |
| 0.745 | **long** | **2.00** |
| 0.8 | long | 2.00 |
| 0.9 | long | 2.00 |
| 0.99 | long | 2.00 |
| none, `differential-q` | **long** | **2.00** |

*The `best policy` column is from `value_iteration` on the model, not from a
run. `q-learning` at each of those discounts finds the same answer on all five
seeds, and the script prints both columns side by side.*

**The agent at 0.7 is not going wrong.** It is playing the exactly optimal
policy for the question it was asked, and the question had a discount in it.
Nobody chooses 0.7 to mean "prefer the loop that pays half as much". They
choose it because it converges quickly.

### The threshold depends on the environment, and nothing knows it

Lengthen the long loop and it pays less per step, so a discounted agent has to
be more patient to prefer it:

```console
$ python scripts/measure_average_reward.py --length 2 --episodes 20
$ python scripts/measure_average_reward.py --length 3 --episodes 20
$ python scripts/measure_average_reward.py --length 5 --episodes 20
$ python scripts/measure_average_reward.py --length 8 --episodes 20
$ python scripts/measure_average_reward.py --length 10 --episodes 20
```

| long loop | it pays per step | crossover | 0.9 picks | 0.99 picks |
| ---: | ---: | ---: | ---: | ---: |
| 2 steps | 5.00 | 0.1111 | long | long |
| 3 steps | 3.33 | 0.3935 | long | long |
| 5 steps | 2.00 | 0.7394 | long | long |
| 8 steps | 1.25 | 0.9408 | **short** | long |
| 10 steps | 1.00 | 1.0000 | short | short |

One row per command, and the per step column is written to the two places the
command prints rather than the three it used to carry.

At eight steps the long loop is still a quarter better per step, and **a
discount of 0.9 takes the short one**. That is not an exotic setting: 0.9 is
what this environment suggested for itself until checking this table found it
wrong. The suggestion is computed from the crossover now, and the environment
can only do that because it is a toy that knows its own answer.

At ten steps the two loops are worth the same per step and no discount below
one prefers the long one, which is correct rather than a failure: there is
nothing to prefer.

**There is no safe default for the discount on a task that never ends.** It has
to be close enough to one for the environment it is used on, and nothing about
the agent knows how close that is. This project has met that shape once
already, in `kappa` on Dyna-Q+, and the answer is the same both times: a
setting that has to be right against numbers the agent cannot see is a setting
that will be wrong somewhere.

### The agent with no discount to get wrong

`differential-q` is Q-learning with the rate it is collecting subtracted from
every reward, and no discount anywhere:

    error = reward - average + max Q(s', a) - Q(s, a)
    Q(s, a) += step_size * error
    average += average_step * step_size * error

It takes the long loop at every step size and every average step tried, which
is what having no discount buys.

**The rate it learns is the rate it collects.** After twenty episodes it
believes 2.000 a step on a task whose better policy collects exactly 2.000 a
step, so the number it subtracts is an estimate of something real rather than a
bias term that happens to work.

The average is learned from the same error rather than averaged over the
rewards that arrived. A running mean of the rewards would be the rate of the
behaviour policy, exploration included, and the rate the update needs is the
one of the policy being learned.

### Where it is weak

**The values are relative.** Adding the same number to every cell of a
differential table changes nothing, so its value map is a picture of what is
better than what and not of what anything is worth. The policy read off it is
the same policy, which is all that is asked of it, but the numbers beside the
picture do not mean what they mean for every other agent on this page.

**It assumes the task does not end.** The derivation is about one long run with
a rate. Given a terminated step this drops the bootstrap, which is the only
sensible thing to do, and what comes out on an episodic task is an agent
maximising reward per step over the whole run rather than per episode. That is
sometimes the question and usually not.

**Only the off-policy version is here.** The on-policy one needs the action the
policy really takes next, which means holding the last transition and waiting a
step exactly as `sarsa` does. On this environment the two agree, so it was not
built to find that out.

---

## A picture that is mostly the solver

```console
$ python scripts/measure_gambler.py
$ rel solve --env gambler
$ rel solve --env fair-gambler
```

A gambler stakes part of a capital on a coin. Heads pays the stake, tails takes
it, reaching 100 pays 1 and reaching nothing pays nothing. This is Sutton and
Barto's example 4.3, and it is usually drawn twice: the value of each capital,
and the stake to make at it. The second picture is a jagged staircase and it is
the one people remember.

### The fair coin has a closed form

A fair game stopped at either end is worth the same however it is played, so
the value of a capital is that capital over the goal.

| capital | closed form | swept |
| ---: | ---: | ---: |
| 1 | 0.010000 | 0.010000 |
| 25 | 0.250000 | 0.250000 |
| 50 | 0.500000 | 0.500000 |
| 99 | 0.990000 | 0.990000 |

The worst gap over all ninety nine capitals is **1.6e-12**. That is a check
against arithmetic rather than against another sweep, and the random walk was
the only environment here that offered one.

### How much of the staircase survives a change that cannot change the answer

Two things are varied and neither is part of the problem: how tightly the sweep
is run, and which of the two solvers runs it. The values agree to the tolerance
every time.

| heads | value at the start | widest gap between stakes | capitals with one best stake | moved by the tolerance | moved by the solver |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.250000 | 0.115 | 25 | 26 | 36 |
| 0.4 | 0.400000 | 0.0632 | 25 | 7 | 36 |
| 0.5 | 0.500000 | **8.05e-13** | **0** | 60 | 72 |
| 0.55 | 0.999956 | 0.45 | 86 | **0** | **0** |

*Tolerances 1e-6 and 1e-9, and the two solvers at 1e-9. A capital counts as
decided when one stake beats the rest by more than the tolerance.*

**At a fair coin nothing is decided at all.** Every stake is exactly as good,
so the largest gap between the best stake and the worst over all ninety nine
capitals is 8e-13, which is the sweep's own rounding. Seventy two of the ninety
nine stakes change when the solver does. Whatever a solver draws there is its
arithmetic and nothing about gambling.

**At an unfavourable coin a quarter of it is real.** Twenty five capitals have
a best stake, and thirty six of the ninety nine still change with the solver.
The jaggedness in the picture is partly the problem and mostly not.

**At a favourable coin all of it is real.** Stake one everywhere, eighty six
capitals decided, and nothing moves at all. The value at the start is 0.999956
rather than one, because a gambler betting a pound at a time can still lose a
hundred in a row.

### Staking nothing is not an action, and that was measured

The problem as stated lets the gambler stake nothing. It is a loop of one
state, so at a discount of one it is worth exactly what the state is worth and
ties with the best real stake everywhere. Built that way, value iteration
staked nothing at **35 of the 99 capitals** and `evaluate_policy` scored the
resulting policy at minus infinity, because a gambler following it never
reaches an ending.

Tie breaking is a property of the solver rather than of this problem, so the
fix belongs in the environment: the action that is never useful and always
ties is the one to leave out. `TestStakingNothingIsNotAnAction` in
`tests/test_gambler.py` drives the version with it back in and holds the
finding.

---

## What the van is worth

```console
$ python scripts/measure_rental.py
$ rel solve --env rental --discount 0.9
```

Two car parks holding twenty cars each. Renting a car pays 10 and needs a car
to be there, cars come back the next day, and overnight a van moves up to five
cars between the locations at 2 each. Requests and returns are Poisson, so this
is the one environment here whose model is a distribution rather than a handful
of branches written out.

Sutton and Barto's example 4.2 is usually run once and drawn once. These are
the two questions it sets up and does not answer.

| a car costs | value at the start | over no van | a day | states that move | largest move |
| ---: | ---: | ---: | ---: | ---: | ---: |
| no van | 550.749 | - | - | 0 | 0 |
| 0 | 590.930 | +40.181 | +4.018 | 400 | 5 |
| 2 | 574.948 | +24.199 | **+2.420** | 171 | 5 |
| 5 | 560.120 | +9.371 | +0.937 | 75 | 5 |
| 10 | 550.749 | 0.000 | 0.000 | **0** | 0 |
| 20 | 550.749 | 0.000 | 0.000 | **0** | 0 |

*A discount of 0.9, so a gain of `g` a day is worth `g` over one tenth. Every
row is a sweep of the model and no seed appears anywhere. `states that move`
is out of 441.*

**At the book's price the van is worth 2.42 a day.** It moves cars in 171 of
the 441 states, and the most it ever moves is the five it holds.

**A free van is worth 4.02 a day and uses almost the whole board.** Four
hundred of the 441 states move a car when moving is free, so most of the
restraint at a price of 2 is the price rather than the layout.

**At 10 a car it is never used.** The value falls back exactly onto the row
with no van, and nothing moves anywhere. Somewhere between 5 and 10 the van
stops being worth its own cost, and the count of states that move is what says
so: it is 75 at a price of 5 and nothing at 10.

The control row is a van that holds nothing, which leaves one action that moves
nothing. Without it every other row would be read against nothing.

---

## The smallest approximation there is

```console
$ python scripts/measure_aggregation.py
$ rel train linear-td --env long-walk --set groups=50
```

Put the states into groups and keep one number for each group. That is state
aggregation, and it is the whole of function approximation with everything
optional taken out: no features to design, no widths to choose, nothing but a
count.

`long-walk` is a thousand cells in a line, and a step covers up to a hundred of
them either way. A table there has a thousand rows and learns nothing about one
cell from standing in the next. Fifty groups is fifty numbers, and every cell
in a group shares all of its learning with the other nineteen.

### The floor is arithmetic

What a staircase of `n` steps can say about a line is limited before any
learning happens, and by how much is arithmetic: the best each step can be is
the mean of the true values under it, and what is left is the error no agent
can remove.

| groups | best a staircase can do | what linear-td reached |
| ---: | ---: | ---: |
| 1 | 0.2717 | 0.3431 |
| 2 | 0.1356 | 0.2637 |
| 5 | 0.0539 | 0.1156 |
| 10 | 0.0267 | 0.0619 |
| 20 | 0.0134 | 0.0436 |
| 50 | 0.0053 | **0.0301** |
| 100 | **0.0027** | 0.0442 |

*Five seeds, a thousand episodes, step size 0.2. The error is the root mean
square against the true values, over the thousand cells an agent can be in.*

**The two columns have their best in different places.** The floor falls all
the way and halves with every doubling, because a staircase of twice as many
steps is twice as close to a line. What the agent reaches falls to fifty
groups and turns back up, because the same thousand episodes are spread over
twice as many numbers to learn.

The gap between the columns is the learning that has not finished, and it grows
as the groups do. At one group the agent is within a quarter of the floor. At a
hundred it is sixteen times it, and almost all of what it is wrong about is
unfinished rather than unreachable.

### Where the true values come from

Not from a formula. The walk with a stride of one has a closed form, `k` over
the number of gaps, and a stride of a hundred does not: a step passes an ending
rather than landing on it, and that overshoot is exactly what the fair game
argument leaves out. Cell 1000 of a thousand is worth 0.961 rather than the
0.999 a straight line would give it.

So the reference is a sweep of the model under the policy the walk is run at,
which `rel.agents.dp.evaluate_shares` does. On the small walk that sweep is
equal to the closed form at every size, and the script prints the two side by
side, because a reference nothing checks is a reference nobody should trust.

---

## The tile coder

A tile coder lays several grids over a continuous space, each shifted by a
fraction of a cell, so two nearby points share most of the switches they turn
on. The rule from the literature shifts each grid by an odd number of units,
one odd number per dimension.

```console
$ python scripts/measure_tiling_offsets.py
```

| dimensions | offsets | mean shared | spread |
| ---: | --- | ---: | ---: |
| 2 | odd displacement | 1.79 | **0.60** |
| 2 | same shift | 1.78 | 0.90 |
| 4 | odd displacement | 1.43 | **0.97** |
| 4 | same shift | 1.40 | 1.13 |

A smaller spread means the generalisation depends less on which direction two
points are apart in, which is what the rule is for. It does that in both.

### That measurement said the opposite first, and the code was wrong

The first time this ran, four dimensions came out at a spread of **2.53** for
the odd rule against 1.13 for the same shift: the rule made things worse, by a
lot.

The rule was fine. This project's tile coder was not.

The shift was not being taken modulo one cell. The last grid in the last
dimension is displaced by seven times seven over eight, which is more than six
whole cells, and the cells past the end of the space allocated for them did not
exist. Every point out there was clamped into one cell. Measured over a four
dimensional box, from two hundred thousand random points, the number of cells
each grid could reach ran:

```
4096, 6477, 5166, 4015, 3024, 2155, 1510, 945      out of 6561
```

The last grid had lost six sevenths of itself. Nothing about a run said so. The
agents still learned, and they learned with a coder that had quietly thrown
away most of its resolution.

After taking the shift modulo one cell:

```
4096, 6477, 6515, 6465, 6560, 6469, 6526, 6477     out of 6561
```

The count is of cells that at least one drawn point landed in, so it rises
with the number of points. `python scripts/measure_tiling_offsets.py` prints
both rows and says how many it drew.

It was found by a mutation test. A test that removed the clipping from the
input still passed, which meant the clipping was doing nothing and something
else was catching those points. `TestEveryGridKeepsItsResolution` in
`tests/test_tiles.py` is the regression test, and it is exhaustive in two
dimensions rather than sampled.

The moral is not that measuring is good. It is that **a measurement that
disagrees with the literature is a place to look at your own code first**, and
that a claim in a docstring should say what was measured rather than what is
usually said.

---

## A second way to make features, and the one number that decides it

A tile coder answers "which cells is this point in", and every answer is a
switch: on or off. Radial basis features answer "how close is this point to
each of these places", and every answer is a number between zero and one.

The difference is a step against a slope. Two points on opposite sides of a
tile boundary share nothing along that dimension however close together they
are. A radial basis has no boundaries: a point a hair away has a value a hair
different, everywhere.

`rbf-sarsa` and `rbf-q` are the same agents as `tile-sarsa` and `tile-q`, over
the other encoder. Nothing about `LinearAgent` knows which it is talking to.

```console
$ python scripts/measure_approximation.py --runs 10
$ python scripts/measure_approximation.py --env cartpole --runs 8 --bins 4
```

### The count of features says the opposite of the cost

<!-- not checked: the microseconds belong to the machine, and the ratio
between the two encoders is what the section is about -->
| environment | agent | features | us/step |
| --- | --- | ---: | ---: |
| mountain car | tile-sarsa | 648 | **52** |
| mountain car | rbf-sarsa | 36 | 120 |
| mountain car | rbf-sarsa kept=8 | 36 | 106 |
| cart pole | tile-sarsa | 52,488 | **70** |
| cart pole | rbf-sarsa | 1,296 | 3511 |
| cart pole | rbf-sarsa kept=8 | 1,296 | 3073 |

On the cart pole the tile coder has forty times more features and costs a
fiftieth as much per step. Reading the feature counts alone gives the opposite
answer to reading the clock, which is why both columns are here.

The reason is not that a tile coder is a better shape. It is that **a tile
coder never asks a feature that is off.** It works out which eight switches are
on directly, by arithmetic on the coordinates, and never touches the other
52,480. A radial basis has no such route: every centre answers every point.

<!-- not checked: the microseconds belong to the machine -->
| a step of a four dimensional radial basis, 1296 centres | us |
| --- | ---: |
| the distance to all 1296 centres | 755 |
| and the exponential of each | 861 |
| and normalising them | 1042 |
| finding the largest 8 by sorting | 132 |
| finding the largest 8 by a heap | 84 |

### Dropping the far centres cannot fix it

`kept` drops all but the largest few values, and it was going to be the
default, and it was going to buy back exactly the sparse shape the tile coder
has. It buys eleven percent of a step on the mountain car and thirteen on the
cart pole, and both move by a point between runs of the same machine.

The table above says why. Three quarters of the step is the distances, and they
are paid before anything can be dropped, because **there is no way to know
which centres are far away without measuring the distance to all of them.** The
sort that then finds the largest eight costs about what dropping the other 1288
saves. A heap is faster than the sort and not by enough to matter.

It also costs the thing the encoder exists for. Between two centres there is a
point where the smallest kept value and the largest dropped one cross, and
either side of it the agent reads a different weight for a feature of the same
size. The jump there is exactly the size of the smallest kept value, and how
large that is depends on where in the box the crossing is: over the whole unit
square it runs from 0.006 to 0.054, with a middle of 0.030. At the crossing the
test measures it is 0.036, where the largest feature of that same point is
0.214.
That is smaller than a tile coder's boundary, which swaps a whole switch of
eight, and it is the same kind of thing.

So `kept` is off by default. `TestKeepingBringsTheBoundaryBack` in
`tests/test_basis.py` is the measurement, and it found the boundary by sweeping
a line across the box and looking for a step, which is the only way it would
have been found: every other test of the encoder passed.

### On the mountain car it learns better, at twice the cost

| agent | mean | every seed |
| --- | ---: | --- |
| tile-sarsa | -164.6 | -170 -157 -166 -136 -158 -172 -178 -163 -173 -173 |
| rbf-sarsa | **-143.3** | -131 -152 -137 -141 -153 -138 -153 -136 -160 -134 |

Tile coding minus radial basis: -21.3, 95 percent interval [-30.4, -11.5],
p 0.008. Both halves of the answer agree, over ten seeds.

Read it as an early learning result and not as a verdict. Sixty episodes is
what these runs get, and a radial basis spreads one reward much further than a
tile does. A tile here is an eighth of the box wide and a point outside it gets
nothing. A centre is still worth 0.64 at a seventh of the box away, 0.25 at a
quarter and 0.08 at a third, so the first time a run reaches the flag, far more
of the space hears about it. The table further up this page runs
`tile-sarsa` for 300 episodes and it reaches -114.2, which is better than
either row here.

### The width is the whole of the setting

The width is a multiple of the spacing between the centres.

| width | mean | every seed |
| ---: | ---: | --- |
| 0.5 | -150.8 | -161 -159 -138 -143 -142 -139 -145 -165 -167 -150 |
| 0.75 | **-143.3** | -131 -152 -137 -141 -153 -138 -153 -136 -160 -134 |
| 1 | -150.0 | -148 -148 -145 -160 -157 -138 -152 -147 -160 -144 |
| 1.5 | -207.6 | -169 -142 -371 -197 -149 -175 -214 -314 -172 -174 |
| 2 | -842.2 | -1000 -1000 -1000 -173 -1000 -1000 -1000 -1000 -871 -379 |
| 3 | -509.5 | -1000 -318 -456 -337 -476 -461 -1000 -279 -308 -459 |

At two spacings, eight of ten seeds never reach the flag. Every centre answers
about the same at that width, so the features say almost nothing about where
the point is, and the agent is learning one number for the whole box.

One whole spacing is the value that looks right and it is not the best one. It
loses to three quarters on the mountain car by 11.2 over twelve seeds, interval
[+4.3, +20.1], p 0.010, and on the cart pole by 267.6 over eight seeds,
interval [+222.1, +304.1], p 0.008. Half a spacing beats one on both and does
not beat three quarters on either. So the default is three quarters, and
`TestTheDefaultWidth` in `tests/test_basis.py` holds it there with the reason
written next to it.

Both come out of the second command above, which prints every width against
the default under the table of means. The first version of this paragraph
quoted an interval and a p value that no command on this page produced, and
its cart pole figure was 183.7 against the 267.6 the command really prints.

### Where it is weak

- **One environment for the learning comparison.** The cart pole at six
  centres a side is 1296 of them and about an hour of runs, so it is swept at
  `bins=4`, which is 256 and ten minutes. The comparison of the two encoders
  is the mountain car only.
- **Sixty episodes.** The comparison is of early learning. Which encoder is
  ahead after three hundred episodes is not measured here.
- **The width was swept on a grid of six values.** The best of six is not the
  best there is, and the sweep is coarse near the top: 0.5, 0.75 and 1 sit
  within 8 points of each other while 1.5 and above fall off a cliff.
- **The timing is one machine, one Python.** The ratio between the two
  encoders is the part worth reading, not the microseconds.

---

## Learning against an answer that is known

```console
$ python scripts/measure_blackjack.py
$ rel solve --env blackjack
```

Blackjack is where the book introduces Monte Carlo, because the dealer plays
out his whole hand and that is awkward to write as a one step model. This
project writes it out anyway: the dealer's ending is a recursion over the cards
he draws while under seventeen and the deal is a recursion over the cards the
player takes to reach twelve, both exact.

**Which is the point of having it.** Solved, the optimal policy and the value
of the deal are known before a card is dealt, so an agent that learns from
episodes is scored against the answer rather than against another agent.

The exact solution reproduces the book's figure square for square: stick on
hard 17 against a seven or better, 13 against a two or three, 12 against a four
to a six, and on soft 19 against a nine, a ten or an ace and 18 otherwise. That
is the check that the model is right rather than merely self consistent,
because a wrong model can add up to one everywhere.

### What a fixed step size buys, and where it stops buying it

`monte-carlo` takes a running average with no step size given and a fixed
weight on the newest return with one. Its docstring claimed the fixed weight is
what control needs even in a fixed environment, because the policy keeps
changing. Here is that claim against an exact answer.

| episodes | step size | what it learned is worth | squares apart | what those squares are worth |
| ---: | ---: | ---: | ---: | ---: |
| 50,000 | running average | -0.06258 | 31.0 | 6.54 |
| 50,000 | 0.01 | -0.06557 | 30.2 | 6.65 |
| 50,000 | 0.05 | **-0.05994** | 25.6 | 5.14 |
| 50,000 | 0.1 | -0.06438 | 30.0 | 5.30 |
| 200,000 | running average | **-0.05210** | 16.2 | 2.55 |
| 200,000 | 0.01 | -0.05341 | 16.2 | 2.66 |
| 200,000 | 0.05 | -0.05865 | 20.6 | 3.23 |
| 200,000 | 0.1 | -0.06852 | 29.2 | 4.18 |
| 500,000 | running average | **-0.04841** | 8.6 | 0.99 |
| 500,000 | 0.01 | -0.05227 | 9.8 | 0.90 |
| 500,000 | 0.05 | -0.06007 | 19.0 | 2.35 |
| 500,000 | 0.1 | -0.06963 | 27.4 | 4.00 |

*Five seeds, epsilon 0.1. Perfect play is worth -0.04656. `what it learned is
worth` is the exact value of the greedy policy over the same model, not the
return the agent collected.*

**The claim holds early and fails later.** A fixed 0.05 wins at fifty thousand
hands. By five hundred thousand the running average is ahead of every fixed
step, and the larger the step the further behind it is, which is what a weight
that never falls does with more data.

The default stays a fixed 0.1, and the docstring says why: the grids this
project measures on run for hundreds of episodes rather than hundreds of
thousands, which is the end of that table where forgetting pays.

**A count of mistakes is not their size.** At five hundred thousand hands the
fixed 0.01 plays *more* squares differently from the optimum than the running
average does, 9.8 against 8.6, and gives up *less* value doing it, 0.90 against
0.99. The squares an agent gets wrong are the ones it rarely reaches, so both
columns are printed.

**Blackjack decides nearly every square, which is why counting them means
something here.** The smallest gap between the two actions anywhere on the
board is 0.0025 and the middle one is 0.32. The gambler's problem above decides
none of its capitals, and a count of squares there would be a count of a
tie breaking rule.

### Where it is weak

- **A natural pays nothing extra.** The book pays 21 on the first two cards
  before the hand is played. Here it is a hand of 21 like any other, which
  changes what the deal is worth and not what to do, because the only move at
  21 is to stick either way.
- **One agent.** `monte-carlo` only. Whether a temporal difference agent gets
  closer on the same hands is not measured here.
- **Five seeds.** Enough to order the rows and not enough for a p below
  0.0625.

---

## Part sample and part expectation

```console
$ python scripts/measure_sigma.py
$ rel train q-sigma --env cliff --set sigma=0.5
```

`tree-backup` takes the expectation over the target policy at every step of its
window and n step SARSA takes the one action really taken and corrects for
having taken it. `q-sigma` is the family between them: sigma of nothing is tree
backup exactly, sigma of one is the sample with a control variate, and the book
raises the middle as a question and leaves it open.

| sigma | at step | greedy, exactly | over tree backup | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.05 | **-13.800** | - | - | - |
| 0.25 | 0.05 | -14.400 | -0.600 | [-1.600, +0.400] | 0.453 |
| 0.5 | 0.05 | -14.400 | -0.600 | [-1.400, +0.400] | 0.453 |
| 0.75 | 0.05 | -15.400 | -1.600 | [-2.400, -0.800] | 0.016 |
| 1 | 0.1 | -14.800 | -1.000 | [-1.600, -0.400] | 0.062 |
| 1 falling to 0 | 0.05 | -14.400 | -0.600 | [-1.400, +0.200] | 0.375 |

*Cliff walk, 400 episodes, ten seeds, n of 3, epsilon 0.1. Every sigma is swept
over four step sizes and read at its own best. The best possible return is
-13.*

**The middle does not beat the ends here. The end this project already had
does.** Tree backup is ahead of every other row, and the two rows furthest from
it are the two clear of zero.

The schedule the book suggests, sigma falling from one to nothing over the run,
lands with the middle rather than with either end.

**The collapse to tree backup is exact and is tested.** Against a greedy target
the two agents agree bit for bit, because a greedy target has shares of one and
nothing so nothing is rearranged between the two ways of writing the step.
Against an averaged target they agree to the last bits: one sums the actions
not taken and adds the taken one separately, the other sums all of them and
subtracts the taken one back out, and float addition is not associative.

That collapse is also what caught a fault. The first version asked the target
policy for its shares three times inside one update, and a greedy target breaks
ties by drawing, so the three answers could disagree and each spent randomness.
It did not reproduce tree backup at sigma of nothing, which is how that was
found.

### The window decides whether sigma can matter at all

```console
$ python scripts/measure_sigma.py --steps 1 3 5 10
```

The table above is at a window of three. Here is the same table at one, five
and ten, and the first of them answers the question before the measurement
does.

**At a window of one, sigma cannot matter, and that is arithmetic.** The n-step
return is built from its tail and its base case is `G(h, h) = Q(S(h), A(h))`.
At one step the base case is the only level, so the term sigma multiplies is
that value minus itself. The target is the reward plus the expectation over the
target policy, at every sigma. One step Q(sigma) is expected SARSA whatever it
is asked for, and `tests/test_tree.py` holds that cell for cell.

| sigma | at step | greedy, exactly | over tree backup | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1 | **-13.000** | - | - | - |
| 0.25 | 0.1 | **-13.000** | +0.000 | [+0.000, +0.000] | 1.000 |
| 0.5 | 0.1 | **-13.000** | +0.000 | [+0.000, +0.000] | 1.000 |
| 0.75 | 0.1 | **-13.000** | +0.000 | [+0.000, +0.000] | 1.000 |
| 1 | 0.1 | **-13.000** | +0.000 | [+0.000, +0.000] | 1.000 |
| 1 falling to 0 | 0.1 | **-13.000** | +0.000 | [+0.000, +0.000] | 1.000 |

*Cliff walk at a window of one. Six rows identical to the last digit,
and every one of them at the best possible return.*

So a reader who sees that table is entitled to wonder whether sigma was dropped
somewhere. It was not. It has nothing to multiply.

| sigma | at step | greedy, exactly | over tree backup | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.05 | -14.800 | - | - | - |
| 0.25 | 0.05 | **-14.200** | +0.600 | [-0.200, +1.400] | 0.375 |
| 0.5 | 0.05 | -14.800 | +0.000 | [-0.600, +0.600] | 1.000 |
| 0.75 | 0.05 | -15.000 | -0.200 | [-1.000, +0.600] | 1.000 |
| 1 | 0.1 | -15.000 | -0.200 | [-0.800, +0.400] | 1.000 |
| 1 falling to 0 | 0.05 | -14.600 | +0.200 | [-0.600, +1.000] | 1.000 |

*Cliff walk at a window of five. Nothing here is decided: every interval
crosses zero and the smallest p is 0.375.*

| sigma | at step | greedy, exactly | over tree backup | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.05 | **-14.200** | - | - | - |
| 0.25 | 0.05 | -15.200 | -1.000 | [-2.400, +0.400] | 0.281 |
| 0.5 | 0.2 | -16.000 | -1.800 | [-3.000, -0.600] | **0.062** |
| 0.75 | 0.1 | -16.000 | -1.800 | [-2.800, -0.600] | **0.031** |
| 1 | 0.2 | -16.000 | -1.800 | [-3.000, -0.600] | **0.047** |
| 1 falling to 0 | 0.4 | -15.600 | -1.400 | [-2.800, +0.200] | 0.172 |

*Cliff walk at a window of ten. Three rows clear of zero, and all three
are worse than tree backup.*

### What the four windows say together

<!-- not checked: one row taken from each of the four tables above, and the
last column is read off their intervals rather than printed by anything -->

| n | best row | tree backup | rows whose interval is clear of zero |
| ---: | ---: | ---: | ---: |
| 1 | -13.000 | -13.000 | none, and none is possible |
| 3 | -13.800 | -13.800 | 2 |
| 5 | -14.200 | -14.800 | 0 |
| 10 | -14.200 | -14.200 | 3 |

**Sigma matters more at a longer window, and every time it is decided it is
decided against sampling.** At one step it cannot matter. At three, two rows
are clear of zero and both are below tree backup. At five nothing is decided.
At ten, three rows are clear and all three are 1.8 below it.

**And the window itself matters more than sigma does.** Tree backup at one step
reaches the optimal -13.000 on all ten seeds. At three it reaches -13.800, at
five -14.800, at ten -14.200. The whole spread across sigma at any window is
smaller than the spread across the window at sigma of nothing.

That is worth saying because the family is presented as a question about sigma.
On this grid the setting that decides the answer is the one the family holds
fixed.

### Where it is weak

- **One grid and one budget.** The cliff walk at 400 episodes. The maze is not
  measured here, and `--env maze` now runs it.
- **The score is coarse.** A cliff walk policy is worth -13 along the edge, -15
  one row up and -17 two rows up and little else, so a mean over ten seeds moves
  in steps of a fifth. That is why the interval is printed rather than the mean
  alone.
- **Most rows are read at the edge of the sweep rather than at a bracketed
  best.** The four step sizes are 0.05, 0.1, 0.2 and 0.4. At a window of three
  and of five, five of the six rows chose 0.05, which is the smallest offered,
  so what those rows would do at a smaller step is not known. At ten, three of
  six sit at an end. Only the window of one is clear of it, where every row
  chose 0.1 and every row reached the optimum anyway.

---

## Waves over the whole box

```console
$ python scripts/measure_fourier.py --runs 10 --step-sizes 0.02 0.05 0.1 0.2 0.5 1.0 2.0
$ rel train fourier-sarsa --env mountaincar --set order=5
```

A tile coder cuts the box into cells and a radial basis puts bumps at places.
Both are local: a point lights the features near it and nothing else. A Fourier
basis is the opposite. Every feature is a cosine wave over the whole box, and
what tells them apart is how fast each one waves.

It costs `(order + 1)` to the power of the dimensions, which is the growth a
radial basis has and worse than a tile coder's. What it does not need is
anything else: no bins, no widths, no centres, no offsets between grids. **An
order is the whole design**, and that is the reason to have it beside two
encoders whose settings a reader has to get right.

### The one setting the literature attaches, measured both ways

A wave that crosses the box eight times moves the value eight times as often
for the same change to its weight, so the usual advice is to divide the step
size for each feature by the length of its own coefficient. That is the only
thing said about this basis that is not the basis itself.

| order | features | scaled | at step | flat | at step | scaled minus flat | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | -147.0 | 0.05 | **-137.6** | 0.1 | -9.4 | [-13.0, -5.6] | 0.004 |
| 3 | 16 | **-124.4** | 0.02 | -170.3 | 0.5 | +46.0 | [+25.9, +67.0] | 0.004 |
| 5 | 36 | -154.4 | 0.5 | **-134.7** | 0.5 | -19.7 | [-31.3, -9.6] | 0.006 |
| 7 | 64 | **-131.2** | 0.1 | -140.1 | 1 | +8.9 | [-5.2, +20.5] | 0.213 |

*Mountain car, ten seeds, 200 episodes, greedy return afterwards. Both sides
swept over seven step sizes and each read at its own best.*

**The sign flips.** Scaling loses at order one, wins at order three, loses at
order five and is not told apart at order seven. Three of those four intervals
are clear of zero at a p of about 0.005, so each is a real difference on this
problem, and no rule that says "divide the step for each feature" survives all
four rows.

**Both sides have to be swept or the answer is the other question.** Every
scale a Fourier basis asks for is at most one, so scaling makes each step
smaller as well as uneven. The first probe of this ran both at one step size,
found the scaled side far ahead at order five, and would have reported smaller
steps as uneven steps. Swept, the flat side is ahead there by 19.7.

The two sides want very different step sizes. At order three the scaled side is
best at 0.02 and the flat side at 0.5, twenty five times apart, which is what
the scaling does and why a shared step size cannot compare them.

**One row is a floor rather than a best.** At order three the scaled side wants
the smallest step the sweep offered, so its 46.0 is what scaling bought inside
the range rather than what it can buy. The script names such rows rather than
letting them read as findings.

### The cart pole cannot answer this, and that is the answer

```console
$ python scripts/measure_fourier.py --env cartpole --orders 1 2
$ python scripts/measure_fourier.py --env cartpole --orders 1 2 --episodes 50
$ python scripts/measure_fourier.py --env cartpole --orders 1 --episodes 50 --runs 12
```

The table above is the mountain car. The obvious next question is whether the
sign still flips somewhere else, and the cart pole is the other box this
project has.

**It is four dimensions rather than two, and that decides how far the question
can be asked.** A Fourier basis is `(order + 1)` to the power of the
dimensions, so the same order is a very different number of waves.

<!-- not checked: (order + 1) to the power of the dimensions, worked out
rather than run. Orders 3, 5 and 7 on the cart pole are the ones no command
here reaches, which is the point the table is making -->

| order | features on the mountain car | features on the cart pole |
| ---: | ---: | ---: |
| 1 | 4 | 16 |
| 3 | 16 | 256 |
| 5 | 36 | 1296 |
| 7 | 64 | 4096 |

*The growth this basis has, and the reason the section above says it does not
go past a few dimensions.*

Orders 1, 2 and 3 were run first. It did not finish in an hour and printed
nothing, so the run below is orders 1 and 2.

| order | features | scaled | at step | flat | at step | scaled minus flat | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 500.0 | 0.05 | 500.0 | 0.2 | +0.0 | [+0.0, +0.0] | 1.000 |
| 2 | 81 | 500.0 | 0.1 | 500.0 | 0.05 | +0.0 | [+0.0, +0.0] | 1.000 |

*Cart pole, five seeds, 200 episodes. The best possible return is 500.*

**Both sides are on the ceiling.** 500 is the cap, both orders reach it, and
the difference is exactly nothing. The comparison has no room to say anything,
because a Fourier basis solves this problem.

A quarter of the budget does not open it either:

| order | features | scaled | at step | flat | at step | scaled minus flat | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 500.0 | 0.2 | 482.6 | 0.2 | +17.4 | [+0.0, +52.3] | 1.000 |
| 2 | 81 | 500.0 | 2 | 500.0 | 0.2 | +0.0 | [+0.0, +0.0] | 1.000 |

*Cart pole, five seeds, 50 episodes.*

### The one row with a difference in it was five seeds talking

The row above at order 1 is the only one on the cart pole where the two sides
differ at all: scaling ahead by 17.4, on an interval that touches zero at the
bottom and a p of 1.000. Twelve seeds instead of five:

| order | features | scaled | at step | flat | at step | scaled minus flat | 95 percent interval | p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 460.1 | 0.5 | 480.4 | 0.5 | **-20.3** | [-103.5, +45.4] | 0.750 |

*Cart pole, twelve seeds, 50 episodes.*

**The sign reversed.** Ahead by 17.4 at five seeds and behind by 20.3 at
twelve, and neither is decided. Both sides also chose a different step size
once there were more seeds to choose on.

That is this page's own rule about seeds, arriving with a number attached. Five
seeds cannot report a p below 0.0625 whatever the difference is, and here they
could not report the direction either.

### So the question is still open, and it is open for a reason

Whether the step scaling rule's sign flips on another environment is not
answered here. The cart pole cannot answer it: a Fourier basis reaches the cap
on it at both orders that finish and at a quarter of the budget, so there is
nothing left for a step size rule to be better or worse at.

**The mountain car answers it because neither side solves it.** That is what
makes it the environment this comparison runs on, and it was not chosen for
that reason.

### Where it is weak

- **Still one environment that can say anything.** The cart pole was measured
  and is on the ceiling, so the sign flip is known on the mountain car and
  nowhere else. A third box with room in it is what this needs.
- **Two hundred episodes.** This is early learning, as the encoder comparison
  above it is.
- **Four orders.** The sign flips twice across four rows, which is enough to
  say the rule does not hold and not enough to say what does.
- **The order is the whole design and it is not swept against the others.**
  Which of the three encoders is best on this problem is a different question
  and this table does not answer it.
- **The cart pole rows are five seeds.** The one that was rerun at twelve
  changed sign, so the other three should be read as saying that both sides
  reach the cap and nothing more.

---

## The two control problems

Neither has a table of states and neither has a model, so there is no exact
value to compare against. What is reported is the greedy return of every seed,
because these agents vary far more than the tabular ones and a mean on its own
would hide it.

```console
$ python scripts/measure_control.py --env mountaincar --episodes 300
$ python scripts/measure_control.py --env cartpole --episodes 600
```

**Mountain car**, 300 episodes, greedy return afterwards:

| agent | mean | each seed |
| --- | ---: | --- |
| random | -1000.0 | -1000 -1000 -1000 -1000 -1000 |
| tile-sarsa | **-114.2** | -129 -109 -122 -105 -105 |
| tile-q | -123.9 | -126 -106 -142 -129 -117 |
| reinforce | -938.5 | -921 -1000 -1000 -771 -1000 |
| actor-critic | -1000.0 | -1000 -1000 -1000 -1000 -1000 |

The random policy never leaves the valley in a thousand steps, on any seed.
That is the clearest separation in the project between an agent that learned
and one that did not: the engine cannot climb the hill, so nothing but a policy
that rocks the car gets out at all.

The last two rows are the other end of that. The actor critic scores what the
random policy scores, on every seed. REINFORCE gets out once in five. This is
the environment the two of them are worst on, and it is worth saying why: the
reward is -1 a step until the goal, so an agent that never reaches the goal
sees one flat number for a thousand steps and has nothing to raise the
probability of. A tile coded value function does not need to reach the goal to
learn something, because a state it has left is worth what the state it moved
to is worth. A policy gradient does.

That is a claim about what these two methods need rather than about these
implementations, and the cart pole below is the same two agents at the top of
the table.

**Cart pole**, 600 episodes, greedy return afterwards, out of a possible 500:

| agent | mean | each seed |
| --- | ---: | --- |
| random | 18.3 | 19 16 20 15 21 |
| tile-sarsa | 498.2 | 500 500 500 500 491 |
| tile-q | 418.8 | 500 500 500 500 94 |
| reinforce | **500.0** | 500 500 500 500 500 |
| actor-critic | 146.8 | 500 **8 8** 80 138 |

A return of 8 is a pole that fell over: the policy has gone entirely
deterministic and pushes the cart one way until it drops. Two of the five
numbers in the last row are that.

Three of these four rows solve the problem and one does not, and the one that
does not is the most recent method on the list. That is not a claim about
actor critic methods. It is a claim about this actor critic, and the sweep
that says so is below.

Both of the numbers in the last two rows moved a long way during one session.
REINFORCE was at 220.7 until the step limit fault above was fixed, and at
356.5 until its entropy default was measured rather than guessed. The actor
critic was at 8.4 until the same entropy default moved. A row of this table is
a statement about an implementation and its settings, and never about a method.

### The exploration setting on a tile coder

The mountain car chapter uses a greedy policy with no exploration at all, and
that is what the classes here default to. It is the wrong default for this
project, and the measurement says so plainly.

```console
$ python scripts/measure_control.py --env mountaincar --episodes 300 \
    --agents tile-sarsa tile-q --set epsilon=0.0
$ python scripts/measure_control.py --env mountaincar --episodes 300 \
    --agents tile-sarsa --set epsilon=0.01
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents tile-sarsa tile-q --set epsilon=0.0
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents tile-sarsa --set epsilon=0.01
$ python scripts/measure_control.py --env mountaincar --episodes 300
$ python scripts/measure_control.py --env cartpole --episodes 600
```

The last two are the registry default of 0.05, and they are the same two
commands as in `The two control problems` above. They are repeated here
because the rows they produce are in this table, and a block that named four
of the six commands behind a table would be a block that sent a reader to
look for the other two.

This block said `--env cartpole --agents tile-sarsa` for nineteen tracks. That
command takes the default of 300 episodes, so it produced no row of the table
under it, and there were no commands at all for the other nine.

Five seeds each, greedy return afterwards:

<!-- not checked, column epsilon: the cells hold the setting each row was run
at rather than anything a run produced -->
| environment | agent | epsilon | mean | each seed |
| --- | --- | ---: | ---: | --- |
| mountain car | tile-sarsa | 0.00 | -115.8 | -118 -132 -103 -120 -106 |
| mountain car | tile-sarsa | 0.01 | -137.0 | -137 -135 -129 -140 -143 |
| mountain car | tile-sarsa | 0.05 | **-114.2** | -129 -109 -122 -105 -105 |
| mountain car | tile-q | 0.00 | -127.9 | -123 -133 -119 -130 -134 |
| mountain car | tile-q | 0.05 | -123.9 | -126 -106 -142 -129 -117 |
| cart pole | tile-sarsa | 0.00 | 157.7 | 95 500 95 **8** 89 |
| cart pole | tile-sarsa | 0.01 | 432.6 | 500 500 500 163 500 |
| cart pole | tile-sarsa | 0.05 | **498.2** | 500 500 500 500 491 |
| cart pole | tile-q | 0.00 | 106.7 | **8** 500 **8** **8** **8** |
| cart pole | tile-q | 0.05 | 418.8 | 500 500 500 500 94 |

On the mountain car the setting hardly matters: every row is between -114 and
-137. On the cart pole it is the difference between an agent that solves the
problem on five seeds of five and one that falls over in eight steps on four of
them.

A return of 8 is a policy that has gone entirely deterministic and pushes the
cart one way until the pole drops. Without exploration a tile coded value
function has nothing to pull it back out.

So the registry default is 0.05: **a setting that solves one environment and
costs nothing on the other is a better default than the one from the chapter a
method came from.** The classes keep the chapter's default, because that is
what a reader comparing this code with the book needs to see.

---

## The three things that are safe in pairs

```console
$ python scripts/measure_triad.py
$ rel train linear-td --env baird --episodes 20
$ rel train gradient-td --env baird --episodes 20
```

Everything above this point on the page has used at most two of these three at
once:

    function approximation   more states than weights, so states share
    bootstrapping            a target built from an estimate, not a return
    off-policy               data from one policy, a question about another

Any two together are safe. All three are called the **deadly triad**, and this
project now has all three legs, so it should say what happens when they are
put together rather than leave a reader to assume it is fine.

### The counterexample

`baird` is seven states arranged so that nothing else can be blamed.

```
o o o o o o     dashed  goes to one of the six upper states, evenly
      o         solid   goes to the lower state
```

Every reward is zero, so every state is worth zero under either policy. There
are eight weights for seven states, laid out so the rows overlap:

    upper state i   worth 2 * w[i] + w[7]
    lower state     worth w[6] + 2 * w[7]

Eight weights of zero say the answer exactly. **The approximation is not the
limitation here.** The behaviour policy dashes six times out of seven and the
target policy always goes solid, so the data is nearly all about the upper
states and the question is about the lower one.

That mismatch is the third leg. Every state's estimate is made partly of one
shared weight, `w[7]`, and the lower state leans on it twice as hard as any
upper state does. The target of every update is the lower state's value, so an
update at an upper state moves the target it was aiming at.

On-policy that is not a problem: the states an update reaches are the states
the next data comes from, so the target being moved is a target that gets
measured again. Off-policy the data is nearly all upper states and the target
is about the lower one, and the correction never arrives.

### What that does

| agent | discount | median value error | every seed |
| --- | ---: | ---: | --- |
| linear-td | 0.99 | 1.327e+22 | 4.83e+21 6.08e+22 1.56e+22 8.32e+20 1.33e+22 |
| gradient-td | 0.99 | 1.927 | 1.93 1.93 1.93 1.93 1.93 |
| linear-td | 0.5 | 1.839e-15 | 1.3e-15 1.84e-15 1.84e-15 1.93e-15 1.69e-15 |
| gradient-td | 0.5 | 5.632e-15 | 5.63e-15 5.63e-15 7.78e-15 5.63e-15 1.1e-14 |

*Five seeds, twenty thousand steps each, step size 0.05. The true value is
zero at every state, so the error is the whole of what the agent believes.*

`linear-td` is semi-gradient TD over the features the environment hands out.
At 0.99 its estimates pass 1e22 and keep going. `gradient-td` is the same
update with one term added, from the same start over the same features under
the same two policies, and it stays at 1.9.

**The bottom two rows are the same three ingredients at a different discount.**
Both agents reach the answer to the last bits a float has. Nothing else about
those runs is different, and the section below is about why.

### It is not the step size

| step size | median value error of linear-td |
| ---: | ---: |
| 0.005 | 6584 |
| 0.02 | 9.128e+09 |
| 0.05 | 1.327e+22 |
| 0.2 | 2.493e+78 |

*The same runs at four step sizes.*

A divergence gets blamed on a step size often enough to be worth ruling out
rather than denying. **A smaller step size does not stop this, it slows it
down.** At 0.005 the estimates are past six thousand after twenty thousand
steps and still climbing.

### It is the discount, and there is a number

| discount | largest weight after the expected update |
| ---: | ---: |
| 0.5 | 6.769 |
| 0.8 | 6.769 |
| 0.85 | 6.771 |
| 0.88 | 9.104 |
| 0.9 | 2365 |
| 0.95 | 1.152e+08 |
| 0.99 | 1.567e+22 |

*Four thousand steps of the update the model says to make, which is
deterministic and has no seeds.*

The three ingredients are the same at every row. **Below a discount of about
0.88 they are stable and above it they are not**, and the whole of the
difference between the top of that table and the bottom is a number nobody
would think twice about choosing.

Bisecting on it gives 0.8840. `rel.envs.baird` gives the same crossing in
closed form, at `(9 + n) / (5 + 2n)` for `n` upper states, which is fifteen
seventeenths or **0.8824**. Nothing that measures reads the closed form, so
the two agreeing is two answers rather than one printed twice.

That is why the discount in the literature is 0.99 rather than a round number.

### More states make it worse

| upper states | closed form | measured |
| ---: | ---: | ---: |
| 4 | never | never |
| 5 | 0.9333 | 0.9369 |
| 6 | 0.8824 | 0.8840 |
| 8 | 0.8095 | 0.8073 |
| 10 | 0.7600 | 0.7574 |
| 20 | 0.6444 | 0.6466 |

*The crossing at six sizes of the same counterexample, worked out two ways.*

The crossing falls as the problem grows, towards a half. **Four upper states
or fewer never diverge at any discount below one**, and twenty of them diverge
at discounts that six of them are safe at. A counterexample with more places
to share a weight between is a counterexample that needs less help.

### What the correction buys, and what it does not

`gradient-td` is TDC. Semi-gradient TD follows the gradient of half the
squared error and drops the part that runs through the next state's value, and
off-policy that dropped part is what stops the correction coming back.
Estimating it in a second weight vector and subtracting it is the whole
method, and it costs one more vector and one more step size.

It does not merely stay bounded. The bottom row of the first table is this
agent below the crossing, and it reaches the answer.

At 0.99 it is still at 1.93 after twenty thousand steps, and
`python scripts/measure_triad.py --episodes 200` leaves it at 1.85 after two
hundred thousand. That is not a failure and it is worth knowing about. What is
left at that point is one direction, in which every state's estimate is the
same number, and the error a constant estimate creates is one minus the
discount times that constant. At 0.99 that is a hundredth of it, so the last
direction is removed a hundred times more slowly than at 0.5, and a run that
looks stuck is a run with one mode left.

---

## A value network, and the two pieces that make one work

```console
$ rel train deep-q --env cartpole --set step_size=0.02
$ python scripts/measure_value_network.py --env cartpole --episodes 400 --runs 10 --set step_size=0.02
$ python scripts/measure_value_network.py --env cliff --runs 10 --each-seed
```

`deep-q` is Q-learning with a network in place of the table. A table has one
number for each state and action, and moving one never moves another. A network
has one set of weights for all of them, so two things that are harmless with a
table become the reason a value network diverges.

**The steps arrive correlated.** Twenty steps of a pole falling to the left are
twenty samples of nearly the same thing, and a network fitted to them in a row
forgets what it knew about falling to the right. A **replay buffer** keeps the
last few thousand steps and learns from a batch drawn out of them, so one batch
mixes experience from far apart in time.

**The target moves with the estimate.** Q-learning aims at
`r + discount * max Q(s')`, and the same weights produce both sides, so every
step moves the thing being fitted and the thing it is fitted to. That is a
feedback loop rather than a regression. A **target network** is a second copy of
the weights, refreshed every so often, that the target is computed from.

Both are settings of one agent, so the four rows below are the same code with
two numbers changed. `replay=0` learns from the step just taken;
`target_refresh=0` takes the target from the live network.

### On the cart pole, neither piece alone does anything

| replay | target | median | worst | best |
| ---: | ---: | ---: | ---: | ---: |
| on | on | **66.3** | 15.6 | 88.8 |
| on | off | 9.0 | 8.8 | 16.2 |
| off | on | 9.4 | 8.6 | 9.8 |
| off | off | 8.8 | 8.4 | 8.9 |

*Ten seeds, 400 episodes, step size 0.02. The number is the mean episode length
over the last fifty, which on this environment is the return.*

A pole nobody is balancing falls in about nine steps. **So the bottom three
rows are all indistinguishable from not learning at all.** The two pieces are
not additive and neither is a partial answer: with either one missing this
agent is a random policy with extra arithmetic.

### On the cliff walk, they change how often it blows up

| replay | target | median | mean | worst | best |
| ---: | ---: | ---: | ---: | ---: | ---: |
| on | on | -55.2 | -103.1 | -545.2 | -29.7 |
| on | off | -49.6 | -57.2 | -129.5 | -36.1 |
| off | on | -55.6 | -104.4 | -310.9 | -20.0 |
| off | off | -55.3 | -239.1 | -664.3 | -24.6 |

*Ten seeds, 200 episodes. The best possible return is -13.*

**The four medians are the same number.** -55.2, -49.6, -55.6, -55.3. On the
typical seed of this grid the two pieces change nothing at all.

What they change is the tail. Every seed, in seed order:

```
on / on     -49.1  -46.6  -44.1  -70.1  -29.7  -79.1  -67.6  -545.2  -38.3  -61.4
on / off    -44.4  -47.9  -46.2  -52.4  -51.2 -129.5  -57.6   -47.0  -60.0  -36.1
off / on   -108.3  -35.8  -30.9  -91.5  -41.2  -40.1  -20.0  -310.9  -70.0 -295.3
off / off   -44.4 -242.9  -29.9  -61.9  -47.3 -664.3 -629.8   -24.6  -48.8 -597.0
```

With neither piece, four seeds of ten end past -240 and three of those past
-590. With both, one does. **The pieces are not making the agent better. They
are making it fail less often.**

### Why the two environments disagree

The cliff walk hands the network a one-hot vector: forty eight inputs, one of
them a 1. Each state then has close to its own parameters and a typical update
barely generalises, so the shared-weights problem hardly arises. What is left
is the occasional run where it does arise and the whole thing diverges.

The cart pole hands it four real numbers, and every update moves every
estimate. There is no typical run in which the problem does not arise, so
without the two answers there is no learning to speak of.

**The same ablation on two environments gives opposite-looking tables, and the
difference is the representation rather than the algorithm.**

### The mean would have said something true and useless

On the cliff walk the means rank the rows -57.2, -103.1, -104.4, -239.1, which
reads as a performance ordering and is a failure-rate ordering wearing its
clothes. The medians are all -55 and the rank comes entirely from how many runs
blew up.

Five seeds said something different again. Every mean moved by tens between
five seeds and ten, because a diverged run moves a mean further than the
distance between any two rows of this table. That is why
`scripts/measure_value_network.py` leads with the median, prints the worst
beside it, and takes `--each-seed`.

### What it costs

Every step is a forward pass for each member of the batch, plus one for the
target, plus a backward pass: `2 * batch + 1` passes through a network against
one dictionary lookup. On the cliff walk the four rows took between 298 and 999
seconds for ten seeds of 200 episodes, where every tabular agent on the same
grid takes about two.

### Where it is weak

**Nothing here solves the cart pole.** The best configuration reaches a median
of 66 steps of a possible 500, where `tile-sarsa` reaches 498. A value network
over a tile coder is not being compared with one over a small network here: the
tile coder wins, and this agent is the honest small version of a method whose
published results come from far more compute than pure Python has.

---

## Drawing from the buffer by priority

```console
$ python scripts/measure_prioritised.py
$ python scripts/measure_prioritised.py --skip-task
$ rel train deep-q --env cartpole --set priority=0.6 --set weighting=0.4
```

A buffer that draws evenly spends most of a batch on steps the agent already
predicts. `priority` draws a step in proportion to the size of the error the
agent last made on it, so the batch is spent where there is something left to
learn. Both are settings of the same buffer, so the rows below are one piece of
code with two numbers changed.

It also breaks the estimate, and that is the more interesting half. Fitting to
a batch is an average over the steps in it, and an average estimates what it is
meant to estimate only if the steps arrived with the right probability. Drawing
by priority changes those probabilities deliberately. The agent then settles
somewhere else. It is not a faster route to the same answer.

`weighting` is the correction: a step drawn `k` times more often than even
counts `k` to the power minus `weighting` as much, so at one the two effects
cancel.

### The bias is not an argument, it is a number

The first section of the script is not a task. One state, one action, and
twenty rewards with no future, so the target of every step is the reward itself
and the agent is fitting a constant to a fixed set of numbers.

That is worth doing because the answer is arithmetic. The constant that
minimises the mean squared error over a set of numbers is their mean. The
constant an uncorrected priority draw is pulled to instead is the root of

    sum over the numbers of |c - y| * (c - y) = 0

which for fifteen rewards of nothing and five of one is `1 / (1 + sqrt 3)`. The
script solves it by bisection rather than quoting it, and a test holds the
closed form against the bisection.

| setting | priority | weighting | settled at | off the mean | spread |
| :--- | ---: | ---: | ---: | ---: | ---: |
| even | 0 | 0 | 0.2395 | -0.0105 | 0.0734 |
| priority | 1 | 0 | 0.3442 | **+0.0942** | 0.0092 |
| corrected | 1 | 1 | 0.2389 | -0.0111 | 0.0354 |

*Five seeds, 400 passes each. The mean of the rewards is 0.2500 and the root
above is 0.3660.*

**The uncorrected row is about nine times further from the mean than either
other row, and its spread is the smallest of the three.** It is 0.0942 off
against 0.0105 and 0.0111, and it varies over 0.0092 where the even draw varies
over 0.0734 and the corrected one over 0.0354. It is not noisy. It has settled,
and it has settled in the wrong place.

The rewards are skewed on purpose. On a symmetric set the mean and the root are
the same number and this table would show nothing.

**It does not reach the root, and the reason is in the buffer.** Every step put
in is given the largest priority the buffer has held, whatever the agent now
believes about it, so each step is pulled back up once per pass and that pull
is towards the even draw. 0.3442 sits between the mean and 0.3660. The bias is
smaller than the argument says because of a choice made elsewhere for a
different reason, which is worth knowing before reading any single number here
as the size of the effect.

### On the cart pole it helps, and the correction helps more

<!-- not checked, column seconds: seconds belong to the machine -->

| setting | priority | weighting | return | seconds | minus even | 95 percent interval | p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| even | 0 | 0 | 24.2 | 32 | - | - | - |
| priority | 0.6 | 0 | 98.5 | 69 | +74.3 | [+0.3, +181.9] | 0.1230 |
| corrected | 0.6 | 0.4 | **187.8** | 98 | **+163.5** | [+51.7, +284.0] | **0.0254** |

*Cart pole, ten seeds, 150 episodes, read as the return of the greedy policy at
the end. The best possible return is 500.*

**Only the corrected row is decided.** Priority alone is ahead by 74 and its
interval touches zero at +0.3, which at ten seeds is not a result. Correcting
it as well is ahead by 163 with the interval clear.

That ordering is the useful one. The correction is not a tax paid to be
principled: on this task the setting that draws by priority and corrects for it
beats both the even draw and the uncorrected draw, and the uncorrected draw
beats nothing at a p anybody should act on.

### What it costs

A priority draw adds up the weights of the whole buffer once for each batch,
where an even draw asks the generator for a whole number and stops. That is two
thousand additions a step against eight, and the seconds above show it: 32
seconds becomes 69 and then 98. Half of that second step is the priority draw
and half is that the better rows survive longer episodes, which are more steps.

The alternative is a sum tree, which draws and updates in the log of the buffer
size rather than in the buffer size. It is the next section, and at the default
buffer of two thousand it is already ahead.

### Where it is weak

- **One task and one budget.** The cart pole at 150 episodes. What priority does
  on the cliff walk, where the one-hot representation makes each state nearly
  its own parameters, is not measured.
- **One pair of powers.** 0.6 and 0.4 on the task table, chosen before the
  measurement and not swept. Whether some other pair does better is open.
- **The seconds are not controlled for.** Every row got the same episodes and
  not the same time. A comparison at equal seconds would be a different table
  and might well be a different result.
- **Ten seeds.** Enough to decide the corrected row and not enough to decide
  the uncorrected one. The floor at ten seeds is a p of 0.002.

---

## Drawing in the log of the buffer

```console
$ python scripts/measure_tree.py
$ python scripts/measure_tree.py --skip-agent
$ rel train deep-q --env cartpole --set priority=0.6 --set tree=on
```

A priority draw needs the running totals of the weights. The scan builds them
from nothing once for each batch, so a buffer of two thousand costs two
thousand additions whether the batch is one step or eight, and a buffer twenty
times larger costs twenty times as much for the same eight steps.

A tree keeps those totals between batches. The leaves are the weights and every
other cell is the sum of the two below it, so the root is the total. A draw
walks down from the root, going left where the target fits inside the left
child and going right on the remainder, and a changed weight walks up from its
leaf mending one cell per level. Both are `log2(n)` where the scan is `n`.

`tree` is a setting on the same buffer, so the two sides below are one piece of
code with one flag changed.

### What one update costs

<!-- not checked, column scan microseconds,tree microseconds,scan over tree: timings belong to the machine, and so does a ratio of two of them -->

| buffer | scan microseconds | tree microseconds | scan over tree |
| ---: | ---: | ---: | ---: |
| 128 | 21.7 | 38.8 | 0.56 |
| 512 | 33.3 | 42.7 | 0.78 |
| 2000 | 82.1 | 49.6 | **1.66** |
| 8192 | 299.4 | 52.2 | 5.73 |
| 32768 | 1031.3 | 64.7 | 15.94 |
| 131072 | 4035.0 | 74.9 | 53.84 |

*One update is a batch of eight drawn and its errors put back. Each reading
averages over at least 200 updates and over more where the buffer is small.*

**The crossover is between 512 and 2000, and the default buffer is 2000.** So
the tree is already the cheaper draw at the setting `deep-q` ships with, by
two thirds, and the reason it is not the default is in the next table rather
than in this one.

The tree column is the shape to read. It goes from 38.8 to 74.9 while the
buffer goes up by a factor of a thousand: seven levels become seventeen, and
the cost roughly doubles. The scan column multiplies by 186 over the same
range.

Below 512 the tree loses, and it loses for a reason worth stating plainly. A
scan of 128 weights is one tight loop that Python runs quickly, and a descent
of seven levels is seven rounds of index arithmetic in the interpreter. The
tree wins on the count of operations from the start and it wins on the clock
only once the count is large enough to pay for what each one costs here.

### What an agent collects

<!-- not checked, column scan seconds,tree seconds,scan over tree: timings belong to the machine, and so does a ratio of two of them -->

| buffer | scan seconds | tree seconds | scan over tree | same digests |
| ---: | ---: | ---: | ---: | :--- |
| 2000 | 5.6 | 5.3 | 1.05 | yes |
| 32768 | 42.3 | 26.1 | **1.62** | yes |

*`deep-q` on the cart pole, 600 episodes, seed 1, priority 0.6.*

A draw is a small part of an update, because the batch it draws is then run
forward and backward through a network. So the agent collects less of the
saving than the update table promises: at the default buffer, 1.66 on the draw
becomes 1.05 on the run. At 32768 it becomes 1.62, which is most of an hour off
a day of sweeps.

**The two runs at each buffer have the same two digests.** That is the claim
that matters more than the seconds. The tree is a cost and not a behaviour, so
turning it on does not oblige a rerun of anything this page has written down.

### The digests agree, and it is not obvious that they should

Both structures add the same weights and neither adds them in the same order. A
scan accumulates left to right, so the running total before place `k` carries
one chain of `k` roundings. A tree adds in pairs, so the same total is
assembled from about `log2(k)` subtotals rounded separately. Two different sums
of the same numbers can straddle a target, and then one structure returns the
place before the boundary and the other returns the place after it, from one
random number.

That is not an argument that it never happens. The first section of the script
counts it exactly.

| buffer | share that disagree | one draw in | widest gap, last places | narrowest place over it |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 2.11e-14 | 4.7e+13 | 5 | 8.0e+10 |
| 1024 | 2.24e-13 | 4.5e+12 | 6 | 4.9e+09 |
| 4096 | 2.28e-12 | 4.4e+11 | 16 | 2.8e+07 |
| 8192 | 4.16e-12 | 2.4e+11 | 17 | 2.5e+07 |

*Weights drawn evenly from 1e-06, which is the floor a priority can be left
at, to 5, which is about the largest error a network makes early. Seed 5.*

**Nothing there is sampled.** For each boundary the script searches the bit
patterns of a double for the first target the tree sends past that place. The
distance from there to the running total the scan compares against is the exact
width of the band where the two differ, and the widths added and divided by the
total are the chance that one uniform draw disagrees. A sampling run would need
a hundred billion draws to see one, and would then have measured it to one
significant figure.

At 8192 the widest gap is 17 of the smallest steps a double can take, and the
narrowest place in the buffer is 25 million of those gaps wide. That is what
makes a disagreement, when one comes, a disagreement about neighbouring
places: two boundaries can only cross the same target where a place is
narrower than the gap, and no place is close.

**So a run of a million updates at the default buffer disagrees with
probability about two in a million million.** The digests match because they
were never going to do anything else, not because the two draws are the same
arithmetic.

### Why the default is still the scan

Because "about two in a million million" is not "never", and the recorded
digests on this page are the project's own check that it has not changed
underneath itself. A default that is right almost always is the wrong kind of
default for that job. `tree=on` is there for the buffer sizes where it is
worth turning on, and those are exactly the sizes nothing on this page uses.

### Where it is weak

- **One shape of weight.** Drawn evenly between the floor and 5. A real
  priority draw has most of its weight near the floor and a long tail, and how
  that changes the roundings is not measured.
- **One machine, one interpreter.** The crossover at 512 to 2000 is where the
  count of operations crosses the cost of an operation in CPython. A faster
  interpreter moves it left and there is no measurement of by how much.
- **The exact count stops at 8192.** Each boundary costs a search over bit
  patterns, so the ladder ends for time rather than because the answer settles.
  The trend over the four rows is between `n` and `n` and a half, which is
  enough to say the answer stays negligible and not enough to fit a law to.
- **The agent table is one seed.** It is a timing, not a result, and the
  digests either side of it are what carries the claim.

---

## The maximum overstates, and what splitting it takes off

```console
$ python scripts/measure_double.py
$ rel solve --env bias
$ rel train deep-q --env bias --set double=true
```

Q-learning backs up `max Q(s')`. That is the largest of several estimates, each
carrying error in both directions, and the largest of several noisy numbers is
above the largest true number. The error does not average out with more
samples of any one action, because it comes from the choosing.

`bias` is the smallest problem where that matters, Sutton and Barto's example
6.7. Going right ends the episode with nothing. Going left leads to a gamble
whose mean is -0.1 and whose spread is 1. `rel solve --env bias` says the best
possible return is 0, so an agent that goes left is wrong by a tenth, and no
horizon, representation or exploration problem is in the way.

**The number read is the share of episodes that went the wrong way,** which the
environment reports as an audit rather than paying for in reward. The return
would say far less: an episode that went left is worth -1.1 or +0.9 and one
that went right is worth nothing, so a mean return mixes the mistake with the
noise that caused it.

An agent that has learned the answer still goes left sometimes, because
epsilon-greedy explores. That floor is `epsilon / actions`, which is 0.010
here. Everything above it is the bias.

### Two tables against one

| agent | left, first 50 | last 200, median | mean | mean over the floor | late minus q-learning | 95 percent interval | p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q-learning | 0.484 | 0.013 | 0.071 | 7.1 | - | - | - |
| double-q | **0.146** | 0.010 | **0.011** | **1.1** | -0.060 | [-0.109, -0.017] | **0.0209** |

*Thirty seeds, 1000 episodes, epsilon 0.1, step size 0.1.*

**Over the first fifty episodes Q-learning goes the wrong way about half the
time and double Q-learning a seventh of the time.** That is the figure the book
draws and it reproduces.

**The late columns say something the book's figure does not.** The median seed
of both agents is at the floor by the end: 0.013 against 0.010. The means are
0.071 and 0.011. So the difference late is not that the typical Q-learning run
is worse. It is that a few Q-learning runs never settle, and a mean counts
those.

### Why a few runs never settle

Going right from the start is deterministic, ends the episode and pays nothing,
so `Q(A, right)` is exactly 0 and stays exactly 0 for the whole run. The
policy therefore turns on the sign of `Q(A, left)` against an exact zero.

The step size is a constant, so `max Q(B, a)` does not converge. It wanders. On
the worst of thirty seeds it swung between -0.03 and +0.27 over the last two
hundred episodes, and `Q(A, left)` followed it from -0.006 up to +0.26 and back
down. Every excursion above zero flips the policy back to going left.

**So the bias here is intermittent rather than persistent.** It is not that the
estimates settle in the wrong place. It is that they never settle, and one side
of the comparison is an exact zero that they keep crossing.

### Whether a longer run pays it off

| episodes | q-learning, median | mean | double-q, median | mean |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 0.0150 | 0.0238 | 0.0100 | 0.0123 |
| 1000 | 0.0100 | 0.0795 | 0.0100 | 0.0125 |
| 2000 | 0.0125 | 0.0315 | 0.0075 | 0.0103 |
| 4000 | 0.0100 | 0.0248 | 0.0050 | 0.0090 |
| 8000 | 0.0100 | 0.0210 | 0.0050 | 0.0083 |

*Twenty seeds at each budget, the share over the last 200 episodes of it. The
floor is 0.010.*

**No.** Sixteen times the budget takes the Q-learning mean from 0.0238 to
0.0210 and its median never leaves the floor. Both columns are flat, and they
are flat because they are measuring different things: the median seed was
finished by five hundred episodes, and the seeds in the mean are the ones that
are never finished.

Double Q-learning's median goes below the floor at the long budgets, to 0.0050.
That is not it doing better than the exploring policy allows. It is that a run
of two hundred episodes contains two exploring wrong turns on average, and half
of such runs contain one or none.

### On a network, neither one learns the answer

`deep-q --set double=true` makes the live network name the best action of the
next state and the target network say what it is worth. That costs nothing: a
target network is already a second set of weights kept apart from the live one.

| agent | left, first 50 | last 200, median | mean | mean over the floor | late minus deep-q | 95 percent interval | p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deep-q | 0.361 | 0.382 | 0.370 | 37.0 | - | - | - |
| deep-q double | **0.195** | **0.227** | **0.266** | **26.6** | -0.105 | [-0.225, +0.017] | 0.1111 |

*Thirty seeds, 1000 episodes, epsilon 0.1, step size 0.01.*

**Both rows are near the top and neither is near the floor.** The median is 37
and 27 times the exploration floor, so this is not a tail of bad seeds. It is
every seed. The split halves nothing here and the interval crosses zero.

The problem is two steps long, so a thousand episodes is about fifteen hundred
steps, and the buffer of two thousand never fills while the target refreshes
seven times. Whether the network is biased or simply has not been shown enough
is not separated by this table.

### On a task where the bias is not the point

| agent | return | minus deep-q | 95 percent interval | p |
| :--- | ---: | ---: | ---: | ---: |
| deep-q | **24.2** | - | - | - |
| deep-q double | 13.3 | -10.9 | [-20.7, -3.1] | **0.0078** |

*Cart pole, ten seeds, 150 episodes, read as the return of the greedy policy at
the end. The best possible return is 500.*

**Removing the bias makes it worse, and that is decided.** The interval is
clear of zero at ten seeds.

An overstated maximum is an optimistic value, and optimism about actions not
yet taken is what drives an epsilon-greedy agent to take them. On a problem
built so that optimism is the whole mistake, removing it is the whole answer.
On a problem where nothing is learned yet, removing it takes away a reason to
explore. **The correction is a correction, not an improvement.**

### Where it is weak

- **The cart pole row is two agents that both fail.** 24.2 and 13.3 out of 500.
  A difference between two failures is a real difference and it is not evidence
  about which is better where either one works.
- **One budget for the network.** 1000 episodes on a two step problem is not
  much for a network that carries a buffer and a target copy, and the section
  above says what that leaves undecided.
- **One step size each.** 0.1 for the tables and 0.01 for the networks, neither
  swept. The whole mechanism above turns on the size of the wander, which is
  proportional to the step size, so a sweep would move these numbers and is not
  done here.
- **Thirty seeds is not many for a tail.** The late means are counting how many
  runs of thirty never settled. That is a small integer wearing a decimal
  point.

---

## An action that is a number

```console
$ python scripts/measure_levels.py
$ rel train gaussian-actor-critic --env pendulum --episodes 200
$ rel demo tile-q --env pendulum-levels --episodes 300
```

Everything above this point takes an action from a short list. That covers
every environment on this page until now, and it leaves out half of control: a
torque, a steering angle and a throttle are numbers, and the list they would be
cut into is a choice somebody has to make.

`pendulum` is a weight on a rod and a motor too weak to lift it. The action is
a torque anywhere between minus two and two, nothing pays anything above zero,
and the only way to the top is to swing. **Held at full power from the bottom
the weight climbs to 0.659 of the way down and stops there**, which is the
angle where the motor balances gravity, so a policy that only pushes hard one
way cannot solve it. Full power in whichever direction the weight is already
moving does solve it, in two hundred steps, and that is a policy of one line.

### Cutting the box

`pendulum-levels` is the same environment with the torque cut into a short
list, so that every other agent here can run on it. Two levels is a switch,
nine is the registry default, and the count is a setting.

| levels | 300 episodes | 900 episodes |
| ---: | ---: | ---: |
| 2 | **-245.6** | -446.8 |
| 3 | -295.2 | -290.2 |
| 5 | -319.6 | -329.5 |
| 9 | -410.3 | **-229.4** |
| 17 | -816.2 | -335.3 |

*`tile-q`, five seeds, the mean of ten greedy episodes. Doing nothing at all
scores -1187 on this problem.*

**The two columns disagree about which count is best, and that is the answer.**
At three hundred episodes the switch wins and the ladder falls all the way to
-816.2 at seventeen levels. At nine hundred the middle wins, and seventeen
levels has come from -816.2 to -335.3 while the switch has gone the other way.

Every extra level is another column of the weight table to learn, so a small
budget is spent better on fewer of them. A large budget buys the finer control
that a switch cannot express. Which count is right is a question about the
budget rather than about the environment, and a page that gave one number here
would be describing its own budget.

Neither column is a settled number, and the switch going from -245.6 to -446.8
with three times the training says so. A constant step size does not converge,
it tracks, which this page says under prediction and which is as true of a
control agent over a tile coder as it is of a table. These are two points on a
curve that is still moving, not two answers.

### What learns this problem at all

| agent | actions | mean return |
| --- | ---: | ---: |
| random | 9 | -1190.5 |
| tile-q | 9 | **-410.3** |
| tile-sarsa | 9 | -624.1 |
| deep-q | 9 | -696.2 |
| actor-critic | 9 | -1384.8 |
| gaussian-actor-critic | the box | -1284.3 |

*Five seeds, three hundred episodes, the mean of ten greedy episodes. Every
row but the last acts on the cut version and the last acts on the box itself.*

The agents that keep a value and rank it learn this problem. The two that
learn a policy directly do not, and **both of them are behind acting at
random**: -1384.8 for the discrete actor critic on the cut version and -1284.3
for the continuous one on the box, against -1190.5 for a policy that draws.

That is worth reading carefully, because the obvious reading is wrong. The
continuous agent is not behind because its actions are numbers. Its discrete
sibling, on the same problem cut into nine levels, with the same engine and the
same optimiser, is further behind still. What the two share is a target
bootstrapped from a value network, and the values here run to several hundred
while the difference one action makes is about one. The advantage is the
difference of two large numbers that the critic is still wrong about.

The same reading is already on this page under the cart pole, where the actor
critic reaches 8 on two seeds of five and REINFORCE, which shares everything
but the target, reaches five hundred on all five. This is the second
environment to say it.

### The policy that needs no list

`gaussian-actor-critic` keeps no list. Its policy is a normal distribution for
each dimension of the action: a mean from a network reading the observation,
and a spread that is learned and the same everywhere. An action is a draw from
it, and the gradient of the log of its density is what a policy gradient needs.

The mean is squashed through a tanh onto the box. Without that it can walk out
of the box, every draw then clips to the same bound, and the gradient of a
clipped action keeps pushing it further out.

**What holds the arithmetic is not the pendulum.** The pendulum has no known
answer, so a run of it says nothing about whether the update is right.
`tests/test_gaussian.py` builds a one step problem whose answer is a number
written into the environment, and there the agent finds it to within 0.15 from
either side of the box, narrows its spread to a quarter of what it started at,
and its value network settles on minus the square of that spread, which is
exactly what a policy aiming at the answer with that spread is worth.

---

## When the answer is not a ranking

```console
$ python scripts/measure_aliased.py
$ rel train q-learning --env aliased --episodes 600
$ rel train reinforce --env aliased --episodes 600
```

Every other environment here hands the agent an observation that names the
state. `aliased` does not. It is three cells that all give the same number,
the middle one has its actions reversed, and every step pays -1. So a table
has one row for the whole corridor and a policy network has one set of weights
that cannot depend on where it is standing.

Both fixed choices never finish. Always right bounces between the first two
cells for ever, always left walks into the start's wall. **An agent that ranks
its two actions and takes the better one is choosing between two policies that
never reach the goal.** Nothing about the ranking is wrong. The answer is not a
ranking.

### The arithmetic

Write `p` for the probability of going right. The expected steps from the start
are `2 (2 - p) / (p (1 - p))`, which `rel/envs/aliased.py` derives in five
lines from the three cells.

| share of right | steps to the goal |
| ---: | ---: |
| 0.05 | 82.11 |
| 0.2 | 22.50 |
| 0.4 | 13.33 |
| 0.5858 | **11.66** |
| 0.75 | 13.33 |
| 0.95 | 44.21 |

*It runs off to infinity at both ends, which is the two fixed choices never
finishing.*

The smallest value is at `2 - sqrt 2`, which is 0.5858, and it is `6 + 4 sqrt
2` steps, which is 11.66.

**An epsilon-greedy agent is stuck at one end of that column.** It takes its
favourite action `1 - epsilon / 2` of the time, so at an epsilon of 0.1 its
share is 0.95 or 0.05 and the best of the two is 44.21 steps. That gap is not a
badly tuned agent. It is the whole family of methods that produce a ranking,
measured against the best a policy of this shape can do.

### What each agent reached

| agent | last 50 while learning | its own policy, steps | frozen greedy, steps | share of right |
| :--- | ---: | ---: | ---: | ---: |
| q-learning | -13.6 | 43.9 | 1000 | 1.000 |
| sarsa | -24.5 | 42.7 | 1000 | 1.000 |
| expected-sarsa | -14.0 | 42.2 | 1000 | 1.000 |
| reinforce | **-12.2** | **12.2** | 1000 | **0.569** |
| actor-critic | -77.0 | 68.9 | 1000 | 0.963 |

*Ten seeds, 600 episodes, epsilon 0.1. The last three columns have the learning
switched off, and the step limit is 1000.*

**The arithmetic predicts the measurement.** The three ranking agents' own
policies take 43.9, 42.7 and 42.2 steps against the 44.21 the closed form says
an epsilon-greedy policy is stuck at. `reinforce` takes 12.2 against the 11.66
the best policy of this shape reaches, and puts 0.569 of its policy on going
right against the 0.5858 that is best.

### The first column is a mirage

**q-learning scores -13.6 while learning and the policy it learned is worth
-43.9.** Read only the first column and the three ranking agents look within
two return of `reinforce`, which finds the answer. They are three and a half
times worse.

The reason is that their action values never settle. The corridor is one
observation, so both values are updated by every step of every episode, and
they cross each other repeatedly. The agent is therefore acting as a mixture,
and the mixture happens to be near the middle of the range where the corridor
is easy. Nobody chose that mixture, it is not what the agent has learned, and
stopping the learning removes it.

**This is what the last two columns are for.** An agent is worth what its
policy is worth, and the number a training run prints is not that unless the
policy has stopped moving.

### No agent here has a deterministic policy that works

The `frozen greedy` column is 1000 for every row, which is the step limit.
That includes `reinforce`: the most likely action of a softmax is still one
action, and one action never finishes here.

So `rel train`'s headline number, the return of the greedy policy, is -1000 for
every agent on this environment and says nothing about any of them. The
environment is the one place in this project where that number is the wrong one
to read.

### The actor critic does not find it

`actor-critic` can hold the answer. Its policy is a probability like
`reinforce`'s and 0.5858 is inside the range. It ends at 0.963, which is
nearly a fixed choice, and its policy is worth -68.9, which is worse than the
agents that cannot represent the answer at all.

That is consistent with what
[the actor critic section](#the-two-agents-that-learn-a-policy-directly) below
already reports about it and is not explained here.

### Where it is weak

- **One epsilon.** 0.1, for the ranking agents. A larger epsilon moves them
  towards the middle of the column and would help them here, which is the
  opposite of what it does everywhere else in this project.
- **One budget.** 600 episodes. Whether the ranking agents' values would
  eventually settle, and their learning column fall to meet their policy
  column, is not measured.
- **The share is read at one observation.** There is only one, so that is not
  a limitation here, and it would be on anything larger.
- **No seeds column.** The table is means over ten seeds and the spread is not
  printed. The finding is a factor of three and a half, which is far outside
  anything ten seeds hide, and a smaller finding here would need more.

---

## The two agents that learn a policy directly

Neither is as steady as anything tabular in this project, and both are far
slower. That is reported here rather than tuned away.

### On the cliff walk

```console
$ python scripts/measure_agents.py --env cliff --agents reinforce \
    --runs 12 --episodes 400 --each-seed
```

Twelve seeds, four hundred episodes, exact value of the greedy policy
afterwards:

```
REINFORCE:  -13 -13 -13 -13 -13 -13 -15 -13 -15 never -15 -15
```

Seven seeds of twelve find the optimal policy of -13. Four reach -15. One does
not reach the goal inside four hundred episodes, and that is a statement about
four hundred rather than about the seed: it reaches the goal at episode 784.
Over the eleven that finish the mean is **-13.73**, against a best possible of
-13.

Twelve rather than six, and the reason is in
[the entropy section](#the-entropy-bonus-on-a-policy-gradient) below. Six
seeds said this setting was perfect. It is not.

#### The step limit is not an ending

That last count used to be three, and this page used to say the cause was not
known. It is known now, and the guess written here at the time was wrong in a
way worth keeping.

`Reinforce._returns` read an episode that the step limit cut off as an episode
that ended. A cliff walk episode that never reaches the goal is five hundred
steps of -1. With the tail read as zero, the return of the last step is -1 and
the return of the first is -99.34. Standardising those returns gives:

<!-- not checked: the standardised weights of one episode, worked out from
the returns described above rather than printed by anything -->
| step of the episode | weight |
| --- | ---: |
| the first three | -0.78, -0.78, -0.78 |
| the last three | +3.16, +3.20, +3.24 |

So the agent was told to repeat whatever the step limit stopped it doing, and
to stop doing whatever it began the episode with. It circles, the circling is
rewarded, and it circles harder.

The guess on this page was that standardising **removes** the signal when every
step of an episode is equally bad. The measurement says the opposite. The
signal was large, its spread across the episode was 4.02, and it pointed
backwards.

The cart pole shows the same fault from the other side, because there reaching
the step limit is the goal. Five hundred steps of +1 with a zero tail gives the
first steps a weight of +0.78 and the last steps -3.24. An agent that held the
pole up for the whole episode was told to stop doing whatever held it up at the
end.

The tail is estimated now. The value network answers if there is one, and the
actor critic in the same file always did this, which is what made the fault
visible. Six seeds of the cliff walk, before and after:

<!-- not checked: the before column is a version of the code that no
longer exists, so nothing can print it again -->
| seed | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before | -17 | -13 | never | never | -13 | never |
| after | -15 | -13 | never | -13 | -13 | -13 |

The one that is left is a different fault. On seed 3 the agent never reaches
the goal once in four hundred episodes, before the fix or after it. So no rule
for sharing out a return can help: there is no return that reached the goal to
share out. The entropy bonus is what keeps the policy wide enough to find the
goal in the first place, and 0.01, which is the default, is not enough on this
seed. That one is in [milestones.md](milestones.md) as an open question.

### On the cart pole

```console
$ python scripts/measure_control.py --env cartpole --episodes 600 --agents reinforce
```

REINFORCE, five seeds, six hundred episodes: **500 on every one of them**.
That is the full score. The same command gave 198, 181, 77, 500, 148 before
the step limit fault above was fixed, and 500, 143, 140, 500, 500 after the
fix and before the entropy default was measured.

The actor critic does not solve the cart pole at any setting that was tried.
What was tried:

<!-- not checked: a record of the settings that were tried, with what each
one did written out rather than tabulated from a run -->
| setting | result |
| --- | --- |
| step size 0.02, 400 episodes | 500 on one seed, and it collapses by 1000 |
| step size 0.05 | 8 on every seed, immediately |
| entropy 0.03, three seeds | 66, 121, 8 |
| a hidden layer of 32 | 78 |

The entropy bonus is the one setting that moves it a long way, so it was swept
on five seeds and six hundred episodes:

```console
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents actor-critic --set entropy=0.01
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents actor-critic --set entropy=0.05
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents actor-critic --set entropy=0.1
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents actor-critic --set entropy=0.2
```

One command a row. Each run is built with one entropy and prints one row, so
naming only the 0.05 command left the other three rows with nothing behind
them.

<!-- not checked, column entropy: the cells hold the setting each row was run
at rather than anything a run produced -->
| entropy | mean | each seed |
| --- | ---: | --- |
| 0.01, the default | 8.4 | 8 8 8 8 9 |
| 0.05 | 146.8 | **500** 8 8 80 138 |
| 0.10 | 33.3 | 113 22 8 16 8 |
| 0.20 | 207.2 | 212 177 **500** 126 22 |

The shape of that table is the answer. The mean does not rise with the
setting: 0.10 is worse than 0.05 and worse than 0.20. Two settings reach the
full five hundred on one seed each, and no setting reaches it on two. A knob
that a run is this sensitive to is not a knob that has a right value.

It does find the optimal policy on the cliff walk, which says the pieces are
put together correctly rather than that the algorithm is right. The gradient
engine underneath is checked against the definition of a derivative for every
operation, so the fault is not there either. REINFORCE, which shares this
file, this encoder and this optimiser, reaches five hundred on all five seeds
once its entropy is raised. So the difference is the one thing the two do not
share: bootstrapping a target from a value network that is still wrong.

The honest reading is that this is a working implementation of a method that is
sensitive, and that making it work would need something this project does not
have: a target network, or n-step returns to lengthen the part of the target
that is measured rather than guessed. That is in
[milestones.md](milestones.md) as well.

### The entropy bonus on a policy gradient

Williams' method has no entropy bonus at all, and this project's classes
default to little or none. The registry does not, and the reason is the same
one as for the tile coder above: a default is worth choosing by measurement.

```console
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents reinforce --set entropy=0.05
$ python scripts/measure_control.py --env cartpole --episodes 600 \
    --agents reinforce --set entropy=0.01
$ python scripts/measure_agents.py --env cliff --agents reinforce \
    --runs 12 --episodes 400 --each-seed --set entropy=0.05
$ python scripts/measure_agents.py --env cliff --agents reinforce \
    --runs 12 --episodes 400 --each-seed --set entropy=0.01
```

One command a row again, and the default of this agent is 0.05, so the 0.01
rows need to ask for it.

**Cart pole**, five seeds, six hundred episodes:

<!-- not checked, column entropy: the cells hold the setting each row was run
at rather than anything a run produced -->
| entropy | mean | each seed |
| --- | ---: | --- |
| 0.01 | 356.5 | 500 143 140 500 500 |
| **0.05** | **500.0** | 500 500 500 500 500 |

**Cliff walk**, twelve seeds, four hundred episodes, exact value of the greedy
policy:

<!-- not checked, column entropy: the cells hold the setting each row was run
at rather than anything a run produced -->
| entropy | mean of those that finish | policy never finishes | each seed |
| --- | ---: | ---: | --- |
| 0.01 | **-13.22** | 3 | -15 -13 never -13 -13 -13 -13 never -13 never -13 -13 |
| 0.05 | -13.73 | **1** | -13 -13 -13 -13 -13 -13 -15 -13 -15 never -15 -15 |

So 0.05 is the default. On the cart pole it is the whole score against two
thirds of it. On the cliff walk it is a trade rather than a win: it loses one
seed instead of three, and the policies it does find are 0.51 blunter. An agent
that fails one run in four is worse than one that is a little short of optimal,
so the trade is taken, and it is a trade rather than a free lunch.

#### Six seeds said this was a clean win, and six seeds were wrong

The first sweep of this was six seeds. At 0.05 it read `-13 -13 -13 -13 -13
-13`: the optimal policy on every seed, no failures, nothing to trade. Twelve
seeds says otherwise, because seeds 7 to 12 are `-15 -13 -15 never -15 -15`.

Nothing went wrong in the first measurement. It is what six samples of a noisy
thing look like, and the mistake would have been to publish it. The number of
seeds behind a claim on this page is now part of the claim.

#### The seed that never reached the goal reaches it at episode 784

```console
$ python scripts/measure_lost_seed.py --entropies 0.05 0.1 0.2 --ladder-all
```

The table above says one seed of twelve does not reach the goal. It says that
about four hundred episodes.

| episodes | not there yet | which seeds |
| ---: | ---: | --- |
| 500 | 1 | 10 |
| 1000 | 0 | |
| 2000 | 0 | |

*Twelve seeds at the default entropy. Each is run for 2000 episodes once and
read at every budget.*

Seed 10 reaches the goal at episode 784 and ends at -20.2 by episode 2000. It
is not a seed that cannot learn. It is a seed the budget was too short for.

The milestones left three possibilities open: more entropy, more episodes, or
something that is not a knob at all. On seed 10 the entropy bonus moves it a
long way, because that is the dial that holds the policy wide enough to wander
as far as the goal. First reached at episode 784 at the default of 0.05, 172 at
0.1, 92 at 0.2 and 25 at 0.4. Past 0.1 the policy it ends with gets worse
instead: -17.8 at 0.1, -26.0 at 0.2 and -173.9 at 0.4.

So the interesting question is not what rescues seed 10 but what the default
should be, and that is a question about all twelve:

| entropy | first goal, median | last 100 | exact value | stuck |
| ---: | ---: | ---: | ---: | ---: |
| **0.05** | 48 | **-20.0** | **-13.18** | 1 |
| 0.1 | **35** | -27.9 | -13.67 | **0** |
| 0.2 | 37 | -46.6 | -13.55 | 1 |

*Twelve seeds, a thousand episodes. Every seed reaches the goal at every
setting, so the first column is a median over all twelve. `stuck` counts the
seeds whose greedy policy never reaches an ending.*

**The default stays at 0.05.** Raising it to 0.1 removes the one stuck seed and
0.2 brings it back, so the stuck column is not something the dial controls, and
both settings cost 8 and 27 return while learning and leave the finished policy
no better. This is the same trade as the 0.01 to 0.05 move above, one notch
further along, and past 0.05 the trade stops paying.

The answer to the open question is the dull one: **that seed wants more
episodes.** The measurement that called it lost runs for four hundred.

### They are hundreds of times slower

Two hundred episodes of the cliff walk, seed 1:

```console
$ rel train q-learning --env cliff --episodes 200 --seed 1
$ rel train reinforce --env cliff --episodes 200 --seed 1
```

<!-- not checked, column time: seconds belong to the machine, and the
steps beside them are what the section compares -->
| agent | time | steps |
| --- | ---: | ---: |
| q-learning | 0.05s | 5,159 |
| reinforce | 37.41s | 74,726 |

Fourteen times the steps and seven hundred times the time. The steps are the
agent's own fault: it does not find the goal until episode 139 of the 200, so
most of the run is episodes that reach the five hundred step limit. The rest is
a network in pure Python.

---

## Sweeping without writing a script

Every table on this page began as a script. `rel sweep` varies one or two
settings and prints the table directly.

```console
$ rel sweep n-step-sarsa --env cliff --over n=1,2,4 --over step_size=0.1,0.5 \
      --episodes 500 --runs 10
```

```
n  step_size  last 100  error     exact value  stuck
-  ---------  --------  --------  -----------  -----
1        0.1    -21.84  +/- 0.74       -15.20
1        0.5    -26.89  +/- 1.77       -17.00      2
2        0.1    -22.13  +/- 0.68       -16.00
2        0.5    -30.10  +/- 2.74       -17.00      5
4        0.1    -22.53  +/- 1.14       -17.00      1
4        0.5    -49.79  +/- 8.73       -18.67      4
```

Two settings give every pair, and that is the point of sweeping two rather than
one at a time. The trade in
[How many steps, and how big a step](#how-many-steps-and-how-big-a-step)
is only visible in the grid: at a step size of 0.1 the three values of `n` are
within noise of each other, and at 0.5 reaching further back costs more with
every step it reaches.

`--each-seed` prints the number from every seed behind each row. A mean over
runs that vary a great deal has been the wrong answer twice on this page.

## The band on the compare chart

```console
$ rel compare sarsa q-learning --env cliff --runs 10
```

The chart draws the mean over the seeds, and behind it the best and the worst
seed at each point. On the cliff walk that line is smooth while the runs under
it swing by a hundred and fifty, and the band is the only thing on the picture
that says so.

The band is its two edges rather than the filled area between them. Filled
reads better in colour and it is the same braille dot as a curve, so with
colour switched off it swallows the line it is the spread of. `--no-band` turns
it off.

---

## Whether the difference is real, and how large

```console
$ rel compare sarsa q-learning --env cliff --runs 10
$ python scripts/measure_noise.py --trials 200
$ python scripts/measure_noise.py --trials 200 --runs 10
```

Two means and two standard errors describe two sets of numbers. A reader takes
a comparison from them anyway, so `rel compare` makes the comparison instead of
leaving it to be made by eye:

```
sarsa against q-learning, paired by seed
  difference    +16.53
  95% interval  +10.50 to +21.83
  p value       0.0078
```

**The comparison is paired.** Both agents meet the same seeds, so a difference
on seed 4 is a difference between the agents rather than between the problems
they were given. That matters here more than usual: one agent's ten cliff walk
seeds run from -20 to -545, and a comparison that threw the pairing away would
be looking for a difference of sixteen inside a spread of five hundred.

The p value comes from flipping the sign of each paired difference every
possible way and counting how often the mean lands at least as far from zero.
Under the claim that the two agents are the same, the label on each pair is
arbitrary, so every sign pattern was equally likely.

### Five seeds cannot reach five percent

A paired test over `n` seeds has `2 ** n` sign patterns, and the two most
extreme of them are always at least as far from zero as whatever was seen. So
the smallest p it can ever report is `2 / 2**n`:

<!-- not checked: this is two over two to the n, worked out rather than
run. Nothing prints it and nothing can move it. -->
| seeds | smallest p possible |
| ---: | ---: |
| 3 | 0.2500 |
| 4 | 0.1250 |
| 5 | **0.0625** |
| 6 | 0.0313 |
| 10 | 0.0020 |

**Whatever the difference is.** A million to one gap on five seeds still reads
0.0625, because there are only thirty two ways to arrange five signs.

Several measurements on this page ran five seeds, so none of them could have
reached 0.05 whatever they found. `rel compare` says so when it applies, since
otherwise a p of 0.06 reads as a result that nearly happened rather than as the
best the arithmetic allows.

### How large a difference turns up by chance

`scripts/measure_noise.py` runs one agent against a copy of itself: same code,
same settings, same environment seeds, and a different draw for the agent on
each run. Everything it reports is noise by construction.

| over 200 comparisons | 5 seeds | 10 seeds |
| --- | ---: | ---: |
| difference, median | 3.14 | 2.32 |
| difference, nine in ten under | 7.71 | 4.76 |
| difference, largest seen | 16.63 | 8.99 |
| interval excluded zero | **39 of 200** | 12 of 200 |
| p below 0.05 | 0 of 200 | 6 of 200 |

*`q-learning` on the cliff walk, 100 episodes each.*

Three things to read off it.

**A 95% interval on five seeds is wrong about one time in five.** It should
have excluded zero on about ten of those two hundred and it did on thirty nine.
The bootstrap has five numbers to resample and it is optimistic about what it
can tell from them. At ten seeds it is twelve of two hundred, which is what a
95% interval is supposed to look like.

**The permutation test is never wrong at five seeds and correctly cautious at
ten.** Zero of two hundred at five, because it cannot fire at all. Six of two
hundred at ten, which is 3% against a nominal 5%: an exact test is conservative
rather than calibrated, and that is the right direction to be wrong in.

**The noise floor halves between five seeds and ten.** The largest difference
two identical agents showed is 16.63 at five seeds and 8.99 at ten. That is the
number to hold a claimed difference against, and it is why the sarsa result
above is worth believing: +16.53 is roughly twice anything noise produced over
two hundred tries at ten seeds.

### What this project should do, and mostly did not

**Ten seeds, not five.** It costs twice the time and it moves the interval from
wrong one time in five to wrong one time in seventeen, and it moves the
permutation test from unable to fire to correctly calibrated. Several tables on
this page run five, and they are marked with the seed count for exactly this
reason.

### Where it is weak

**Pairing is an assumption about how the runs were made.** These functions are
the wrong ones for two agents run on different seeds, and nothing in the
numbers could tell.

**The environment seed does nothing on a deterministic grid.** The cliff walk
has no chance in it, so every seed builds the identical grid and all the
variation in a run of it is the agent's own. Its seeds are agent seeds, and the
same is true of every environment here except the bandits and the frozen lake.

**Two agents, not three.** With three the reader wants three pairs, or one
against each of the others, and which of those was meant is a question rather
than a default.

---

## Making the engine faster without changing what it computes

```console
$ python scripts/measure_engine.py
$ python scripts/measure_engine.py --agent deep-q --env cliff --episodes 3
```

The gradient engine is pure Python, so it is slow, and the agents built on it
are the slowest things here. Speeding it up is easy. Speeding it up **without
changing a single number** is the part worth doing carefully, because an
optimisation that reassociates a sum is faster and gives different answers, and
on a learning agent different answers look exactly like the same agent on
another seed.

So every change below was checked against the two digests, and every one of
them left both untouched.

<!-- not checked: every cell is seconds on one machine, and the before
column is a version of the code that no longer exists -->
| 20,000 passes each | before | after |
| --- | ---: | ---: |
| a whole `reinforce` run | 0.142s | **0.131s** |
| 4 to 16 forward | 0.262s | 0.246s |
| 4 to 16 backward | 0.221s | 0.208s |
| 48 to 16 forward | 1.155s | 1.137s |
| 48 to 16 backward | 2.062s | **1.868s** |
| Adam over 852 numbers | 5.359s | **4.289s** |

*Digests before and after: `26caed70127e02dc` and `65988fc88c846945`, both.*

Three changes, and none of them is clever. Every list an inner loop reads is
pulled into a local first, in `linear` and again in `Adam`, because
`weight.data` inside a loop that runs once per weight is an attribute lookup
per weight. An internal constructor skips the copy and the shape check that the
public one owes a caller, because every operation in the engine builds a fresh
list of floats and knows the shape it made it to.

**The same numbers are added in the same order.** That is the whole discipline,
and the digest is what turns it from an intention into a check.

### Three things the measuring got wrong first

**A single timing is worth about as much as a single seed.** The first version
of the benchmark timed each shape once. Unchanged code answered anywhere from
1.06 to 1.22 seconds for the same twenty thousand passes, depending on what had
run in the process before it, and the first timing was the slow one every time.
The first optimisation was measured against that and its commit message claims
a gain twice the size of the real one. Best of three now.

**A profile is not a measurement of the run.** `cProfile` put the optimiser at
15% of a `deep-q` run, which is roughly right, and made it look like the place
to work. Adam did turn out to be 18% faster after the change. The run moved by
0.3%, because the whole-run timing cannot resolve the 2% that an 18% saving on
a tenth of the work buys. A component a whole run cannot see needs its own line
in the table, which is why there is now one for the optimiser.

**A faster inner loop can be slower.** The first version of the `linear` change
zipped over a slice of the weight row instead of indexing into it. That is
faster on a wide layer and slower on a narrow one: 48 to 16 forward went from
1.22 to 1.10, and 4 to 16 forward went from 0.29 to 0.38, because slicing four
elements sixteen times a pass costs more than the indexing it removes. This
project runs both shapes, so the slice is out.

### Where it is weak

**Nothing here changes the shape of the work.** The engine still allocates a
tensor for every intermediate value and walks the graph on every backward pass,
and those are what a real engine avoids. What is left after these changes is
what pure Python costs to do the arithmetic at all.

**The backward pass still computes a gradient nobody reads.** `linear` fills in
the gradient with respect to its input, and for the first layer of a network
that input is the observation, which nothing differentiates with respect to. On
a 48 to 16 layer that is half the inner loop of the most expensive backward
here. Skipping it needs the engine to be told which tensors want gradients, and
that is a change to what `backward` means rather than to how fast it is, so it
is not here.

---

## Checking that this page still says what the code does

Every table here names a command. Nineteen tracks have gone past them and
several changed a default, and a default that changes makes every table
produced with the old one wrong, silently. Nothing about reading the page says
which.

```console
$ python scripts/check_numbers.py --list
$ python scripts/check_numbers.py --only tiling
$ python scripts/check_numbers.py --all --jobs 4 --cache outputs.json
```

It runs every command on this page once, then puts each table against every
output and attributes it to the command that accounts for most of its numbers.
`--list` runs nothing and says what would be checked. `--cache` keeps what each
command printed, because the answer stops being true as soon as the page is
edited.

### What a whole run costs

<!-- not checked: seconds on one machine, and the two ordering rows are the
same measured times arranged two ways rather than two runs -->

| | seconds | as time |
| :--- | ---: | ---: |
| every command, added up | 14,921 | 4 hours 9 minutes |
| the slowest single command | 1,800 | 30 minutes |
| four at a time, in page order | 4,531 | 75 minutes |
| four at a time, longest first | 3,731 | 62 minutes |
| the total divided by four | 3,730 | 62 minutes |

*105 commands on four processors. The two ordering rows are the measured times
of one run arranged two ways: page order is what that run did, and longest
first is what the same times give when the long ones start first.*

**Sixty two minutes is the floor and the ordering reaches it.** Four hours of
work on four processors cannot take less than an hour and two minutes, and
starting the long commands first is enough to get there. Page order costs
twelve minutes more, because the twenty five minute command can start last.

`--jobs` is what makes any of that possible. Every command on this page is
seeded and prints the same numbers whatever else is running, so it moves the
wall clock and nothing else: the seven fast ones the continuous integration job
runs take 16.9 seconds one at a time and 8.4 at four, and account for the same
three tables.

**The first run cannot be ordered.** Nothing is known about how long anything
takes until something has run, so a first run is in page order and takes the
seventy five. The cache carries the times into the next one, and it carries
them across a code change, which is the case that matters: what a command
prints is worthless once the code moves and how long it takes is not.

### Matching, because position cannot work

The first version took the tables that followed a console block and asked
whether that block's commands printed them. Two thirds of what it reported was
a table sitting under a command it did not come from, and the section called
`The same table, four more environments` is why. It wrote one command and then
four tables, and the three other commands were nowhere on this page, so no rule
about position could have been right.

### What it finds

**A command the page never named.** Four tables under one of their four
commands, in `The same table, four more environments`. Three grids of
exploration rules with no console block above them at all. Five rows of a
crossover table, one for each length of the long loop. Every one is written
down now and every one reproduces.

**A command the page named wrongly.** The importance sampling block said
`python scripts/measure_importance.py` while the prose beside it said 1200
episodes. The default is 1500 and 1500 prints a different table. Nothing had
drifted: the current code at `--episodes 1200` reproduces every number in that
table exactly, on both grids. The exploration sweep on a tile coder was the
same defect with a different setting: it named a command at the default of 300
episodes where every cart pole row under it is 600.

**A table no command could print.** One was a row of updates per seed behind a
median, which no option of any script produced, so `measure_sweeping.py` gained
`--each-seed` and prints it. The rest genuinely cannot be printed: differences
between two runs, arithmetic, seconds on a machine. Those carry a comment
saying so and why, which the report prints instead of listing their numbers as
missing. The reason is required, because a table that exempts itself silently
is how a number that moved would hide.

**Nothing had drifted.** That is the result, and it is the whole run rather
than a sample: **34 of the 34 tables this page states results in are wholly
accounted for by a command it names, and none by no command at all.** Every
number that was ever in doubt turned out to be right. What the page had wrong
was where its numbers came from, and it had that wrong in a dozen places.

### What it cannot see

- **Which cell a number is in.** It asks whether a number appears anywhere in
  an output. A number that moved disappears from every output and is caught. A
  number that swapped places with another in the same table is not.
- **A table that rounds.** The digits are compared as text, so a page writing
  1,970,224,597,202 where the command prints 1970224597202.702 states a number
  the output does not hold. A rule loose enough to match a truncation is loose
  enough to confirm a number that really moved, so this page prints what the
  command prints instead.
- **Half a table.** A command has to account for half a table before it is
  called its source. Below that it is a coincidence: a small integer that every
  output happens to print is enough to win when nothing else matches anything.
- **Any number written in a sentence.** It reads tables, and **686 of this
  page's numbers are in prose rather than in a cell**, against 1792 in cells.
  A table cell is a result by construction and a sentence is not: most of
  those 686 are settings, seed counts and episode caps rather than anything a
  run produced, so matching them against the outputs would bury the report in
  noise. Two of the wrong numbers found while building this were in prose, and
  both were found by reading rather than by the tool. A test holds this count,
  because it is itself a number in a sentence.

### It reads any page

`--doc` takes a path, so the same tool answers for the other documents in this
repository. Four of them state numbers and all four come back clean:

<!-- not checked: every row is a run of this script, and it does not run
itself, so no command on this page prints any of these -->
| page | commands | tables | numbers | it took |
| --- | ---: | ---: | ---: | ---: |
| `README.md` | 13 | 4 | 59 | 112s |
| `docs/specification-gaming.md` | 8 | 6 | 72 | 59s |
| `docs/milestones.md` | 6 | 1 | 6 | 44s |
| `docs/grids.md` | 5 | 0 | 0 | 4s |

The grid page's one table is the settings a grid file takes and what each one
defaults to. Those are the code's own defaults rather than anything a run
prints, so the column is marked and `tests/test_gridfile.py` holds every one of
them against the signature that carries it.

Pointing it at those four found six things, and the worst of them was in the
tool. **It ran `git clone` and `pip install .`**, because the readme's install
block is a console block and it ran every command in every console block. It
cloned this repository into itself and installed the package, in order to check
a table of line counts. It now runs a script under `scripts/` or the package's
own command line and nothing else, and names in the report what it left alone.

The other five were in the pages. Five commands of the gaming page carry a
trailing comment, which a shell drops and this handed to the program as
arguments, so all five were called commands that would not run at all. The
readme wrote `-4` where `rel gaming` prints `-4.0`, its table of results named
no command at all, and its table of sizes had drifted by the docstrings this
sitting rewrote. The grid page shows what a bad setting looks like, which is a
transcript of a failure rather than a command to run, so it is no longer
written as a console block.

### Where it is weak

- **Two and a half hours.** Three commands take more than half an hour each,
  and one of them has never finished: `rel train mcts --env maze --set
  reuse=off` was still going at fifty minutes and gave up there. The cache is
  what makes a second opinion cheap, and it writes down which commands ran out
  of time so a resumed run does not spend the budget on them again. With every
  command cached the whole page answers in a second.
- **A command can outrun the default budget.** `python
  scripts/measure_levels.py` takes about fifteen minutes on its own and longer
  beside anything else, so the first check of its two tables reported them as
  accounted for by nothing at all. That reads exactly like a table whose
  numbers have moved, and the difference is one line further down the report,
  which names the command and the budget it wanted. `--timeout` is what to
  reach for before believing the first reading.
- **1463 of 1792 numbers are checked.** The rest are in a table or a column that
  says why it cannot be, and `--list` prints the split. A test holds this
  sentence against what `--list` says, because a count written in prose is
  exactly the kind of number this whole exercise is about.
- **Line numbers go stale.** The report names the line a table starts on, and
  fixing anything above it moves every one after.

---

## Where the numbers on this page can be checked

Every command here is written the way a console block on this page writes it,
because a row that drops a setting sends a reader to a different table. Ten of
these rows named a command this page does not run, and seven of the ten would
have produced a different table: the importance sampling row ran the default of
1500 episodes where the block runs 1200, and the value network row ran 200
where the block runs 400. The other three named the same run in other words,
which is harmless and still worth spelling one way.

| Table | Command |
| --- | --- |
| Every agent on a grid | `python scripts/measure_agents.py --runs 10` |
| The agents that approximate | `python scripts/measure_control.py --env cartpole --episodes 600` |
| One setting swept | `python scripts/measure_control.py --env cartpole --episodes 600 --agents actor-critic --set entropy=0.05` |
| Every seed behind a mean | `python scripts/measure_agents.py --env cliff --agents reinforce --runs 12 --episodes 400 --each-seed` |
| The tile coder offsets | `python scripts/measure_tiling_offsets.py` |
| Tile coding against a radial basis | `python scripts/measure_approximation.py --runs 10` |
| Whether this page still says what the code does | `python scripts/check_numbers.py --all --jobs 4 --cache outputs.json` |
| What importance sampling costs | `python scripts/measure_importance.py --episodes 1200` |
| Ordered replay against uniform | `python scripts/measure_sweeping.py --episodes 400` |
| The seed that gets lost | `python scripts/measure_lost_seed.py --entropies 0.05 0.1 0.2 --ladder-all` |
| What an option costs | `python scripts/measure_options.py --runs 20` |
| What crediting the middle buys | `python scripts/measure_intra_option.py --episodes 800 --block 100` |
| Prediction against a known answer | `python scripts/measure_prediction.py` |
| Replay and a target network | `python scripts/measure_value_network.py --env cartpole --episodes 400 --runs 10 --set step_size=0.02` |
| Drawing from the buffer by priority | `python scripts/measure_prioritised.py` |
| The maximum overstates | `python scripts/measure_double.py` |
| When the answer is not a ranking | `python scripts/measure_aliased.py` |
| Four ways of exploring | `python scripts/measure_exploration.py` |
| Which loop a discount chooses | `python scripts/measure_average_reward.py` |
| How large a difference is noise | `python scripts/measure_noise.py --trials 200` |
| The engine, faster and unchanged | `python scripts/measure_engine.py` |
| Decision time against background planning | `python scripts/measure_search.py --episodes 40 --runs 3` |
| One rule at several dials | `python scripts/measure_exploration.py --rules count-bonus:0.1,count-bonus:2` |
| The three things that are safe in pairs | `python scripts/measure_triad.py` |
| A picture that is mostly the solver | `python scripts/measure_gambler.py` |
| What the van is worth | `python scripts/measure_rental.py` |
| An action that is a number | `python scripts/measure_levels.py` |
| The smallest approximation there is | `python scripts/measure_aggregation.py` |
| Waves over the whole box | `python scripts/measure_fourier.py --runs 10 --step-sizes 0.02 0.05 0.1 0.2 0.5 1.0 2.0` |
| Learning against an answer that is known | `python scripts/measure_blackjack.py` |
| Part sample and part expectation | `python scripts/measure_sigma.py` |
| Drawing in the log of the buffer | `python scripts/measure_tree.py` |

### The commands that are behind no table

These produce nothing on this page. They are how the tool is driven, and the
settings written in them are examples rather than the ones some table above
was made with. They sat in the table above until it became clear that a reader
cannot tell one kind of row from the other by looking.

| What it does | Command |
| --- | --- |
| Any one or two settings | `rel sweep <agent> --env <env> --over name=a,b,c` |
| Specification gaming | `rel gaming` |
| One run in detail | `rel train q-learning --env cliff --seed 7` |
| Two agents side by side | `rel compare sarsa q-learning --env cliff` |
| The best possible policy | `rel solve --env cliff` |
