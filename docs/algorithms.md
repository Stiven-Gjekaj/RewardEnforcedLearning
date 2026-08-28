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
```

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
$ python scripts/measure_importance.py
```

Ten seeds, 1200 episodes, a behaviour policy that explores a fifth of the time.
`spread` is how far apart the ten seeds' estimates of one cell are, averaged
over every cell all ten of them credited.

| grid | estimator | cells | spread | widest | policy |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen lake, best 0.824 | ordinary | 10 | 4.194 | 13.204 | 0.237 |
| frozen lake | **weighted** | 10 | **0.710** | **0.913** | **0.369** |
| maze, best 0.513 | ordinary | 2 | 1,970,224,597,202 | 3,928,201,830,698 | never finishes |
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
$ rel train dyna-q --env maze --set planning_steps=50
```

Seed 3, step size 0.1, epsilon 0.1, discount 0.95. Steps taken in each of the
first ten episodes:

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

The median is a factor of nearly ten, and the worst seed the sweeper solves takes
fewer updates than the median seed the uniform planner solves. Both see the
same steps and hold the same model. The only difference is which remembered
step is replayed next.

Per seed, in updates:

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

## An action that lasts several steps

```console
$ rel train options-q --env rooms
$ rel train options-q --env rooms --set hallways=off
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

*Four rooms, ten seeds, 500 episodes. The best possible return is -20.*

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

*Twenty seeds. `long` is the share of choices that were an option lasting more
than a step, and `length` is the mean number of steps an option ran for.*

Epsilon changes by a factor of ten and the fourth column sits between 22 and 31
with no trend in it. The cost is the exploring.

That is not an argument against options. It is a measurement of what this
particular pairing costs on this particular grid, where the reward is dense and
one step of feedback arrives after every step. An option is a commitment, and a
commitment is worth what it saves minus what it costs to be wrong about. Here
there is nothing to save: the value of a cell is already carried back one cell
per step by every reward, and the doorways are not where the agent wants to be.

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
$ python scripts/measure_control.py --env cartpole --agents tile-sarsa
```

Five seeds each, greedy return afterwards:

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

Seven seeds of twelve find the optimal policy of -13. Four reach -15. One
never reaches the goal at all. Over the eleven that finish the mean is
**-13.73**, against a best possible of -13.

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
    --agents actor-critic --set entropy=0.05
```

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
$ python scripts/measure_agents.py --env cliff --agents reinforce \
    --runs 12 --episodes 400 --each-seed --set entropy=0.05
```

**Cart pole**, five seeds, six hundred episodes:

| entropy | mean | each seed |
| --- | ---: | --- |
| 0.01 | 356.5 | 500 143 140 500 500 |
| **0.05** | **500.0** | 500 500 500 500 500 |

**Cliff walk**, twelve seeds, four hundred episodes, exact value of the greedy
policy:

| entropy | mean of those that finish | never reaches the goal | each seed |
| --- | ---: | ---: | --- |
| 0.01 | **-13.22** | 3 of 12 | -15 -13 never -13 -13 -13 -13 never -13 never -13 -13 |
| 0.05 | -13.73 | **1 of 12** | -13 -13 -13 -13 -13 -13 -15 -13 -15 never -15 -15 |

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

### They are hundreds of times slower

Two hundred episodes of the cliff walk, seed 1:

| agent | time | steps |
| --- | ---: | ---: |
| q-learning | 0.05s | 5,159 |
| reinforce | 37.41s | 74,726 |

Fourteen times the steps and seven hundred times the time. The steps are the
agent's own fault: it does not find the goal until episode 139 of the 200, so
most of the run is episodes that reach the five hundred step limit. The rest is
a network in pure Python.

---

## Where the numbers on this page can be checked

| Table | Command |
| --- | --- |
| Every agent on a grid | `python scripts/measure_agents.py --env cliff --runs 10` |
| The agents that approximate | `python scripts/measure_control.py --env cartpole` |
| One setting swept | `python scripts/measure_control.py --env cartpole --set entropy=0.05` |
| Every seed behind a mean | `python scripts/measure_agents.py --env cliff --each-seed` |
| The tile coder offsets | `python scripts/measure_tiling_offsets.py` |
| What importance sampling costs | `python scripts/measure_importance.py` |
| Ordered replay against uniform | `python scripts/measure_sweeping.py --episodes 400` |
| What an option costs | `python scripts/measure_options.py --runs 20` |
| Specification gaming | `rel gaming` |
| One run in detail | `rel train q-learning --env cliff --seed 7` |
| Two agents side by side | `rel compare sarsa q-learning --env cliff` |
| The best possible policy | `rel solve --env cliff` |
