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
| n-step-sarsa | -49.79 +/- 8.73 | -18.67 | 4 |
| dyna-q | -52.80 +/- 1.96 | **-13.00** | |
| dyna-q-plus | -47.56 +/- 1.91 | **-13.00** | |

*500 episodes, seeds 1 to 10, step size 0.5, epsilon 0.1.*

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

Two of ten SARSA seeds and four of ten n-step seeds produce a greedy policy
that never finishes.

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
| n-step-sarsa | -27.10 +/- 1.05 | -17.50 | 6 |
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
| n-step-sarsa | -27.83 +/- 0.55 | -20.67 | 1 |
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
| n-step-sarsa | 0.82 +/- 0.10 | 0.45 | 7 |
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
| n-step-sarsa | 0.08 +/- 0.01 | 0.14 | |
| dyna-q | 0.14 +/- 0.01 | 0.36 | |
| dyna-q-plus | 0.12 +/- 0.01 | 0.29 | |

Nothing here gets near 0.82 in six hundred episodes, and the ones that do worst
are the ones whose assumptions the lake breaks. Dyna keeps one result for each
state and action, and on a slippery lake it remembers the last slip and replays
it as if it were certain. n-step SARSA carries several steps of a trajectory
that the slipping made unrepresentative.

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
dimensional box, the number of cells each grid could reach ran:

```
4096, 6374, 5151, 3983, 3023, 2149, 1506, 943      out of 6561
```

The last grid had lost six sevenths of itself. Nothing about a run said so. The
agents still learned, and they learned with a coder that had quietly thrown
away most of its resolution.

After taking the shift modulo one cell:

```
4096, 6374, 6466, 6369, 6554, 6382, 6463, 6384     out of 6561
```

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

## Where the numbers on this page can be checked

| Table | Command |
| --- | --- |
| Every agent on a grid | `python scripts/measure_agents.py --env cliff --runs 10` |
| The agents that approximate | `python scripts/measure_control.py --env cartpole` |
| The tile coder offsets | `python scripts/measure_tiling_offsets.py` |
| Specification gaming | `rel gaming` |
| One run in detail | `rel train q-learning --env cliff --seed 7` |
| Two agents side by side | `rel compare sarsa q-learning --env cliff` |
| The best possible policy | `rel solve --env cliff` |
