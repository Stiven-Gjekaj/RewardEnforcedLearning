<div align="center">
  <a href="../README.md"><b>Reward Enforced Learning</b></a>
</div>

# Milestones

What is not built, what was looked at and deliberately left, and the open
questions. Everything here that has a number behind it says what was measured.

---

## Not built yet

| | What it would add | Why it is not here |
| --- | --- | --- |
| Continuous actions | Half of control, and everything about robotics | The whole action interface here is `Discrete`. This is a large change and it should be a large change |

---

## Built since that table was written

**Exploring by something other than chance.** `explore` is a setting on every
tabular agent, and three rules answer it: epsilon-greedy, softmax, and a
count-based bonus. Optimistic initialisation is the fourth way and it is a
starting value rather than a rule.

The finding is one sentence. **The dial of a rule cannot help find a first
reward, and only what the rule ranks by can.** Before anything pays, every
value in the table is the starting number, so a rule that ranks by value ranks
a row of equal numbers. The digest says it exactly rather than approximately:
on the corridor, epsilon 0.1, 0.5 and 0.9 walk one path, softmax at two
temperatures a hundredfold apart walks another, and the count bonus at three
confidences walks a third.

Measuring it needed a grid to measure it on. Every grid this project had can be
solved by an agent that wanders, so all four ways read the same on all of them.
The corridor is one folded path forty eight steps long where nothing pays until
the end, and there the median run first reaches the goal on episode 18, 21, 6
and 1. All four end with the optimal policy, and what differs is the seventeen
thousand five hundred steps epsilon-greedy spent before there was anything to
learn from.

**A value network**, as `deep-q`, with a replay buffer and a target network as
two settings of one agent rather than two agents. The ablation is the point,
and it reads differently on two environments. On the cart pole neither piece
alone does anything: the median of the last fifty episodes is 66.3 with both
and 9.0, 9.4 and 8.8 with either one missing, where a pole nobody balances
falls in about nine steps. On the cliff walk all four medians are the same
number, -55.2, -49.6, -55.6 and -55.3, and only the tail moves: with neither
piece four seeds of ten end past -240, and with both one does. The difference
is the representation. A one-hot grid gives each state close to its own
parameters, so the shared weight problem barely arises and what is left is the
occasional run where it does. The honest note stands: nothing here solves the
cart pole. The best setting reaches 66 steps of a possible 500, where
`tile-sarsa` reaches 498.

**Intra-option learning**, as `intra-option-q`. One real step is evidence about
every option that would have taken that action there, so the states an option
passes through are credited for it rather than only the state it started in.

It was built to test a reading rather than to improve a number. The algorithms
page reads the cost of having options as the price of exploring, on the
evidence of an exploration ladder, and that reading predicts that fixing the
credit assignment moves the early episodes and leaves the late ones alone. It
does: 15% shorter episodes in the first hundred, 24% in the second, within
noise by the fourth, and 0.54 recovered of the 2.02 that having the options
costs at all. Two independent measurements now say the cost is the commitment
and not the credit.

**Prioritised sweeping**, as `prioritised-sweeping`. It needs a median of 880
updates to solve the Dyna maze where `dyna-q` needs 8520, and its worst seed
costs less than the uniform planner's median one. It also stops: once nothing
in the model would move, the queue empties and it makes about one update every
thousand steps where `dyna-q` keeps making six a step forever. That is also its
weakness, and one seed of ten settles two steps off the shortest route and
never looks again.

**Options**, as `options-q`, on the four rooms grid the literature uses for
them. The eight hallway options are read off the layout rather than written
down, and the same construction finds none on the Dyna maze and none on the
cliff walk. The result is not the expected one: they cost 2.57 return while
learning, and a ladder of exploration rates says the cost is the price of
exploring. An exploratory choice that lands on a long option commits several
steps in one direction and is paid for several times over.

**Eligibility traces**, as `sarsa-lambda` and `q-lambda`. One dial rather than
a whole number of steps. The sweep found the same interaction with the step
size that n-step SARSA had, including the same exception, which is the reason
[algorithms.md](algorithms.md) now has the two of them in one section.

**n-step tree backup**, as `tree-backup`. Off-policy n step learning with no
importance ratio anywhere in it. On the cliff walk it reaches -15.00 with
nothing stuck, where the method below is stuck on all ten seeds.

**Off-policy Monte Carlo**, as `off-policy-mc`, with both estimators. The
variance lesson is measured rather than asserted: the ordinary estimator's
worst cell on the maze reads four trillion on a problem whose best possible
value is 0.513.

---

## Looked at, and left

### NumPy

**Measured.** One agent, two hundred episodes of the cliff walk, seed 1:

| agent | time | steps |
| --- | ---: | ---: |
| q-learning | 0.05s | 5,159 |
| sarsa | 0.06s | 6,193 |
| dyna-q, twenty planning steps | 0.16s | 3,721 |
| monte-carlo | 0.20s | 21,863 |
| random | 0.59s | 96,745 |
| **reinforce** | **37.41s** | 74,726 |

The tabular agents are fast enough that a dependency would buy nothing: a run
that takes a twentieth of a second does not need to take a thousandth.

The network agents are a different matter. REINFORCE is seven hundred times
slower than Q-learning on the same environment, and almost all of that is
Python running loops over lists of floats. NumPy would give most of it back.

It is still not here, and the reason is the claim rather than the speed. A
project whose point is that every algorithm is written out and can be read
gains nothing from a gradient engine that is a wrapper around somebody else's
gradient engine. The cost of that decision is the table above, and it is the
reason the cart pole numbers in
[algorithms.md](algorithms.md) come from six hundred episodes rather than six
thousand.

**What would change this.** A network agent that somebody actually wanted to
use, rather than one that exists to show what a policy gradient is. At that
point the honest move is an optional import with the pure Python kept as the
readable version, and a test that holds the two to the same numbers.

### A hashing tile coder

The usual implementation hashes a cell into a table of fixed size and accepts
that two cells sometimes collide. That is necessary when the space is large.

Four dimensions at eight cells each, across eight grids, is fifty two thousand
switches: a list of floats under half a megabyte. Nothing collides, so a fault
in an agent is a fault in the agent.

**What would change this.** A dimension count above about six, where the exact
index stops fitting in memory. Nothing here is near that.

### An actor critic that updates every step

**Measured.** Four hundred episodes of the cliff walk took **220 seconds**
updating after every step and **12 seconds** updating once per episode. Nothing
about the targets changes between them: each one is still a single step of
reward plus the value of where that step landed.

An update touches every weight of both networks. Doing that once per
environment step rather than once per episode is a factor of a few hundred on
an environment with long episodes, and it buys nothing that could be measured.

### A curses interface

The live display redraws a block of lines in place with two escape codes, and
falls back to plain lines when it is not writing to a terminal. A full screen
interface would need a terminal to test against, and it would make
`rel train` unusable in a log, which is where most runs end up.

### An `optimism` default that hides the artefact

A table of zeros is not a neutral start on an environment where every reward is
negative: an untouched entry is worth more than any entry that has been
learned, so the greedy policy prefers whatever it has never done.

The obvious answer is to start the table below every real value. It is not the
default, because the artefact is worth seeing: it is why `train` and `evaluate`
are two runs and both are reported, and it is why `rel train` says how many
evaluation episodes ran out of steps. `optimism` is an argument, and
`docs/algorithms.md` says what it does.

---

## Open questions

None at the moment. The two that were here are below, with what the
measurement said.

---

## Questions that were open, and what the measurement said

**Why does one cliff walk seed of twelve still never reach the goal?** It does,
at episode 784. The measurement that called it lost runs for four hundred, and
by a thousand every seed of the twelve has reached the goal.

The three possibilities were more entropy, more episodes, or something that is
not a knob at all. It is more episodes. Entropy does move that seed, and a long
way: it first reaches the goal at 784 at the default of 0.05, at 172 at 0.1, at
92 at 0.2 and at 25 at 0.4. But over all twelve seeds raising the default buys
nothing reliable. At 0.1 the one seed whose greedy policy never finishes goes
away and at 0.2 it comes back, so the dial does not control that, and both
settings cost 8 and 27 return while learning and leave the finished policy no
better. The default stays at 0.05 and
[algorithms.md](algorithms.md#the-seed-that-never-reached-the-goal-reaches-it-at-episode-784)
has the tables.

**Should the digest cover the agent as well as the environment?** Neither, and
both. Merging the two would have changed what the existing digest means, and
every number on the algorithms page was compared against it, so all of them
would have silently stopped being comparable.

There are now two digests. The path digest is unchanged and a test pins it to
the value it had before any of this. Beside it sits a digest of what the agent
learned: the table for a tabular agent, the weights for one over a tile coder,
every parameter of every layer for a policy gradient. Q-learning and expected
SARSA fed one identical step already differ in it, and the path digest cannot
tell them apart. An agent that keeps nothing reports nothing rather than the
hash of nothing.

[docs/architecture.md](architecture.md) says what each covers.

**Should `n-step-sarsa` default to a smaller `n`?** Swept, and the answer is
that the question was half of one. `n` and the step size trade against each
other, and [algorithms.md](algorithms.md) has both sweeps. At a step size of
0.5 one step wins on all five grids, which means the n step return earns
nothing at the setting the rest of the table uses. At 0.1 the sign flips and
four steps wins by a wide margin, because one step has not carried the value
far enough in five hundred episodes.

The default is now n=2 at a step size of 0.2. Over five grids and thirty seeds
each, the count of policies that never reach the goal falls from 47 to 10. Two
explanations were tested before that one and neither survived the measurement.

**Why did REINFORCE fail on three seeds of six on the cliff walk?** Three
of those failures were one fault and it is fixed. `_returns` read an episode
that the step limit cut off as an episode that ended, and
[algorithms.md](algorithms.md) has what that did to the weights. The guess
recorded here at the time was wrong, and the way it was wrong is worth keeping:
it said standardising removes the signal when every step is equally bad, and
the measurement says the signal was large and pointed backwards.

What was left was seed 3, and that was a different question. That agent never
reached the goal once in four hundred episodes, before the fix or after it, so
there was no return that reached the goal for any rule to share out. The
entropy bonus is what holds the policy wide enough to find the goal, and it is
swept now: at 0.01 that seed first reaches the goal at episode 1088 of 1200,
and at 0.05 it reaches it at episode 173 and finds the optimal -13. The
registry default is 0.05 and [algorithms.md](algorithms.md) has both tables.

One seed of twelve does not reach the goal inside four hundred episodes, and
the trade is written down rather than tuned away: 0.05 loses one seed where
0.01 loses three, and the policies it does find are 0.51 blunter. That seed
reaches the goal at episode 784, which is the next question below.

**Is the actor critic here wrong, or only untuned?** Measured, and the answer
is neither on its own. The entropy bonus was swept on five seeds and six
hundred episodes of the cart pole, and [algorithms.md](algorithms.md) has the
table. The mean does not rise with the setting: 0.05 gives 146.8, 0.10 gives
33.3, and 0.20 gives 207.2. Two settings reach five hundred on one seed each
and none reaches it on two.

So it is not a default away from working, and it is not put together wrongly
either: it finds the optimal policy on the cliff walk, and REINFORCE shares
this file, this encoder and this optimiser and reaches five hundred on all five
seeds. The difference is the one thing the two do not share, which is a target
bootstrapped from a value network that is still wrong. A target network or
n-step returns are what would test that.

A target network is now in the project, on `deep-q` rather than on this critic,
and what it does there is evidence for the reading rather than against it: on
the cart pole it buys nothing on its own and a great deal beside a replay
buffer. Putting one on this critic is the test that is still not run.

---

## Things that would be nice and are not milestones

- A `--out` that writes a run to a file, and a `rel replay` that reads one.
- Sweeping a setting from the command line, with the table falling out.
- An environment described entirely by a text file, so that a new grid needs no
  Python at all.
- A confidence band on the compare chart rather than a mean line.
