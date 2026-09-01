<div align="center">
  <a href="../README.md"><b>Reward Enforced Learning</b></a>
</div>

# Milestones

What is not built, what was looked at and deliberately left, and the open
questions. Everything here that has a number behind it says what was measured.

---

## Not built yet

Nothing, for the first time since this page was written. The one entry that
was here was continuous actions, and it is below.

That is a statement about this table rather than about the subject. Plenty is
missing: a policy gradient method that works on a hard problem, an
environment with images, a buffer that draws in the log of its size rather
than in its size. None of those is written here, because a list of everything
not built is a list of the subject.

---

## Built since that table was written

**A replay buffer that draws by priority**, as `priority` and `weighting` on
the buffer `deep-q` already had. The entry above named it as missing. Drawing a
step in proportion to the size of the error the agent last made on it spends
the batch where there is something left to learn, and on the cart pole at 150
episodes it takes the greedy return from 24 to 99.

The half worth having built it for is the other one. Drawing by priority
changes the probability each step arrives with, so the average over a batch now
estimates a different thing and the agent settles somewhere else. That is not
an argument on this page, it is a number: on a fixed set of twenty rewards
whose mean is 0.2500, an uncorrected priority draw settles at 0.3442 with a
spread of 0.0092. It is not noisy, it is wrong.

Correcting it takes the cart pole to 188, which is the only one of the three
settings whose interval is clear of zero. So the correction is not a tax paid
to be principled. It is the setting that wins.

**Continuous actions**, as `pendulum`, `pendulum-levels` and
`gaussian-actor-critic`. The entry above said the whole action interface was
`Discrete` and that changing it should be a large change. It was: `Env` and
`Agent` both carry a second type parameter now, `DiscreteEnv` and
`DiscreteAgent` are those classes with it filled in, and every one of the
thirty six agents that came before is one of the second kind.

What the track measured is not what it set out to. The plan was that cutting a
box into levels costs something and the question was how much. It costs
nothing at a large budget and it costs a great deal at a small one, in both
directions: at three hundred episodes a switch beats seventeen levels by 570
return and at nine hundred nine levels beats the switch by 217.

The continuous agent does not solve the pendulum. Neither does the discrete
actor critic on the same problem cut into levels, and both are behind acting
at random, while the agents that keep a value and rank it learn it. That is
the second environment on which the reading is the same, and the first is
already on the algorithms page under the cart pole.

Two faults came out of building it. A recording of a continuous run could be
written and not read back, because the action went into the file through an f
string and came out through `int`. And a demonstration printed a torque to
seventeen digits, which pushes the reward off a narrow terminal.

**State aggregation and a Fourier basis**, as `aggregated`, `fourier-sarsa`
and `fourier-q`. Two more ways of approximating a value, at the two ends of how
much design they need.

Aggregation is the smallest there is: put the states in groups and keep one
number for each, so a thousand cells becomes fifty numbers and the count is the
whole design. What a staircase of `n` steps can say about a line is arithmetic
rather than a run, and the floor and what an agent reaches have their best at
different counts: the floor halves with every doubling and what `linear-td`
reaches turns back up past fifty groups.

A Fourier basis is the other end. Every feature is a cosine wave over the whole
box and an order is the whole design, with no bins, widths, centres or offsets
to get right. Its one attached setting, dividing the step for each feature by
how fast that feature waves, is measured both ways over ten seeds and **the
sign flips**: it loses at order one, wins at order three, loses at order five
and is not told apart at order seven, with three of those four intervals clear
of zero.

Both sides of that had to be swept over step sizes or the answer is a different
question. Every scale a Fourier basis asks for is at most one, so scaling makes
each step smaller as well as uneven, and the first probe of it found the scaled
side far ahead at order five where the swept comparison puts the flat side
ahead by 19.7.

**A coder can ask for a different step size on each feature**, which is what
the basis above needed and what nothing else here wants. `GradientTD` had to be
changed as well as `SemiGradientTD`: the base class reads the scales in its
constructor, so a subclass that forgot to apply them would keep a number it
never used and quietly take the wrong step.

**Two problems dynamic programming can answer exactly**, as `gambler`,
`fair-gambler` and `rental`.

The gambler stakes part of a capital on a coin, and its famous jagged staircase
of a policy is mostly the solver. Varying two things that cannot change the
answer, the sweep's tolerance and which of the two solvers ran it, moves 72 of
the 99 stakes at a fair coin and 36 at the coin the literature draws. At a fair
coin nothing is decided at all: every stake is exactly as good, and the widest
gap between the best and worst anywhere on the board is 8e-13, which is the
sweep's own rounding.

A fair coin also gives this project its second closed form. A fair game stopped
at either end is worth the same however it is played, so a capital is worth that
capital over the goal, and the sweep agrees to 1.6e-12 over all ninety nine.

The car rental is the one model here that is a distribution rather than a
handful of branches written out. Its two questions are what the van buys and at
what price it stops being used: 2.42 a day at the book's price of 2 a car,
4.02 free, and nothing at all at 10, where the value falls back exactly onto
the row with no van.

Staking nothing is not an action, and the reason is measured. It is a loop of
one state, so at a discount of one it ties with the best real stake everywhere,
and built that way value iteration staked nothing at 35 of the 99 capitals and
the policy it reported never reached an ending from anywhere.

**Blackjack with the model the book says is awkward to write**, as `blackjack`.
The book uses it to introduce Monte Carlo because the dealer plays out his whole
hand. Writing the model out anyway means the optimal policy and the value of the
deal are known before a card is dealt, and an agent that learns from episodes
can be scored against the answer rather than against another agent.

Solved, it reproduces the book's figure square for square on both halves of the
board, which is the check that the model is right rather than merely consistent.

Against it, Monte Carlo control at five hundred thousand hands reaches -0.048
where perfect play is -0.047, eight or nine squares from the optimum. The
docstring claim that a fixed step size is what control needs even in a fixed
environment does not survive: the running average wins at the large budgets and
a fixed step wins only at the small ones.

**n-step Q(sigma)**, as `q-sigma`, which is the family with tree backup at one
end and sampling at the other. Sutton and Barto raise the middle as a question
and leave it open, and on the cliff walk over ten seeds the answer is no: tree
backup reaches -13.800 and every other sigma is behind it, with 0.75 behind by
1.600 on an interval clear of zero. The schedule the book suggests, sigma
falling from one to nothing, lands with the middle rather than with either end.

The collapse to tree backup at sigma of nothing is exact against a greedy target
and the last bits against an averaged one. Writing it found a fault of its own:
the first version asked the target policy for its shares three times inside one
update, and a greedy target breaks ties by drawing, so the three answers could
disagree and each spent randomness.

**A sweep that did not settle guessed at why.** Both messages said the policy
probably never ends. There are two reasons and they need different answers, and
the gambler at a fair coin is where the guess became visibly wrong: every policy
there ends, and an undiscounted walk over a hundred states settles slowly enough
that a tight tolerance runs past twenty thousand sweeps while still converging.

**Continuous integration runs the number checker.** It had existed for a while
and nothing ran it, because a whole run of the page is three hours. Five fast
commands and three tables is twenty seconds, and `--strict` is what turns a
report into a gate: without it a table nothing accounts for is printed and the
run exits zero.

**The deadly triad**, as `baird`, `linear-td` and `gradient-td`. Function
approximation, bootstrapping and off-policy learning were all here separately
and nothing said what happens when they meet. Baird's counterexample is seven
states that pay nothing, whose answer is zero and whose approximation can say
zero exactly, and on it `linear-td` reaches a value error of 1e22 in twenty
thousand steps while `gradient-td` stays at 1.9.

The result worth having is not the divergence, which is in every textbook. It
is that **the divergence has a discount below which it does not happen**, at
fifteen seventeenths, and that the number falls as the counterexample grows
and is above one for four upper states or fewer. `scripts/measure_triad.py`
finds it by bisecting on a run and `rel.envs.baird` derives the same number
from the trace of a three by three matrix, and neither reads the other.

Two faults came out of building it. The behaviour policy was a constant of six
sevenths, which is right at six upper states and leaves the agent three times
more often in one state than another at twenty, so the closed form described a
different problem than the run. And the readme said thirty two agents in two
places and thirty four in a third, because the test that held the count asked
whether the right number appeared anywhere in the file.

**A command that checks the numbers on the algorithms page.** It runs every
command that page names, then puts each of its fifty one tables against every
output and attributes it to the command that accounts for most of its numbers.
`scripts/check_numbers.py --list` says what would be checked without running
anything, and `--cache` keeps what each command printed so that checking a fix
costs seconds rather than the three hours a whole run takes.

Matching rather than position, and the first version did it by position. Two
thirds of that first report was a table sitting under a command it did not come
from, because `The same table, four more environments` writes one command and
then four tables and the other three commands were nowhere on the page.

What it found was not what the exercise expected. Nothing had drifted: 34 of
the 34 tables that state results are wholly accounted for by a command the page
names. What it had instead was tables whose command it never named, or named
wrongly:

- Four tables under one of their four commands, for nineteen tracks.
- Ten rows of an epsilon sweep under a command that takes the default episode
  count where every cart pole row in it is 600, so that command printed no row
  of the table under it.
- The importance sampling table under `measure_importance.py` where the prose
  beside it said 1200 episodes and the default is 1500. The current code at
  `--episodes 1200` reproduces every number in it exactly.
- Three grids of exploration rules with no console block above them at all.
- A per seed row of updates that no option of any script could print, which
  `measure_sweeping.py --each-seed` now prints.

Every one of those was run before its command was written down.

Seventeen tables carry a comment saying they are not checked and why, because
some are differences between two runs, or arithmetic, or seconds on a machine.
The reason is required and printed, because a table that exempts itself
silently is how a number that moved would hide. A marker can name one column
rather than the whole table, which keeps every number of a table whose only
unfixable cell is a timing, instead of dropping the model steps and the returns
beside it.

Nine faults in the tool were found by running it and are fixed. It called
slow commands broken, named coincidences as sources, deleted the cache when
asked a narrow question, trusted a cache made from different code, and would
have run itself for hours recursively. A marker with no end swallowed every
command and table under it and reported the page clean. A marker naming a
column its table has not got exempted nothing and said nothing. A command that
ran out of time was not written down, so every resumed run spent the whole
budget on it again. And the listing held each reason in a column, which is as
wide as its widest cell, so the command behind each table sat off the side of
the terminal.

**A second way of making features, and the reason tile coding wins.** Radial
basis features answer how close a point is to each of some centres, rather than
which cells it is in. `LinearAgent` now takes either, through a `Coder`
protocol that neither encoder imports, and every run over a tile coder gives
the numbers it gave before to the last bit.

What the measurement said is not what the exercise was for. On the cart pole a
tile coder has 52488 features and takes 70 microseconds a step; a radial basis
has 1296 and takes 3511. A tile coder is not cheap because it has few features.
It is cheap because it works out which eight switches are on by arithmetic and
**never asks the other 52480 anything**, and a radial basis has no such route.

The plan was to give it one, by keeping only the largest few values. That
cannot work and the reason is worth the whole track: there is no way to know
which centres are far away without measuring the distance to all of them, which
is three quarters of the step. It buys eleven percent of a step on the
mountain car and thirteen on the cart pole, and it pays for that by putting
back the boundary the encoder exists to remove. It is off by
default, and a test named after it sweeps a line across the box to find the
step it makes.

On the mountain car at sixty episodes the radial basis learns better, by 21.3
points of return over ten seeds. The default width is three quarters of the
spacing between centres, which was measured on two environments rather than
reasoned: one whole spacing looks right and loses on both.

**A faster gradient engine that computes the same numbers.** Three changes,
none of them clever: the lists an inner loop reads are pulled into locals in
`linear` and in `Adam`, and an internal constructor skips the copy and the
shape check that the public one owes a caller. A whole `reinforce` run goes
from 0.142s to 0.131s, a 48 to 16 backward pass from 2.062s to 1.868s, and Adam
from 5.359s to 4.289s. **Both digests are byte for byte the same before and
after**, which is the only reason any of it is worth having: an optimisation
that reassociates a sum is faster and gives different answers, and on a
learning agent different answers look exactly like another seed.

Three things about the measuring came out wrong first and are written down.
A single timing of unchanged code moved by a tenth of a second depending on
what had run before it. `cProfile` made the optimiser look like the place to
work, and an eighteen percent saving there moved a whole run by nothing the
whole-run timing could see, so the benchmark grew a line for it. And a slice
that makes a wide layer faster makes a narrow one slower.

**A comparison rather than two descriptions**, in `rel compare`. It printed two
means and two standard errors, which describe two sets of numbers, and a reader
takes a comparison from them anyway. It prints the paired difference, a 95%
bootstrap interval and a permutation p value now.

The findings are about this project rather than about any agent, and two of
them are uncomfortable.

**A paired test over five seeds cannot report a p below 0.0625**, whatever the
difference is, because there are only thirty two ways to arrange five signs.
Several measurements here ran five seeds, so none of them could have reached
0.05. Six is the fewest that can.

**A 95% bootstrap interval on five seeds excludes zero about one time in five
when there is nothing there.** Measured by running `q-learning` against a copy
of itself two hundred times: 39 of 200 rather than the 10 a calibrated interval
would give. At ten seeds it is 12 of 200, which is right.

The same measurement gives the number a claimed difference has to beat. Two
identical agents on the cliff walk differ by a median of 3.14 over five seeds
and by up to 16.63, and over ten seeds by 2.32 and up to 8.99.

**So ten seeds rather than five.** It costs twice the time and it moves the
interval from wrong one time in five to wrong one time in seventeen.

**Average reward**, as `differential-q`, and a task with no ending to need it.
Every grid here has a goal, so an episode ends and the discount changes what a
run is worth without changing which policy is best. On a task that never ends
it changes which policy is best.

`loops` is one decision made over and over. A short loop pays 1 a step and a
long one pays 2 a step, and their discounted values are equal at 0.7394. Below
that the exactly optimal policy takes the loop paying half as much, which is
the correct answer to the question the discount asked.

The threshold depends on the environment. Lengthen the long loop to eight steps
and it still pays a quarter more per step, and a discount of 0.9 takes the
short one. That is the same shape as `kappa` on Dyna-Q+, met a second time from
a different direction: **a setting that has to be right against numbers the
agent cannot see is a setting that will be wrong somewhere.**

`differential-q` subtracts the rate it is collecting instead of discounting,
and takes the better loop at every setting tried. The rate it learns is the
rate it collects, 2.000 against a true 2.000, so it is an estimate of something
real rather than a bias term that happens to work.

Writing that section found a fault in the environment itself. Its suggested
discount was a fixed 0.9, which is right at the default length and takes the
worse loop at a length of eight. It is computed from the crossover now, and the
environment can only do that because it is a toy that knows its own answer.

**Planning at the moment of choosing**, as `mcts`. It runs simulations from the
state it is standing in and acts on what they said, where `dyna-q` and
`prioritised-sweeping` spend the same work in the background on a table. All
three are counted in model steps, because a table of returns against episodes
lets an agent that asks a model a thousand times a step look free.

Two things came out of measuring it and neither was the expected one. On the
Dyna maze every setting settles in about the same place and they are two orders
of magnitude apart in cost: the search reaches sixteen steps for 94,296 model
steps and prioritised sweeping reaches seventeen for 3,575, having had to learn
its model rather than being handed one. And more simulations buy nothing, so
what carries the agent is not the search from where it stands.

It is the tree. With `reuse=off`, which is decision-time planning with nothing
else in it, the same code with the same model never settles: 330 steps against
a shortest route of fourteen, three seeds of three with a policy that never
reaches the goal, and forty two million model steps to fail.

The reason is the rollout. A random policy of thirty steps reaches an ending on
none of two hundred tries on either the maze or the cliff walk, so the tail of
a simulation is nearly always the same number. On the cliff walk it is exactly
the same number, because every step pays -1 and the discount is one. **The
repair is a value at the leaf instead of a rollout, and it is not built.**

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

**A test for every measurement script.** Two scripts in `scripts/` measure
nothing, one drawing the readme banner and one running the gate that continuous
integration runs. Every other one is now loaded by a test. So they are: none.

The entry this replaces named them rather than counting them, because its first
version said eleven untested in one sentence and twelve in the next and neither
was right. `test_layering.py` still holds the list against what is really
there, and it now reads an empty list as the thing to check for.

Writing them found four faults in the tests rather than in the scripts, which
is the usual ratio here. A test that the two ways of shifting a tile coder
answer differently asked at one point and failed, because they agree at about
three points in ten. A test that a setting reaches an agent used the mountain
car, where a few episodes never reach the flag at any setting, so it passed on
a script that dropped the setting. And the updates per step of `options-q` are
below one rather than above it, because an option that runs for three steps
gives one update.

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

```console
$ rel train q-learning --env cliff --episodes 200 --seed 1
$ rel train sarsa --env cliff --episodes 200 --seed 1
$ rel train dyna-q --env cliff --episodes 200 --seed 1 --set planning_steps=20
$ rel train monte-carlo --env cliff --episodes 200 --seed 1
$ rel train random --env cliff --episodes 200 --seed 1
$ rel train reinforce --env cliff --episodes 200 --seed 1
```

<!-- not checked, column time: seconds belong to the machine, and the machine
this was measured on is not the one anybody reading it has -->
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
