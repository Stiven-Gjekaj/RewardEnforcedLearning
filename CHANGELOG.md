<div align="center">
  <a href="README.md"><b>Reward Enforced Learning</b></a>
</div>

# Changelog

All notable changes to this project are recorded here. The format is based on
Keep a Changelog (https://keepachangelog.com), and the project aims to follow
semantic versioning.

A commit carries no version prefix and changes no version. A version moves only
when something is released.

## Unreleased

### The first one

Everything below is new.

**The core**

- **A faster gradient engine that computes the same numbers.** The lists an
  inner loop reads are pulled into locals in `linear` and in `Adam`, and an
  internal constructor skips the copy and the shape check the public one owes a
  caller. Adam is 18% faster and a 48 to 16 backward pass 9%, and both digests
  are unchanged. `scripts/measure_engine.py` prints the timings and the digests
  together, because a faster engine that computes different numbers looks
  exactly like the same agent on another seed.
- **A seeded source of chance**, PCG-XSH-RR written out, with named
  independent streams. An environment and an agent take different streams of
  one seed, so the number of draws one makes cannot move what the other faces.
  A test holds the generator to the published output of the reference PCG32
  implementation.
- **An environment contract** that separates an episode ending inside the rules
  from one stopped by a step limit, and a base class that decides truncation so
  that an environment cannot forget to.
- **A model on every grid**, so `value_iteration` gives the best possible
  return exactly rather than an estimate.
- **A run digest**, so two runs can be compared without comparing prose.

**Environments**

- **A problem whose answer is arithmetic**, as `walk`: five cells in a line
  between two endings, one paying nothing and one paying 1. A policy that goes
  each way half the time reaches the paying end from the k-th cell exactly k
  times in six, so the true values are 0.167, 0.333, 0.500, 0.667 and 0.833
  with nothing computed to get them. Every other table in this project checks
  an agent against dynamic programming over the same model, which is a check
  against another computation. This one is a check against arithmetic, and a
  test holds the closed form against a sweep over the model so that a fault in
  either shows up as a disagreement.
- **A task with no ending**, as `loops`: one decision made over and over,
  between a short loop paying 1 a step and a long one paying 2 a step. Their
  discounted values are equal at 0.7394, and below that the exactly optimal
  policy takes the loop paying half as much. The agent is not going wrong
  there. The discount was part of the question, and 0.7 is a number chosen
  because it converges quickly. Lengthen the long loop to eight steps and a
  discount of 0.9 does the same thing. The `endless` tag now marks every
  environment whose spec says it cannot end, held to the spec by a test rather
  than to a list kept up to date by hand.
- **A corridor**, one folded path forty eight steps long with no branch
  anywhere in it and nothing paid until the goal. It exists because every other
  grid here can be solved by an agent that wanders, so all four ways of
  exploring read the same on all of them. A random policy covers a line of
  length n in about n squared steps, and a test measures the consequence: over
  twenty episodes of the step limit a random policy arrives at most twice.
- **Bandits.** The ten armed testbed, and a version where every lever wanders.
- **Grids.** The cliff walk, the windy grid, four rooms, the Dyna maze and the
  frozen lake, each written as a picture of itself.
- **Control.** A cart pole and a mountain car with the physics written out.
- **Three where the reward is not the point.** A boat race whose checkpoints
  can be farmed, a room with a vase on the short path, and a thermostat with a
  dial that makes the sensor report comfort. Each ships with the repair and
  with what the repair costs.
- **A ladder that varies how hard the agent optimises**, from a uniform policy
  to the best policy under the stated reward. `rel gaming --pressure` walks it
  and charts both numbers. The result is that the real objective falls the
  whole way on all three, and that the optimum is worse at it than a policy
  that optimises nothing: half a lap against none, a vase left standing seven
  times in a hundred against never, sixteen percent comfortable against three.

**Agents**

- **Estimating what a fixed policy is worth**, as `td`, `n-step-td`,
  `td-lambda` and `mc-prediction`. Control mixes two questions: when a policy
  improves, its estimates chase a moving target, so a measurement of how good
  the estimates are cannot say whether the answer moved or the estimate did.
  Prediction asks one question and the random walk above knows the answer.
  **TD's best is 0.0343 root mean square error and Monte Carlo's best is
  0.0590**, from the same episodes, and the one that uses what it already
  believes about where it ended up gets 42% closer than the one that waits to
  find out what the return was. Every row is a U in the step size, because a
  constant step size does not converge at all: it tracks, and the estimate
  ends in a band around the answer whose width is the step size. Crediting a
  state once for every visit rather than once for the first is worse
  everywhere but the smallest step, and at 0.2 it scores 0.2427 where an
  untrained table scores 0.2357, which is worse than not learning.
- **Crediting the middle of an option**, as `intra-option-q`. One real step is
  evidence about every option that would have taken that action there, so
  every one of them is updated whether or not it was the one running. This was
  written to test a reading rather than to improve a number: the section above
  reads the cost of having options off a ladder of exploration rates, and if
  that reading is right then fixing the credit assignment moves the early
  episodes and leaves the late ones alone. **It does.** The first hundred
  episodes are 15% shorter and the second 24%, and by the fourth block the two
  agents are within noise. It recovers 0.54 of the 2.02 that having the options
  costs at all, for slightly over twice the work: 1.64 applications of the
  learning rule for every step taken against 0.78. Two independent
  measurements now say the cost of an option here is the cost of committing to
  it while exploring, and not the cost of learning about it slowly.
- **A second way to make features**, as `rbf-sarsa` and `rbf-q`. Radial basis
  features answer how close a point is to each of some centres, where a tile
  coder answers which cells it is in, so there are no boundaries: a point a
  hair away has a value a hair different, everywhere. They are the same two
  agents over the other encoder, and nothing in the agent knows which it is
  talking to. A run over a tile coder gives the numbers it gave before the
  agent was generalised, to the last bit. **The feature count says the
  opposite of the cost.** On the cart pole the tile coder has 52,488 features
  against 1,296 and costs a fiftieth as much per step, because a tile coder
  never asks a feature that is off: it works out which eight switches are on
  by arithmetic and never touches the other 52,480. Dropping all but the
  nearest few centres was going to buy that sparseness back and it buys
  eleven percent of a step on the mountain car and thirteen on the cart pole,
  because three quarters of a step is the distances and they are paid before
  anything can be dropped. It also brings a boundary back, of a size that
  depends on where the crossing is: over the unit square it runs from 0.006
  to 0.054, against the eighth of itself a tile coder's boundary costs. So it
  is off by default. The width is the whole of the setting: three quarters of
  the spacing between centres beats one whole spacing by 11.2 return on the
  mountain car over twelve seeds and by 267.6 on the cart pole over eight,
  with the interval clear of zero both times, and at two spacings eight of ten
  seeds never reach the flag.
- **Average reward**, as `differential-q`: Q-learning with the rate it is
  collecting subtracted from every reward and no discount anywhere. There is no
  discount setting on it, and that is the point of it. The rate is learned from
  the same error rather than averaged over the rewards that arrived, because a
  running mean would be the rate of the behaviour policy with exploration in
  it. On a task whose better policy collects exactly 2.000 a step it settles at
  2.000, so the number it subtracts is an estimate of something real.
- **Planning at the moment of choosing**, as `mcts`. It runs simulations from
  the state it is standing in and acts on what they said, where `dyna-q` and
  `prioritised-sweeping` spend the same work in the background on a table. The
  rule that picks an action inside the tree is the count-based exploration rule
  above, applied to the means in a node, so upper confidence selection inside a
  tree and upper confidence exploration in a grid are one piece of code. The
  tree is keyed by state rather than by path, so two paths that meet pool their
  evidence. Reading the policy leaves no trace: a copy of the tree, a copy of
  the generator, and the count of simulated steps put back. On the Dyna maze
  every setting settles in about the same place at two orders of magnitude
  apart in cost, and more simulations buy nothing, because what carries the
  agent is the tree rather than the search. With `reuse=off` the same code with
  the same model never settles at all.
- **Exploring by something other than chance.** `explore` is a setting on every
  tabular agent, and three rules answer it: epsilon-greedy, softmax, and a
  count-based bonus that adds a term shrinking as an action is taken.
  Optimistic initialisation is the fourth way and it is a starting value rather
  than a rule. Each rule answers two questions rather than one, because
  expected SARSA averages over the policy and an off-policy correction divides
  by it, so a rule whose two answers disagreed would put a bias in both and
  neither would report it. All thirteen tabular digests are unchanged by the
  default. The finding is that **the dial of a rule cannot help find a first
  reward, and only what the rule ranks by can**: before anything pays, every
  value in the table is the starting number, so epsilon 0.1, 0.5 and 0.9 walk
  one path, softmax at two temperatures a hundredfold apart walks another, and
  the count bonus at three confidences walks a third. On the corridor the
  median run first reaches the goal on episode 18, 21, 6 and 1, and all four
  end with the same optimal policy.
- **A value network**, as `deep-q`: Q-learning with a network in place of the
  table, and the two pieces that make one work as settings that can each be
  switched off. A replay buffer breaks up the correlation between steps that
  arrive one after another, and a target network stops the estimate and the
  thing it is fitted to from moving together. On the cart pole neither piece
  alone does anything: a median of 66.3 steps with both, and 9.0, 9.4 and 8.8
  with either one missing, where a pole nobody balances falls in about nine.
  On the cliff walk all four medians are the same number and only the tail
  moves, because a one-hot grid gives each state close to its own parameters
  and the shared weight problem barely arises. The same ablation on two
  environments gives opposite looking tables, and the difference is the
  representation rather than the algorithm.
- **Prioritised sweeping**, Dyna that replays the step that matters rather than
  one drawn at random. Work follows the change backwards through the model: it
  takes the largest change off a queue and then asks which steps lead into the
  cell it just moved. A median of 880 updates to solve the Dyna maze against
  `dyna-q`'s 8520, and its worst seed costs less than the uniform planner's
  median one. It also stops. Once nothing in the model would move, the queue
  empties and it makes about one update every thousand steps where `dyna-q`
  keeps making six a step forever, and that is also its weakness: one seed of
  ten settles two steps off the shortest route and never looks again.
- **Options**, as `options-q`: Q-learning whose choices last several steps. The
  eight hallway options of the four rooms grid are read off the layout rather
  than written down, and the same construction finds none on the Dyna maze and
  none on the cliff walk. An option that stops after every step is a primitive
  action, so the agent holding only those is Q-learning, to the last two
  decimal places on a whole grid. The measured result is not the expected one:
  the hallway options cost 2.57 return while learning, and a ladder of
  exploration rates says the cost is the price of exploring rather than
  anything about what was learned.
- **n-step tree backup**, which asks the same off-policy question as the entry
  below and never multiplies by a ratio. At n of one it is Q-learning cell for
  cell with a greedy target and expected SARSA with the exploring one, and both
  are checked exactly. On the cliff walk it reaches -15.00 with nothing stuck,
  where importance sampling is stuck on all ten seeds.
- **Off-policy Monte Carlo**, learning the greedy policy from episodes an
  exploring one collected, with both the ordinary and the weighted estimator.
  The variance the first is known for is measured rather than described: its
  worst cell on the Dyna maze reads four trillion, on a problem whose best
  possible value is 0.513.
- **Eligibility traces**, on SARSA and on Watkins' Q, in accumulating,
  replacing and dutch flavours. One dial from one step learning to crediting
  the whole episode, in place of n-step SARSA's whole number. At a decay of
  zero each is the one step agent it was built on, cell for cell, and a test
  holds it to exactly that.
- Four bandit agents, six tabular ones, Dyna and Dyna-Q+, semi-gradient SARSA
  and Q-learning over a tile coder, REINFORCE and an actor critic over a
  gradient engine written out here, a random baseline, and the exact optimal
  policy from the model.
- Dynamic programming: value iteration, policy iteration, and the exact value
  of any fixed policy, which says outright when a policy never finishes rather
  than returning a large negative number.

**Writing down what happened**

- **Two digests rather than one.** The path digest hashes the transitions and
  says whether two runs did the same thing. Beside it sits a digest of what the
  agent learned, which says whether two agents came to the same conclusion.
  Merging them would have changed what the first one means, and every number in
  the documentation was compared against it.
- **Recording a run**, with `rel train --out`. The digest at the top of the
  file is a claim the file makes about its own contents, and `rel replay`
  checks that claim rather than trusting it. The first four fields of a step
  line are exactly what the digest hashes, so no part of the reader has its own
  idea of how a transition is spelled. A name ending in `.gz` is compressed:
  fifty episodes of the cliff walk go from 34822 bytes to 2525.

**The command line**

- **`rel compare` says whether the difference is real**, with a paired
  bootstrap interval and a permutation test rather than two means and two
  standard errors. The comparison is paired because both agents meet the same
  seeds, which matters here: one agent's ten cliff walk seeds run from -20 to
  -545. A run whose seed count cannot reach 0.05 is told so, and the reason:
  a paired test over five seeds has thirty two sign patterns and the smallest p
  it can report is 0.0625. Measuring an agent against a copy of itself two
  hundred times says the rest. At five seeds a 95% interval excludes zero on
  39 of 200 when nothing is there, and at ten seeds on 12 of 200.
- `rel train`, `rel compare`, `rel solve`, `rel demo`, `rel list` and
  `rel gaming`, with learning curves drawn out of braille dots, grid policies
  drawn as arrows, and a value map that says which cells the agent has never
  stood in.
- `rel sweep` varies one or two settings and prints the table. Two settings
  give every pair, which is the point of sweeping two: the settings here trade
  against each other, and varying them one at a time misses that.
- `rel replay` reads a recorded run back and draws it.
- **A grid can be written in a text file** and given to any command with
  `--env-file`, so a new environment needs no Python at all. Every built in
  grid ships as one, checked against the environment it names on its model
  rather than on its text.
- **The compare chart draws the best and the worst seed** behind the mean. On
  the cliff walk that line is smooth while the runs under it swing by a hundred
  and fifty.

**Checking the documentation**

- **A command that checks the numbers in the documentation.**
  `scripts/check_numbers.py` runs every command a page names, then puts each of
  its tables against every output and attributes it to the command that
  accounts for most of its numbers. `--list` says what would be checked without
  running anything, and `--cache` keeps what each command printed, because a
  whole run of the algorithms page is hours and the answer stops being true as
  soon as the page is edited.

  Matching rather than position, and the first version did it by position. Two
  thirds of that report was a table sitting under a command it did not come
  from, because one section writes one command and then four tables and the
  other three commands were nowhere on the page.

  **Almost nothing had drifted.** That is the result of the exercise. What the
  pages had wrong was where their numbers came from rather than what they were:
  four tables under one of their four commands, three grids of exploration
  rules with no console block at all, five rows of a crossover table, two
  blocks naming a command at the wrong setting, and a closing index that named
  ten commands the page does not run. Every table with no command that could
  print it turned out to be right, and two of them gained one: the exploration
  bonus table is `scripts/measure_shortcut.py` now, and the collapse of options
  into Q-learning is a section of `scripts/measure_options.py`.

  A table can say it is not checked, or name a column that is not, and the
  reason is required and printed. Seventeen tables of the algorithms page say
  so, because some are differences between two runs, or arithmetic, or seconds
  on a machine.

  Nine faults in the tool were found by running it. The worst is that it ran
  `git clone` and `pip install .` when pointed at the readme, because those are
  in a console block and it ran every command in every console block. It runs a
  script under `scripts/` or the package's own command line and nothing else
  now. The rest: it called slow commands broken, named coincidences as sources,
  deleted the cache when asked a narrow question, trusted a cache made from
  different code, would have run itself recursively, let a marker with no end
  swallow the page, let a marker name a column that is not there and say
  nothing, forgot a command that ran out of time so every resumed run paid the
  budget again, and held each exemption's reason in a column wide enough to
  push the commands off the screen.

  `--doc` takes any page. The readme, the gaming page, the milestones and the
  grid page all come back clean, in under two minutes each.

### Faults found while building it, and fixed

- **REINFORCE read the step limit as an ending.** The return of an episode
  that was cut off reached back through it as if its future were worth zero. A
  failed cliff walk episode is five hundred steps of -1, so standardising those
  returns gave the last steps a weight near +3.2 and the first steps a weight
  near -0.8. That tells the agent to repeat whatever the step limit stopped it
  doing. Three seeds of six never reached the goal, and one does now. The
  actor critic in the same file always estimated the tail correctly, which is
  what made the difference visible.
- **The tile coder threw away most of its resolution.** The shift of each grid
  was not taken modulo one cell, so the later grids were displaced several
  whole cells past the space allocated for them and every point out there was
  clamped into one cell. The last of eight grids could reach 945 of its 6561
  cells. Every agent still learned, and every curve still went up. Found by a
  test that removed the input clipping and still passed, which meant the
  clipping was doing nothing.
- **A policy gradient took the log softmax twice.** The network already returns
  log probabilities and the loss applied the operation again. The gradient
  still pointed roughly the right way, so the agent still learned. Found by the
  linter noticing an unused import.
- **n-step SARSA updated the wrong cell.** It wrote down the action it expected
  to take rather than the one that was taken, which is only correct while the
  loop takes the action it was handed. Both on-policy agents were rewritten to
  wait for the action instead of promising it.
- **The boat race reported two hundred laps in two hundred steps.** The lap
  marker is not part of the state under the gameable reward, so the model had
  nowhere to put it, and the audit was reading the zero that left behind.
- **Policy iteration never returned on the cliff walk.** It began from "take
  action zero everywhere", which walks into a corner and stays, and evaluating
  that policy with no discount does not converge. It now begins from a policy
  built by a backward search, which reaches an ending by construction.
- **Evaluating a policy failed over a corner the policy never visits.** The
  sweep covered every state, so a policy that circles somewhere unreachable
  stopped the whole evaluation converging. `rel train q-learning --env cliff`
  reported no exact value at all until this was fixed.
- **Drawing a policy taught the agent something.** The renderer asked for the
  greedy action of every cell, and the accessor behind it made a table row when
  there was not one. The value map then reported that the agent had stood in
  all eleven cliff cells and that they were the best cells on the grid.

### Measurements that did not come out as expected

- **The tile coder offset rule appeared to be wrong.** A measurement of how
  evenly the standard odd displacement generalises said it was worse than the
  naive alternative in four dimensions, 2.53 against 1.13. It was measuring the
  clamping fault above. With that fixed the rule is better in both, 0.97
  against 1.13.
- **Four agents made a table row for a cell they never stand in.** Every one of
  them computes its target from the value of the state it is moving to, and
  read through `values` that lookup makes the row. After two hundred episodes
  of the cliff walk the table held 38 states where the agent had acted in 37,
  and the extra one is the goal. `knows` then said yes for a cell the agent can
  only arrive at. This is the second time this project has had this fault, and
  the first is why `peek` exists. Found by a test written for something else:
  Watkins' Q with no decay has to be one step Q-learning cell for cell, and it
  was not, by exactly one row.
- **The n of n-step SARSA was measuring the step size.** The agent took four
  steps at a step size of 0.5, and it got stuck more often than any other
  agent on four grids of five. The obvious reading is that four is too many.
  It is not: at a step size of 0.1 four steps beats one step by a wide margin,
  and one step gets stuck on 23 of 30 windy grid seeds where four gets stuck
  on none. The two settings trade, because an n step return carries a value n
  cells further and n times the spread. Two other explanations were tested
  first and neither survived.
- **Six seeds said the entropy default was a clean win.** At 0.05 the first
  sweep read -13 on all six cliff walk seeds: the optimal policy every time.
  Twelve seeds says it loses one seed and that the policies it finds are 0.51
  blunter, because seeds 7 to 12 are nothing like seeds 1 to 6. Nothing went
  wrong in the first measurement. It is what six samples of a noisy thing look
  like. The number of seeds is now part of every claim on that page.
