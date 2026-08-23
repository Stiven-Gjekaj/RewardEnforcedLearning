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

- **Bandits.** The ten armed testbed, and a version where every lever wanders.
- **Grids.** The cliff walk, the windy grid, four rooms, the Dyna maze and the
  frozen lake, each written as a picture of itself.
- **Control.** A cart pole and a mountain car with the physics written out.
- **Three where the reward is not the point.** A boat race whose checkpoints
  can be farmed, a room with a vase on the short path, and a thermostat with a
  dial that makes the sensor report comfort. Each ships with the repair and
  with what the repair costs.

**Agents**

- Four bandit agents, six tabular ones, Dyna and Dyna-Q+, semi-gradient SARSA
  and Q-learning over a tile coder, REINFORCE and an actor critic over a
  gradient engine written out here, a random baseline, and the exact optimal
  policy from the model.
- Dynamic programming: value iteration, policy iteration, and the exact value
  of any fixed policy, which says outright when a policy never finishes rather
  than returning a large negative number.

**The command line**

- `rel train`, `rel compare`, `rel solve`, `rel demo`, `rel list` and
  `rel gaming`, with learning curves drawn out of braille dots, grid policies
  drawn as arrows, and a value map that says which cells the agent has never
  stood in.

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
  clamped into one cell. The last of eight grids could reach 943 of its 6561
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
- **The n of n-step SARSA was measuring the step size.** The agent took four
  steps at a step size of 0.5, and it got stuck more often than any other agent
  on four grids of five. The obvious reading is that four is too many. It is not: at a step size
  of 0.1 four steps beats one step by a wide margin, and one step gets stuck on
  23 of 30 windy grid seeds where four gets stuck on none. The two settings
  trade, because an n step return carries a value n cells further and n times
  the spread. Two other explanations were tested first and neither survived.
- **Six seeds said the entropy default was a clean win.** At 0.05 the first
  sweep read -13 on all six cliff walk seeds: the optimal policy every time.
  Twelve seeds says it loses one seed and that the policies it finds are 0.7
  blunter, because seeds 7 to 12 are nothing like seeds 1 to 6. Nothing went
  wrong in the first measurement. It is what six samples of a noisy thing look
  like. The number of seeds is now part of every claim on that page.
