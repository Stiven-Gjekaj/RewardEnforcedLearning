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
| Eligibility traces | One dial from one step learning to Monte Carlo, done properly rather than by choosing an `n` | The clearest thing missing. n-step SARSA is the crude version of it and it is in |
| n-step tree backup | An off-policy n-step method with no importance sampling | Wanted, and it needs the off-policy machinery below first |
| Off-policy Monte Carlo | Weighted importance sampling, and the variance lesson that goes with it | The lesson needs a behaviour policy that can be set separately, which no agent here has |
| Prioritised sweeping | Dyna that replays the steps that matter rather than a random one | A queue and a predecessor table. Straightforward, and it would make the maze results far better |
| Replay and a target network | The two pieces that make a value network stable | Needs a network over Q rather than over a policy, and the honest version needs more compute than pure Python has |
| Continuous actions | Half of control, and everything about robotics | The whole action interface here is `Discrete`. This is a large change and it should be a large change |
| Options | Temporally extended actions, on the grid the literature uses for them | Four rooms is in the project waiting for this |

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
| **reinforce** | **49.36s** | 100,000 |

The tabular agents are fast enough that a dependency would buy nothing: a run
that takes a twentieth of a second does not need to take a thousandth.

The network agents are a different matter. REINFORCE is a thousand times slower
than Q-learning on the same environment, and almost all of that is Python
running loops over lists of floats. NumPy would give most of it back.

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

**Should `n-step-sarsa` default to a smaller `n`?** It is the worst agent in
the tables on four of the five grids and gets stuck most often. Four may simply
be too many for grids this small. Nobody has swept it.

**Why does REINFORCE fail on two seeds of six on the cliff walk?** Measured:
seeds 1 and 3 never reach the goal at all, and the other four reach -13 or -17.
The suspicion is that standardising the returns within an episode removes the
signal when every step of an episode is equally bad, which is exactly the case
for an episode that never finishes. That is a guess and it has not been
measured.

**Is the actor critic here wrong, or only untuned?** It finds the optimal
policy on the cliff walk, which says the pieces are put together correctly. It
does not solve the cart pole at any setting that was tried, and the settings
that were tried are listed in `docs/algorithms.md`. An entropy bonus helps and
does not fix it.

**Should the digest cover the agent as well as the environment?** It currently
hashes the transitions, so two agents that happen to walk the same path get the
same digest. That is arguably the right thing, because the digest is about the
run and not about the learner, and arguably not.

---

## Things that would be nice and are not milestones

- A `--out` that writes a run to a file, and a `rel replay` that reads one.
- Sweeping a setting from the command line, with the table falling out.
- An environment described entirely by a text file, so that a new grid needs no
  Python at all.
- A confidence band on the compare chart rather than a mean line.
