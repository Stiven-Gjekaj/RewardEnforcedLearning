# Writing a grid without writing Python

Every grid in this project is ASCII already. `--env-file` reads one out of a
text file, so a new environment is a file rather than a change to the package.

```console
$ rel solve --env-file grids/cliff.txt
$ rel train q-learning --env-file grids/rooms.txt
$ rel compare sarsa q-learning --env-file grids/windy.txt
```

The five files in [`grids/`](../grids) are the built in grids, written out.
They are the ones to copy.

---

## The shape of a file

```
# The cliff walk of Sutton and Barto, example 6.6.
name: cliff
summary: Four by twelve. The short path runs along a cliff that pays -100.
max_episode_steps: 500

............
............
............
SXXXXXXXXXXG
```

Everything before the first blank line is the header. Everything after it is
the layout. A file with no blank line in it is all layout and takes every
default.

A header line is `name: value`, or a comment starting with `#`. The two halves
are separated by a blank line rather than told apart line by line, because `#`
is also the wall tile: a layout starting with a row of walls would otherwise be
read as comments.

## The tiles

| Tile | Means |
| :---: | --- |
| `S` | a start cell. A grid needs at least one. |
| `G` | a goal cell. A grid needs at least one. |
| `#` | a wall. Walking into it is a step that goes nowhere. |
| `X` | a pit. |
| `.` | floor. |
| a space | floor as well, which is what makes a room readable. |

**A layout line is taken exactly as written.** A space is a floor tile, so a
line padded out to the width of the others would open a wall that an editor had
trimmed. A file whose rows are not all the same width is refused instead, and
that is the error a trimmed line gives. None of the shipped files needs a
trailing space, and a test says so.

## The settings

| Setting | Means | Default |
| --- | --- | --- |
| `name` | what the reports call it | `grid` |
| `summary` | one line, shown by `rel list` | `An agent on a grid.` |
| `step_reward` | what an ordinary step pays | `-1` |
| `goal_reward` | what reaching a goal pays, or `none` for the step reward | `none` |
| `pit_reward` | what entering a pit pays | `-100` |
| `pit_ends_episode` | whether a pit ends the episode rather than sending the agent back to the start | `false` |
| `slip` | how often a move goes at right angles instead | `0` |
| `wind` | one strength per column, blowing upward | all zero |
| `wind_varies` | whether the wind gusts by one in either direction | `false` |
| `king_moves` | eight directions rather than four | `false` |
| `can_stay` | add an action that does nothing | `false` |
| `max_episode_steps` | the step limit, or `none` for no limit | `none` |
| `solved_return` | the return that counts as solved | `none` |
| `suggested_discount` | what a run uses unless `--discount` says otherwise | `1` |

Anything else is refused, and the message names the line:

```console
$ rel solve --env-file mine.txt
mine.txt, line 3: 'gravity' is not a grid setting. Use one of can_stay,
goal_reward, king_moves, max_episode_steps, name, pit_ends_episode,
pit_reward, slip, solved_return, step_reward, suggested_discount, summary,
wind, wind_varies.
```

The values are typed rather than guessed from the text. `step_reward` is a
number, `king_moves` is either true or false, and `wind` is one number for each
column. A reader that guessed would take `name: 12` as an integer and hand the
environment a name it cannot print.

`tests/test_gridfile.py` checks the table above against the arguments
`GridWorld` really takes, in both directions, so a setting cannot be added to
one and forgotten in the other.

## Changing one number without editing the file

`--env-set` applies on top of whatever the file said:

```console
$ rel solve --env-file grids/cliff.txt --env-set step_reward=-2
$ rel train q-learning --env-file grids/lake.txt --env-set slip=0
```

That is what lets a reader start from somebody else's grid and ask one question
about it.

## What a file cannot do

A file describes a grid. The three environments this project is named for are
not grids: a boat race that can be farmed, a room with something breakable in
it and a thermostat with a dial that makes the sensor lie all keep an audit of
what was really wanted, and none of that is a layout.
[docs/specification-gaming.md](specification-gaming.md) has those.

The two control problems are not grids either. A cart pole and a mountain car
are physics with real numbers in them, and there is nothing to draw.
