# AGENTS.md

These are the rules for an agent that does work in this repository.
The rules apply to all changes.
They also apply to a change that you start without a request.

## Who writes a commit

- A human is the writer of each commit. An agent is not.
- Set `git config user.name` and `git config user.email` one time.
  Then do not change them.
- Do not add your name to a commit message, a pull request, or a review
  comment.
- Do not add a `Co-Authored-By` line.
- Do not add a link to your session.
- Do not add a footer that says that a tool made the text.
- Make these changes in the configuration of the tool.
  Do not remove the text manually each time.
- Reason: a commit shows that a human read the code.
  That human answers questions about the code six months later.
  An agent cannot do this.

## How to write

- Write all text for this repository in ASD-STE100 Simplified Technical
  English.
  This includes the source code, the comments, the documentation, the commit
  messages, the examples, and the pull requests.
- Use short sentences.
  Use the active voice.
  Use the present tense.
  Write one instruction in one sentence.
- Do not use an em-dash.
- Do not use an emoji.

## Commits

- Each commit has one change only.
- Split a feature into many commits.
  One commit does one step of the feature.
  Do not put a full feature into one commit.
- Reason: a reader examines one step at a time.
  A reader also removes one step and keeps the other steps.
  A commit that holds a full feature stops both of these actions.
  Six months later, a reader finds the one step that caused a fault.
  That reader cannot find it in a commit that changed twenty files.
- Put the code and its tests in the same commit.
- Put the documentation in a different commit.
- If you change a name in many files, put that change alone in one commit.
  Do not change what the code does in that commit.
- Write the subject line in the present tense.
  The subject line tells what the change does.
- Do not put a version number in the subject line.
- A commit does not change the version.
- Do not open a pull request if the human does not ask for it.

### An example of granular commits

These commits are correct.
Each commit does one step.
The steps are in the same feature group.

    feat: add the tile coder
    feat: add the semi-gradient control agent
    feat: connect the tile coder to the command line

This commit is not correct.
It holds the full feature.

    feat: integrate function approximation

## How to work

- Run the code.
  Do not only say that the code will operate correctly.
- Tell the human when the results do not agree with your statement.
- Read the open issues before you add an environment or an agent.

## What to measure

- Measure the thing that you tell the human.
  Do not measure something near it.
- Give the number that a run in this repository produces.
  Do not give a number that you remember from a paper.
- Say the seed, the episode count and the command with every number.
  A number with no seed is not a result. It is an anecdote.
- Reason: a size is not a state.
  A watcher looked at the size of a directory and said that a download was
  complete.
  One gigabyte was still to come, because a file that nothing used was in the
  same directory.
- Reason: an average over ten seeds is not the average over one seed.
  A learning curve from one seed can show a jump that no other seed shows.
  Say how many seeds are behind a curve.

## What learning code makes easy to get wrong

- An agent that learns nothing still returns a number.
  A run that ends with no error is not a run that worked.
  Compare against the random policy before you say that an agent learned.
- A reward is not a score.
  A run reports the reward that the environment paid.
  Ask the environment for the true objective as well, and report both.
- A seed that is not passed down is a seed that does nothing.
  The environment and the agent both draw chance.
  Both take their source from the same seed, and a test holds that promise.

## What a test can hold on to

- A test asks the code a question.
  Do not let it ask the example configuration a question.
- Build the state that a test needs inside the test.
  Do not read it out of a file that the author edits.
- A test that covers learning states the bound that it needs and no more.
  A test that pins the exact return of episode 400 fails when a tuning change
  moves it by one step, and that failure teaches a reader to ignore the suite.
- Break the thing that a new test covers on purpose, and watch the test fail.
  A test that cannot fail is worse than no test.
  It makes a part look guarded while the part drifts.

## What to keep

- Look in a directory before you delete it.
  The size of a directory is not the contents of it.
- Put each thing that the human chooses into the repository.
  Do not keep it only in a working directory.
