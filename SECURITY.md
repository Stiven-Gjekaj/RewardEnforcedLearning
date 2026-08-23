# Security

RewardEnforcedLearning is a command line program that trains agents on its own
environments. It opens no ports, makes no network connection, and imports
nothing outside the Python standard library.

It writes files only when you ask it to, under the path you give to
`--out`, and it reads a file only when you name one.

If you find something that behaves otherwise, please report it privately
through GitHub's
[security advisories](https://github.com/Stiven-Gjekaj/RewardEnforcedLearning/security/advisories/new)
rather than in a public issue.

Two things are worth saying plainly, because they look like security
properties and are not:

- **A run digest is a check against drift, not against tampering.** It is a
  BLAKE2b hash of the transitions of a run. It tells you that two runs of the
  same build took the same path. It does not tell you that a build is
  unmodified.
- **A saved run holds no code.** It is a text file of numbers, and the reader
  parses it with the standard library rather than evaluating it. A file from a
  stranger cannot run anything when this program reads it.

Only the most recent release is supported.
