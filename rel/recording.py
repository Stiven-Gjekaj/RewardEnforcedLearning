"""A run, written down step by step, and read back later.

Every run already hashes its own transitions and prints the result, and two
runs with the same digest took the same path through the environment. That
makes two runs comparable. It does not make one run checkable: a digest alone
cannot say what happened, only whether it happened twice.

A recording is the steps themselves. The digest at the top of the file is then
a claim the file makes about its own contents, and reading it checks that claim
rather than trusting it.

## The shape of a file

    rel-run 1
    env: cliff
    agent: q-learning
    seed: 7
    episodes: 20
    discount: 1
    steps: 412
    digest: 0f9c2b7a1d4e6580

    36|0|-1|00|24
    24|1|-1|00|25

A header of `name: value` lines, a blank line, then one line for each step.

## Why the next state comes last

The first four fields of a step line are exactly what the digest hashes, in the
order it hashes them. So verifying a file is re-hashing a prefix of each line,
and no part of this module has its own idea of how a transition is spelled. The
state the step landed in is what the digest does not cover, so it goes on the
end where it changes nothing.

## What a recording cannot give back

The audit of an environment is not in it. That is a set of numbers an
environment keeps about what was really wanted, it is different for each one,
and a file of steps is not the place to invent a format for it. A replay
reports the returns, the lengths and the endings, which is what the chart of a
run is drawn from.

The observations come back as the text they were written as. Nothing here turns
"0.5,-1.25" back into a pair of floats, because doing so would need to know
which environment wrote it, and a recording that could only be read with the
environment to hand would not be worth writing.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rel.agents.base import Transition
from rel.training import Digest, Record, digest_line, encoded

#: The first line of every file. A reader that met a later format would
#: otherwise take its guesses about the fields for facts.
FORMAT = "rel-run 1"

#: Header names a file must carry to be read at all.
#:
#: These two are what checking the file needs. Everything else a writer puts
#: in the header is worth having and none of it is worth refusing a file over:
#: a reader that demanded the name of the environment could not read a
#: recording of something that has no name yet.
REQUIRED = ("steps", "digest")


class RecordingError(ValueError):
    """A recording that cannot be read, or does not match its own digest."""


class Recorder(Digest):
    """A digest that also keeps every line it hashed.

    A run is watched by one of these in place of its digest, so recording
    costs the run nothing but the memory of what it did.
    """

    __slots__ = ("_kept",)

    def __init__(self) -> None:
        super().__init__()
        self._kept: list[str] = []

    def add(self, transition: Transition[Any]) -> None:
        super().add(transition)
        self._kept.append(
            digest_line(transition).rstrip("\n")
            + "|"
            + encoded(transition.next_observation)
        )

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self._kept)


@dataclass(frozen=True, slots=True)
class RecordedStep:
    """One step of a recording. The observations are the text they were written as."""

    observation: str
    action: int
    reward: float
    terminated: bool
    truncated: bool
    next_observation: str

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class Recording:
    """What a file said, and what can be worked out from it.

    `lines` is what the file really holds and `steps` is that read into
    numbers. The digest is worked out from the lines rather than from the
    numbers: writing a parsed reward out again would be a second spelling of
    it, and a file whose reward had been changed from 1.0 to 1.00 would then
    pass a check it should fail.
    """

    header: Mapping[str, str] = field(default_factory=dict)
    steps: tuple[RecordedStep, ...] = ()
    lines: tuple[str, ...] = ()

    @property
    def discount(self) -> float:
        return float(self.header.get("discount", 1.0))

    def digest(self) -> str:
        """The digest of the steps in this file, worked out again."""
        running = Digest()
        for line in self.lines:
            running.add_line(line.rsplit("|", 1)[0] + "\n")
        return running.hexdigest()

    def record(self) -> Record:
        """The per episode numbers, worked out from the steps.

        The audit is not in a recording, so this is a `Record` with none.
        """
        rebuilt = Record()
        total = 0.0
        discounted = 0.0
        weight = 1.0
        length = 0

        for step in self.steps:
            total += step.reward
            discounted += weight * step.reward
            weight *= self.discount
            length += 1

            if step.done:
                rebuilt.returns.append(total)
                rebuilt.lengths.append(length)
                rebuilt.discounted.append(discounted)
                rebuilt.terminated.append(step.terminated)
                total = discounted = 0.0
                weight = 1.0
                length = 0

        return rebuilt


def write(recorder: Recorder, **header: Any) -> str:
    """The text of a file holding this run.

    The digest and the number of steps are written from the recorder rather
    than taken from the caller, because they are what the file is checked
    against and a caller that could set them could set them wrongly.
    """
    lines = [FORMAT]
    for name, value in header.items():
        text = str(value)
        if "\n" in text:
            raise RecordingError(f"The header value for {name!r} has a line break.")
        lines.append(f"{name}: {text}")
    lines.append(f"steps: {recorder.steps}")
    lines.append(f"digest: {recorder.hexdigest()}")
    lines.append("")
    lines.extend(recorder.lines)
    return "\n".join(lines) + "\n"


def _step(line: str, number: int, where: str) -> RecordedStep:
    parts = line.split("|")
    if len(parts) != 5:
        raise RecordingError(
            f"{where}, line {number}: a step has five fields separated by '|'. "
            f"This one has {len(parts)}."
        )

    observation, action, reward, flags, landed = parts
    if len(flags) != 2 or set(flags) - {"0", "1"}:
        raise RecordingError(
            f"{where}, line {number}: the endings are two digits, each 0 or 1. "
            f"{flags!r} is not."
        )

    try:
        return RecordedStep(
            observation=observation,
            action=int(action),
            reward=float(reward),
            terminated=flags[0] == "1",
            truncated=flags[1] == "1",
            next_observation=landed,
        )
    except ValueError as error:
        raise RecordingError(f"{where}, line {number}: {error}") from error


def parse(text: str, where: str = "<text>", *, check: bool = True) -> Recording:
    """Read a run out of the text of a file.

    `check` is what makes the digest in the header mean anything. It is an
    argument only so that a test can read a file it has broken on purpose.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FORMAT:
        raise RecordingError(f"{where} does not start with {FORMAT!r}.")

    header: dict[str, str] = {}
    number = 1
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            break
        name, colon, value = line.partition(":")
        if not colon:
            raise RecordingError(
                f"{where}, line {number}: a header line is 'name: value'. "
                f"{line.strip()!r} is not."
            )
        header[name.strip()] = value.strip()
    else:
        raise RecordingError(f"{where} has a header and no steps under it.")

    missing = [name for name in REQUIRED if name not in header]
    if missing:
        raise RecordingError(
            f"{where} has no {' and no '.join(missing)} in its header."
        )

    kept = tuple(
        (count, line)
        for count, line in enumerate(lines[number:], start=number + 1)
        if line.strip()
    )
    steps = tuple(_step(line, count, where) for count, line in kept)

    found = Recording(header=header, steps=steps, lines=tuple(line for _, line in kept))
    if check:
        _verify(found, where)
    return found


def _verify(found: Recording, where: str) -> None:
    claimed = found.header["digest"]
    counted = found.header["steps"]

    if counted != str(len(found.steps)):
        raise RecordingError(
            f"{where} says it holds {counted} steps and holds {len(found.steps)}."
        )
    if found.digest() != claimed:
        raise RecordingError(
            f"{where} says its digest is {claimed} and its steps hash to "
            f"{found.digest()}. The file has been changed since it was written."
        )


def save_run(path: str | Path, recorder: Recorder, **header: Any) -> Path:
    """Write a run to a file, compressed when the name ends in `.gz`."""
    found = Path(path)
    text = write(recorder, **header)
    if found.suffix == ".gz":
        found.write_bytes(gzip.compress(text.encode("utf-8")))
    else:
        found.write_text(text, encoding="utf-8")
    return found


def read_run(path: str | Path, *, check: bool = True) -> Recording:
    """Read a run from a file, compressed or not.

    Which one is decided by looking at the first two bytes rather than at the
    name, so a file renamed by a browser still reads.
    """
    found = Path(path)
    try:
        raw = found.read_bytes()
    except OSError as error:
        raise RecordingError(f"{found} cannot be read: {error}") from error

    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError as error:
            raise RecordingError(f"{found} is not readable gzip: {error}") from error

    return parse(raw.decode("utf-8"), where=str(found), check=check)


def watched(steps: Iterable[Transition[Any]]) -> Recorder:
    """A recorder that has already seen these steps. For tests and for replay."""
    recorder = Recorder()
    for step in steps:
        recorder.add(step)
    return recorder


__all__ = [
    "FORMAT",
    "REQUIRED",
    "RecordedStep",
    "Recorder",
    "Recording",
    "RecordingError",
    "parse",
    "read_run",
    "save_run",
    "watched",
    "write",
]
