"""Tests for a run written down step by step and read back.

The digest at the top of a file is a claim the file makes about its own
contents. Most of what is here is about that claim being checked rather than
trusted, because a recording that could quietly disagree with its own digest
would be worse than no recording: it would look like evidence.
"""

from __future__ import annotations

import gzip

import pytest

from rel.agents.base import Transition
from rel.agents.td import QLearning
from rel.envs.classic import cliff_walk
from rel.recording import (
    FORMAT,
    Recorder,
    RecordingError,
    parse,
    read,
    save,
    watched,
    write,
)
from rel.rng import Rng
from rel.training import train

WALK = (
    Transition(36, 0, -1.0, 24, terminated=False, truncated=False),
    Transition(24, 1, -1.0, 25, terminated=False, truncated=False),
    Transition(25, 2, -100.0, 36, terminated=False, truncated=False),
    Transition(36, 1, -1.0, 37, terminated=True, truncated=False),
)


def a_file() -> str:
    return write(watched(WALK), env="cliff", agent="q-learning", seed=7, discount=1)


class TestWhatIsWritten:
    def test_it_starts_with_the_format(self) -> None:
        assert a_file().splitlines()[0] == FORMAT

    def test_it_holds_one_line_for_every_step(self) -> None:
        body = a_file().split("\n\n", 1)[1].strip().splitlines()
        assert len(body) == len(WALK)

    def test_a_step_line_starts_with_what_the_digest_hashes(self) -> None:
        # The first four fields are exactly the digest's view of the step, in
        # the order it hashes them, so checking a file is re-hashing a prefix
        # of each line rather than spelling a transition a second way.
        body = a_file().split("\n\n", 1)[1].strip().splitlines()
        assert body[0] == "36|0|-1|00|24"

    def test_the_state_it_landed_in_comes_last(self) -> None:
        body = a_file().split("\n\n", 1)[1].strip().splitlines()
        assert body[2].split("|")[-1] == "36"

    def test_the_step_count_and_digest_come_from_the_recorder(self) -> None:
        # Not from the caller. They are what the file is checked against, and
        # a caller that could set them could set them wrongly.
        recorder = watched(WALK)
        text = write(recorder, env="cliff", steps=999, digest="deadbeef")
        assert f"steps: {len(WALK)}" in text
        assert f"digest: {recorder.hexdigest()}" in text

    def test_a_header_value_with_a_line_break_is_refused(self) -> None:
        with pytest.raises(RecordingError, match="line break"):
            write(watched(WALK), env="cliff\nagent: sneaky")


class TestReadingItBack:
    def test_the_header_comes_back(self) -> None:
        got = parse(a_file())
        assert got.header["env"] == "cliff"
        assert got.header["agent"] == "q-learning"
        assert got.header["seed"] == "7"

    def test_every_step_comes_back(self) -> None:
        got = parse(a_file())
        assert len(got.steps) == len(WALK)
        assert got.steps[0].action == 0
        assert got.steps[2].reward == -100.0
        assert got.steps[3].terminated

    def test_the_observations_come_back_as_text(self) -> None:
        # Turning "0.5,-1.25" back into a pair of floats would need to know
        # which environment wrote it, and a recording that could only be read
        # with the environment to hand would not be worth writing.
        got = parse(a_file())
        assert got.steps[0].observation == "36"
        assert got.steps[0].next_observation == "24"

    def test_a_file_that_does_not_start_with_the_format_is_refused(self) -> None:
        with pytest.raises(RecordingError, match="does not start with"):
            parse("some other file\n\n1|0|0|00|1\n")

    def test_a_file_with_no_steps_is_refused(self) -> None:
        with pytest.raises(RecordingError, match="no steps under it"):
            parse(f"{FORMAT}\nenv: cliff\n")

    def test_a_file_with_no_digest_in_it_is_refused(self) -> None:
        # The two names a file must carry are the two the check needs.
        # Refusing one for having no environment name would be refusing a
        # recording of something that has no name yet.
        with pytest.raises(RecordingError, match="no digest"):
            parse(f"{FORMAT}\nenv: cliff\nsteps: 1\n\n1|0|0|00|1\n")

    def test_a_file_with_no_environment_name_still_reads(self) -> None:
        text = write(watched(WALK))
        assert len(parse(text).steps) == len(WALK)

    def test_a_step_with_the_wrong_number_of_fields_says_which_line(self) -> None:
        text = a_file().replace("36|0|-1|00|24", "36|0|-1|00")
        with pytest.raises(RecordingError, match="line 9: a step has five fields"):
            parse(text, check=False)

    def test_endings_that_are_not_two_digits_are_refused(self) -> None:
        text = a_file().replace("36|0|-1|00|24", "36|0|-1|maybe|24")
        with pytest.raises(RecordingError, match="two digits"):
            parse(text, check=False)

    def test_a_header_line_that_is_not_a_pair_is_refused(self) -> None:
        with pytest.raises(RecordingError, match="is not"):
            parse(f"{FORMAT}\njust words\n\n1|0|0|00|1\n")


class TestTheDigestIsChecked:
    def test_a_file_that_matches_its_digest_reads(self) -> None:
        assert len(parse(a_file()).steps) == len(WALK)

    def test_the_digest_is_the_one_the_run_made(self) -> None:
        recorder = watched(WALK)
        assert parse(write(recorder, env="cliff")).digest() == recorder.hexdigest()

    def test_a_changed_step_is_refused(self) -> None:
        text = a_file().replace("25|2|-100|00|36", "25|2|-1|00|36")
        with pytest.raises(RecordingError, match="changed since it was written"):
            parse(text)

    def test_a_reward_written_a_second_way_is_still_refused(self) -> None:
        # The digest is worked out from the lines rather than from the numbers
        # read out of them. Writing a parsed reward out again would make -1
        # and -1.00 the same file.
        text = a_file().replace("36|0|-1|00|24", "36|0|-1.00|00|24")
        with pytest.raises(RecordingError, match="changed since it was written"):
            parse(text)

    def test_a_dropped_step_is_refused(self) -> None:
        text = a_file().replace("24|1|-1|00|25\n", "")
        with pytest.raises(RecordingError, match="holds 4 steps and holds 3"):
            parse(text)

    def test_the_check_can_be_switched_off(self) -> None:
        # Only so that a test can read a file it has broken on purpose.
        text = a_file().replace("25|2|-100|00|36", "25|2|-1|00|36")
        assert len(parse(text, check=False).steps) == len(WALK)


class TestWhatCanBeWorkedOutAgain:
    def test_the_returns_and_lengths_come_back(self) -> None:
        rebuilt = parse(a_file()).record()
        assert rebuilt.returns == [-103.0]
        assert rebuilt.lengths == [4]
        assert rebuilt.terminated == [True]

    def test_the_discount_in_the_header_is_used(self) -> None:
        text = write(watched(WALK), env="cliff", discount=0.5)
        rebuilt = parse(text).record()
        # -1 + 0.5 * -1 + 0.25 * -100 + 0.125 * -1
        assert rebuilt.discounted == [pytest.approx(-26.625)]

    def test_a_run_read_back_matches_the_run_written(self) -> None:
        # The whole point. A real run, recorded, read back and compared on
        # everything a chart of it is drawn from.
        rng = Rng(7)
        env = cliff_walk(rng.stream("env"))
        agent: QLearning[int] = QLearning(
            rng.stream("agent"), env.action_space, step_size=0.5
        )
        recorder = Recorder()
        original = train(env, agent, 30, digest=recorder)

        rebuilt = parse(write(recorder, env="cliff", seed=7, discount=1)).record()
        assert rebuilt.returns == original.returns
        assert rebuilt.lengths == original.lengths
        assert rebuilt.discounted == original.discounted
        assert rebuilt.terminated == original.terminated
        assert rebuilt.steps == original.steps


class TestFiles:
    def test_it_writes_and_reads_a_plain_file(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = save(tmp_path / "run.txt", watched(WALK), env="cliff")
        assert len(read(path).steps) == len(WALK)

    def test_a_name_ending_in_gz_is_compressed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        path = save(tmp_path / "run.txt.gz", watched(WALK), env="cliff")
        assert path.read_bytes()[:2] == b"\x1f\x8b"

    def test_the_compressed_file_reads_back_the_same(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        plain = read(save(tmp_path / "run.txt", watched(WALK), env="cliff"))
        packed = read(save(tmp_path / "run.txt.gz", watched(WALK), env="cliff"))
        assert packed.steps == plain.steps
        assert packed.header == plain.header

    def test_compression_is_decided_by_the_bytes_and_not_the_name(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        # A file renamed by a browser still reads.
        path = tmp_path / "renamed.txt"
        path.write_bytes(gzip.compress(a_file().encode("utf-8")))
        assert len(read(path).steps) == len(WALK)

    def test_a_file_that_is_not_there_says_so(self) -> None:
        with pytest.raises(RecordingError, match="cannot be read"):
            read("nowhere/at/all.txt")

    def test_something_that_is_not_gzip_but_looks_like_it_says_so(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        path = tmp_path / "broken.gz"
        path.write_bytes(b"\x1f\x8b" + b"not really")
        with pytest.raises(RecordingError, match="not readable gzip"):
            read(path)
