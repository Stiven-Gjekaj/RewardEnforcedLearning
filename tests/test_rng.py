"""Tests for the seeded source of chance.

Two of these tests hold fixed numbers. They are the reason the rest of the
project can promise that a seed replays a run. If either one fails, every
result recorded in the documentation has quietly changed meaning, and the
failure has to be treated as that rather than as a number to update.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from rel.rng import Rng

# The published output of the reference PCG32 program, for state 42 and stream
# 54. Every implementation of PCG-XSH-RR agrees on these six numbers.
#
# This is the test that says the generator is really PCG32 and not something
# near it. The pinned test below only says that the generator has not changed
# since somebody wrote the numbers down, which a wrong generator also passes.
REFERENCE_42_54 = [
    0xA15C02B7,
    0x7B47F409,
    0xBA1D3330,
    0x83D2F293,
    0xBFA4784B,
    0xCBED606E,
]

# The first eight draws of the stream this project uses by default, on seed
# 12345. These come from the implementation, and they mean something only
# because the test above proves the implementation is PCG32.
PINNED_12345 = [
    2280515124,
    875822104,
    2165132003,
    3444695176,
    1217744654,
    1270201690,
    318034840,
    1693337798,
]


def test_the_generator_agrees_with_the_reference_implementation() -> None:
    source = Rng(42, 54)
    assert [source.next_uint32() for _ in range(6)] == REFERENCE_42_54


def test_the_sequence_for_a_seed_is_pinned() -> None:
    source = Rng(12345)
    assert [source.next_uint32() for _ in range(8)] == PINNED_12345


def test_two_seeds_do_not_give_the_same_run() -> None:
    a = [Rng(1).next_uint32() for _ in range(4)]
    b = [Rng(2).next_uint32() for _ in range(4)]
    assert a != b


def test_restart_replays_the_same_sequence() -> None:
    source = Rng(99)
    first = [source.next_uint32() for _ in range(10)]

    again = source.restart()
    assert [again.next_uint32() for _ in range(10)] == first


class TestBounds:
    def test_below_stays_inside_the_bound(self) -> None:
        source = Rng(3)
        assert all(0 <= source.below(7) < 7 for _ in range(2000))

    def test_below_reaches_every_value(self) -> None:
        source = Rng(4)
        seen = {source.below(5) for _ in range(500)}
        assert seen == {0, 1, 2, 3, 4}

    def test_below_refuses_a_bound_of_zero(self) -> None:
        with pytest.raises(ValueError):
            Rng(1).below(0)

    def test_between_includes_both_ends(self) -> None:
        source = Rng(5)
        seen = {source.between(3, 6) for _ in range(400)}
        assert seen == {3, 4, 5, 6}

    def test_between_refuses_an_upper_below_the_lower(self) -> None:
        with pytest.raises(ValueError):
            Rng(1).between(4, 3)

    def test_below_does_not_favour_the_low_values(self) -> None:
        # A generator that took the remainder with no rejection would give 0 and
        # 1 more often than 2 for this bound, because 2 to the power 32 is not a
        # multiple of 3. The bias is far too small to see at this bound, so the
        # test uses a bound near the top of the range where it is large.
        #
        # With bound 0xC0000000, plain modulo would return a value below
        # 0x40000000 twice as often as it should. This counts that band.
        bound = 0xC000_0000
        band = 0x4000_0000
        source = Rng(6)
        draws = 20_000
        low = sum(1 for _ in range(draws) if source.below(bound) < band)
        expected = draws * band / bound
        assert abs(low - expected) < 0.15 * expected


class TestFloats:
    def test_random_stays_inside_the_unit_range(self) -> None:
        source = Rng(7)
        assert all(0.0 <= source.random() < 1.0 for _ in range(2000))

    def test_uniform_stays_inside_its_range(self) -> None:
        source = Rng(8)
        assert all(-2.0 <= source.uniform(-2.0, 5.0) < 5.0 for _ in range(2000))

    def test_chance_takes_the_ends_without_drawing(self) -> None:
        source = Rng(9)
        before = source.snapshot()
        assert source.chance(0.0) is False
        assert source.chance(1.0) is True
        assert source.snapshot() == before

    def test_chance_is_about_right(self) -> None:
        source = Rng(10)
        draws = 10_000
        hits = sum(1 for _ in range(draws) if source.chance(0.25))
        assert abs(hits - 2500) < 200


class TestNormal:
    def test_the_mean_and_the_deviation_are_about_right(self) -> None:
        source = Rng(11)
        draws = [source.normal(3.0, 2.0) for _ in range(20_000)]
        mean = sum(draws) / len(draws)
        variance = sum((value - mean) ** 2 for value in draws) / len(draws)
        assert abs(mean - 3.0) < 0.05
        assert abs(variance**0.5 - 2.0) < 0.05

    def test_the_second_of_a_pair_is_kept(self) -> None:
        # The polar method makes two numbers at a time. The first call has to
        # leave the second one behind, or half of every pair is thrown away and
        # the generator does twice the work it needs to.
        source = Rng(12)
        assert source.snapshot()[3] is None
        source.normal()
        assert source.snapshot()[3] is not None
        source.normal()
        assert source.snapshot()[3] is None


class TestSnapshots:
    def test_a_snapshot_carries_on_at_the_same_point(self) -> None:
        source = Rng(13)
        for _ in range(5):
            source.next_uint32()

        seed, state, increment, spare = source.snapshot()
        expected = [source.next_uint32() for _ in range(5)]

        resumed = Rng.restore(seed, state, increment, spare)
        assert [resumed.next_uint32() for _ in range(5)] == expected

    def test_a_snapshot_carries_the_unused_half_of_a_normal_pair(self) -> None:
        # This is the fault the field exists to prevent. Take a snapshot
        # between the two halves of a pair and drop the kept value, and the
        # resumed run draws a fresh pair. Every number after the resume then
        # differs, and nothing reports it.
        source = Rng(14)
        source.normal()

        seed, state, increment, spare = source.snapshot()
        assert spare is not None
        expected = [source.normal() for _ in range(4)]

        resumed = Rng.restore(seed, state, increment, spare)
        assert [resumed.normal() for _ in range(4)] == expected

        dropped = Rng.restore(seed, state, increment, None)
        assert [dropped.normal() for _ in range(4)] != expected

    def test_restore_refuses_an_even_increment(self) -> None:
        with pytest.raises(ValueError):
            Rng.restore(1, 2, 4)


class TestStreams:
    def test_a_stream_does_not_depend_on_what_the_parent_drew(self) -> None:
        # This is what lets two agents run against the same environment dice.
        # An agent that explores more draws more, and if that moved the
        # environment stream then a comparison between two exploration settings
        # would also be a comparison between two environments.
        quiet = Rng(21)
        busy = Rng(21)
        for _ in range(1000):
            busy.next_uint32()

        from_quiet = [quiet.stream("env").next_uint32() for _ in range(4)]
        from_busy = [busy.stream("env").next_uint32() for _ in range(4)]
        assert from_quiet == from_busy

    def test_two_names_give_two_sequences(self) -> None:
        source = Rng(22)
        agent = [source.stream("agent").next_uint32() for _ in range(6)]
        env = [source.stream("env").next_uint32() for _ in range(6)]
        assert agent != env

    def test_the_same_name_on_two_seeds_gives_two_sequences(self) -> None:
        a = Rng(23).stream("env").next_uint32()
        b = Rng(24).stream("env").next_uint32()
        assert a != b

    def test_a_stream_name_means_the_same_thing_in_another_process(self) -> None:
        # Python randomises the hash of a string per process. A stream number
        # taken from `hash()` would therefore change on every start, and a run
        # would never reproduce. Nothing inside one process can see that fault,
        # because every value in that process agrees with itself.
        #
        # This test starts a second interpreter with hash randomisation left
        # on, and asks it for the same stream.
        script = 'from rel.rng import Rng;print(Rng(31).stream("env").next_uint32())'
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(result.stdout.strip()) == Rng(31).stream("env").next_uint32()


class TestCollections:
    def test_pick_returns_a_member(self) -> None:
        source = Rng(41)
        items = ["a", "b", "c"]
        assert all(source.pick(items) in items for _ in range(100))

    def test_pick_refuses_an_empty_sequence(self) -> None:
        with pytest.raises(ValueError):
            Rng(1).pick([])

    def test_shuffle_keeps_every_item(self) -> None:
        source = Rng(42)
        items = list(range(20))
        source.shuffle(items)
        assert sorted(items) == list(range(20))
        assert items != list(range(20))

    def test_a_weight_of_zero_is_never_chosen(self) -> None:
        source = Rng(43)
        weights = [1.0, 0.0, 3.0]
        assert all(source.weighted_index(weights) != 1 for _ in range(1000))

    def test_the_weights_decide_how_often_an_index_comes_up(self) -> None:
        source = Rng(44)
        draws = 10_000
        counts = [0, 0]
        for _ in range(draws):
            counts[source.weighted_index([1.0, 3.0])] += 1
        assert abs(counts[1] / draws - 0.75) < 0.02

    def test_weights_do_not_have_to_add_up_to_one(self) -> None:
        source = Rng(45)
        assert source.weighted_index([50.0, 50.0]) in (0, 1)

    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(ValueError):
            Rng(1).weighted_index([1.0, -1.0])

    def test_weights_that_are_all_zero_are_refused(self) -> None:
        with pytest.raises(ValueError):
            Rng(1).weighted_index([0.0, 0.0])

    def test_sample_indices_gives_different_indices(self) -> None:
        source = Rng(46)
        for _ in range(200):
            drawn = source.sample_indices(5, 40)
            assert len(set(drawn)) == 5
            assert all(0 <= index < 40 for index in drawn)

    def test_sample_indices_can_take_the_whole_population(self) -> None:
        source = Rng(47)
        drawn = source.sample_indices(6, 6)
        assert sorted(drawn) == list(range(6))

    def test_sample_indices_refuses_more_than_the_population(self) -> None:
        with pytest.raises(ValueError):
            Rng(1).sample_indices(4, 3)
