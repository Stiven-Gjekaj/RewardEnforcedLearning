"""Tests for the name to entry table.

The entries here are made up. A test that read the real environment table would
fail the day somebody adds an environment, which is a change the test has no
opinion about.
"""

from __future__ import annotations

import pytest

from rel.registry import Entry, Registry


class Thing:
    def __init__(self, source: str, size: int = 3, colour: str = "grey") -> None:
        self.source = source
        self.size = size
        self.colour = colour


def build_thing(source: str, size: int = 3, colour: str = "grey") -> Thing:
    return Thing(source, size, colour)


def a_registry() -> Registry[Thing]:
    return Registry(
        "thing",
        [
            Entry("small", "A small thing.", build_thing, tags=("round",)),
            Entry("large", "A large thing.", build_thing, tags=("round", "heavy")),
        ],
    )


class TestLookup:
    def test_it_finds_an_entry_by_name(self) -> None:
        assert a_registry()["small"].summary == "A small thing."

    def test_it_reports_a_near_miss(self) -> None:
        # A typed name that is one letter out is the common case, and a plain
        # KeyError puts the burden of finding the list back on the reader.
        with pytest.raises(KeyError, match="Did you mean 'small'"):
            a_registry()["smal"]

    def test_it_points_at_the_list_when_nothing_is_close(self) -> None:
        with pytest.raises(KeyError, match=r"--list"):
            a_registry()["nothing like it"]

    def test_two_entries_cannot_share_a_name(self) -> None:
        registry = a_registry()
        with pytest.raises(ValueError, match="two things named"):
            registry.add(Entry("small", "Another one.", build_thing))

    def test_it_knows_what_it_holds(self) -> None:
        registry = a_registry()
        assert registry.names() == ["small", "large"]
        assert len(registry) == 2
        assert "small" in registry
        assert "missing" not in registry
        assert [entry.name for entry in registry] == ["small", "large"]


class TestTags:
    def test_it_finds_the_entries_carrying_a_tag(self) -> None:
        assert [entry.name for entry in a_registry().tagged("heavy")] == ["large"]

    def test_a_tag_nothing_carries_gives_nothing(self) -> None:
        assert a_registry().tagged("cold") == []

    def test_it_lists_the_tags_in_use(self) -> None:
        assert a_registry().tags() == ["heavy", "round"]


class TestBuilding:
    def test_it_builds_with_the_defaults(self) -> None:
        thing = a_registry().make("small", "seed")
        assert (thing.source, thing.size, thing.colour) == ("seed", 3, "grey")

    def test_a_setting_reaches_the_builder(self) -> None:
        thing = a_registry().make("small", "seed", size=9)
        assert thing.size == 9

    def test_a_setting_that_does_not_exist_is_named_in_the_error(self) -> None:
        # The alternative is a TypeError out of the builder, which names the
        # function rather than the environment and does not say what the
        # environment does accept.
        with pytest.raises(TypeError, match="no setting named 'weight'"):
            a_registry().make("small", "seed", weight=2)

    def test_the_error_lists_what_is_accepted(self) -> None:
        with pytest.raises(TypeError, match="It accepts: colour, size"):
            a_registry().make("small", "seed", weight=2)


class TestOptions:
    def test_it_reads_the_settings_off_the_builder(self) -> None:
        # Reading the signature means an environment that gains a setting gains
        # a command line option with no second place to edit.
        assert a_registry()["small"].options() == {"size": 3, "colour": "grey"}

    def test_the_source_of_chance_is_not_a_setting(self) -> None:
        assert "source" not in a_registry()["small"].options()

    def test_a_setting_with_no_default_reads_as_none(self) -> None:
        def needs_one(source: str, count: int) -> Thing:
            return Thing(source, count)

        entry = Entry("odd", "Needs a count.", needs_one)
        assert entry.options() == {"count": None}
