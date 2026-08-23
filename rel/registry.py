"""One table that names everything the command line can build.

Each environment and each agent is declared once, in `rel/envs/__init__.py` and
`rel/agents/__init__.py`. From that one entry come the name the command line
accepts, the line in `--list`, the group a benchmark runs, and the table in the
documentation.

The alternative is a decorator on each class. That reads well at the class and
badly everywhere else: nothing then holds the whole list, so the list has to be
found by importing every module and hoping none was missed. An explicit table
is one file a reader can open and see all of.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Entry(Generic[T]):
    """One declared thing.

    `summary` is one sentence in the present tense. It is printed by `--list`
    and read straight into the documentation, so it is written for somebody who
    has not seen the code.
    """

    name: str
    summary: str
    build: Callable[..., T]
    tags: tuple[str, ...] = field(default=())

    def options(self, fixed: int = 1) -> dict[str, Any]:
        """The named settings that `build` accepts, with their defaults.

        The first `fixed` parameters are supplied by the caller rather than by
        a setting. For an environment that is the source of chance. For an
        agent it is the source of chance and the environment, which the builder
        reads two things off and the agent never sees.
        """
        parameters = list(inspect.signature(self.build).parameters.values())
        return {
            parameter.name: (
                None
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            for parameter in parameters[fixed:]
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }


class Registry(Generic[T]):
    """A name to entry table, with an error message that helps."""

    __slots__ = ("_entries", "fixed", "kind")

    def __init__(
        self, kind: str, entries: Sequence[Entry[T]] = (), fixed: int = 1
    ) -> None:
        self.kind = kind
        self.fixed = fixed
        self._entries: dict[str, Entry[T]] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: Entry[T]) -> None:
        if entry.name in self._entries:
            raise ValueError(f"There are two {self.kind}s named {entry.name!r}.")
        self._entries[entry.name] = entry

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[Entry[T]]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, name: str) -> Entry[T]:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(self._unknown(name)) from None

    def names(self) -> list[str]:
        return list(self._entries)

    def tagged(self, tag: str) -> list[Entry[T]]:
        """Every entry carrying this tag, in the order they were declared."""
        return [entry for entry in self._entries.values() if tag in entry.tags]

    def tags(self) -> list[str]:
        """Every tag in use, sorted."""
        return sorted({tag for entry in self._entries.values() for tag in entry.tags})

    def make(self, name: str, *args: Any, **options: Any) -> T:
        """Build one, and say something useful when the settings are wrong."""
        entry = self[name]
        accepted = entry.options(self.fixed)

        unknown = [key for key in options if key not in accepted]
        if unknown:
            offered = ", ".join(sorted(accepted)) or "none"
            raise TypeError(
                f"The {self.kind} {name!r} has no setting named "
                f"{unknown[0]!r}. It accepts: {offered}."
            )

        return entry.build(*args, **options)

    def _unknown(self, name: str) -> str:
        near = difflib.get_close_matches(name, self._entries, n=3, cutoff=0.5)
        message = f"There is no {self.kind} named {name!r}."
        if near:
            return f"{message} Did you mean {', '.join(repr(one) for one in near)}?"
        return f"{message} Run --list to see them all."


__all__ = ["Entry", "Registry"]
