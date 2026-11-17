"""Keyword search (BM25) over the markdown help-centre articles in data/kb."""

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")


_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(token: str) -> str:
    """A tiny stemmer so 'refunds', 'refunded' and 'refund' match each other."""
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower())]


@dataclass(frozen=True)
class Section:
    source: str  # e.g. "returns.md#refunds"
    title: str
    text: str


def split_sections(path: Path) -> list[Section]:
    """Split a markdown file into one Section per '## ' heading."""
    sections: list[Section] = []
    title: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if title and lines:
            slug = "-".join(_TOKEN.findall(title.lower()))
            sections.append(Section(f"{path.name}#{slug}", title, " ".join(lines).strip()))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            title, lines = line[3:].strip(), []
        elif title and line.strip():
            lines.append(line.strip())
    flush()
    return sections


class KnowledgeBase:
    def __init__(self, sections: list[Section]) -> None:
        if not sections:
            raise ValueError("knowledge base is empty")
        self.sections = sections
        self._bm25 = BM25Okapi([tokenize(f"{s.title} {s.text}") for s in sections])

    @classmethod
    def from_dir(cls, directory: Path) -> "KnowledgeBase":
        sections = [s for p in sorted(directory.glob("*.md")) for s in split_sections(p)]
        return cls(sections)

    @property
    def sources(self) -> set[str]:
        return {s.source for s in self.sections}

    def search(self, query: str, k: int = 3) -> list[Section]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.sections[i] for i in ranked[:k] if scores[i] > 0]
