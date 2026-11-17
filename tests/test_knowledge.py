from agentdesk.config import ROOT
from agentdesk.knowledge import KnowledgeBase, split_sections, tokenize


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Free shipping, over 499 SEK!") == ["free", "shipp", "over", "499", "sek"]


def test_tokenize_matches_word_forms() -> None:
    assert tokenize("refunds refunded refund") == ["refund", "refund", "refund"]


def test_split_sections_uses_h2_headings_as_sources() -> None:
    sections = split_sections(ROOT / "data" / "kb" / "returns.md")
    assert [s.source for s in sections] == [
        "returns.md#return-window",
        "returns.md#how-to-return",
        "returns.md#refunds",
        "returns.md#items-that-cannot-be-returned",
    ]


def test_search_finds_the_right_section(kb: KnowledgeBase) -> None:
    assert kb.search("how long does a refund take")[0].source == "returns.md#refunds"
    assert kb.search("which payment methods do you accept")[0].source == (
        "payments.md#payment-methods"
    )
    assert kb.search("warranty period for headphones")[0].source == ("warranty.md#warranty-period")


def test_search_returns_nothing_for_unrelated_or_empty_queries(kb: KnowledgeBase) -> None:
    assert kb.search("xyzzy quantum banana") == []
    assert kb.search("!!!") == []


def test_postgres_urls_get_the_async_driver() -> None:
    from agentdesk.config import Settings

    url = Settings(database_url="postgres://u:p@host:5432/db").database_url
    assert url == "postgresql+asyncpg://u:p@host:5432/db"
