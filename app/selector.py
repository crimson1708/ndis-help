"""Pick which documents go to the model for a given question.

BM25 over extracted page text, scored per page and aggregated per document, so
one sharply relevant page beats a long document that mentions the term in
passing. This is keyword matching: it finds "seizure" in seizure docs but will
not connect "temperature" to "fever". See README for the embeddings upgrade.
"""
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

from . import config

K1 = 1.5
B = 0.75
TOP_PAGES_PER_DOC = 3
TITLE_BOOST = 2.5

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it",
    "its", "my", "of", "on", "or", "should", "that", "the", "their", "them",
    "then", "there", "they", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        # Crude plural folding so "seizures" matches "seizure".
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


@dataclass
class Selection:
    document_id: int
    filename: str
    title: str
    page_count: int
    byte_size: int
    score: float
    top_pages: list[int]


@dataclass
class PageSelection:
    """One page, chosen on its own merits rather than its document's."""
    document_id: int
    filename: str
    title: str
    page_no: int
    text: str
    score: float


@dataclass
class _Scored:
    docs: list[sqlite3.Row]
    idf: dict[str, float]
    # (document_id, page_no) -> BM25 score. Only pages that matched at all.
    pages: dict[tuple[int, int], float]
    text: dict[tuple[int, int], str]


def _score(conn: sqlite3.Connection, org_id: int, query_terms: list[str]) -> _Scored:
    """BM25 every page in the org against the query terms.

    Shared by document selection and page selection so the two can never drift
    apart on what counts as relevant.
    """
    docs = conn.execute(
        """SELECT id, filename, title, page_count, byte_size, has_text
           FROM documents WHERE org_id = ?""",
        (org_id,),
    ).fetchall()
    if not docs:
        return _Scored(docs=[], idf={}, pages={}, text={})

    rows = conn.execute(
        """SELECT p.document_id, p.page_no, p.text
           FROM pages p JOIN documents d ON d.id = p.document_id
           WHERE d.org_id = ?""",
        (org_id,),
    ).fetchall()

    page_tokens: dict[tuple[int, int], Counter] = {}
    lengths: dict[tuple[int, int], int] = {}
    text: dict[tuple[int, int], str] = {}
    doc_freq: Counter = Counter()
    for row in rows:
        key = (row["document_id"], row["page_no"])
        counts = Counter(tokenize(row["text"]))
        page_tokens[key] = counts
        lengths[key] = sum(counts.values())
        text[key] = row["text"]
        for term in counts:
            doc_freq[term] += 1

    n_pages = max(1, len(page_tokens))
    avg_len = (sum(lengths.values()) / n_pages) or 1.0

    idf = {}
    for term in set(query_terms):
        n_q = doc_freq.get(term, 0)
        idf[term] = math.log(1 + (n_pages - n_q + 0.5) / (n_q + 0.5))

    pages: dict[tuple[int, int], float] = {}
    for key, counts in page_tokens.items():
        dl = lengths[key] or 1
        score = 0.0
        for term in query_terms:
            f = counts.get(term, 0)
            if not f:
                continue
            score += idf[term] * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avg_len))
        if score > 0:
            pages[key] = score

    return _Scored(docs=docs, idf=idf, pages=pages, text=text)


def select(conn: sqlite3.Connection, org_id: int, question: str,
           max_docs: int | None = None, max_pages: int | None = None,
           max_bytes: int | None = None) -> list[Selection]:
    max_docs = max_docs or config.MAX_DOCS
    max_pages = max_pages or config.MAX_PAGES
    max_bytes = max_bytes or config.MAX_BYTES

    query_terms = tokenize(question)
    scored = _score(conn, org_id, query_terms)
    docs, idf = scored.docs, scored.idf
    if not docs:
        return []

    page_scores: dict[int, list[tuple[float, int]]] = {}
    for (doc_id, page_no), score in scored.pages.items():
        page_scores.setdefault(doc_id, []).append((score, page_no))

    results: list[Selection] = []
    for doc in docs:
        pages = sorted(page_scores.get(doc["id"], []), reverse=True)[:TOP_PAGES_PER_DOC]
        score = sum(s for s, _ in pages)

        title_terms = set(tokenize(doc["title"]))
        score += TITLE_BOOST * sum(idf.get(t, 0) for t in set(query_terms) & title_terms)

        # A scanned PDF has no text to match on. Give it the corpus-average
        # score so it stays eligible instead of being silently unreachable.
        if not doc["has_text"] and not pages:
            score = max(score, 0.01)

        if score > 0:
            results.append(Selection(
                document_id=doc["id"], filename=doc["filename"], title=doc["title"],
                page_count=doc["page_count"], byte_size=doc["byte_size"],
                score=score, top_pages=[p for _, p in pages],
            ))

    results.sort(key=lambda s: s.score, reverse=True)

    chosen: list[Selection] = []
    pages_used = bytes_used = 0
    for sel in results:
        if len(chosen) >= max_docs:
            break
        if pages_used + sel.page_count > max_pages and chosen:
            continue
        if bytes_used + sel.byte_size > max_bytes and chosen:
            continue
        chosen.append(sel)
        pages_used += sel.page_count
        bytes_used += sel.byte_size

    # Deterministic order: prompt caching is a prefix match, so an identical
    # document set must serialise identically to reuse the cache.
    chosen.sort(key=lambda s: s.document_id)
    return chosen


def select_pages(conn: sqlite3.Connection, org_id: int, question: str,
                 max_pages: int | None = None) -> list[PageSelection]:
    """Pick individual pages for a model that reads text, not PDFs.

    A local model pays for every token of context before it writes a word, so
    whole documents are not affordable the way they are on the API. When the
    whole corpus fits in the budget we send all of it and skip retrieval
    entirely - that is the honest thing to do at this size, and it removes BM25
    (which cannot connect "temperature" to "fever") from the failure surface.
    """
    max_pages = max_pages or config.LOCAL_MAX_PAGES
    query_terms = tokenize(question)
    scored = _score(conn, org_id, query_terms)
    if not scored.docs:
        return []

    meta = {d["id"]: d for d in scored.docs}
    total_pages = sum(d["page_count"] for d in scored.docs)

    if total_pages <= max_pages:
        # Corpus fits: send it all, in document order.
        keys = sorted(scored.text.keys())
        ranked = {k: scored.pages.get(k, 0.0) for k in keys}
    else:
        top = sorted(scored.pages.items(), key=lambda kv: kv[1], reverse=True)[:max_pages]
        if not top:
            # Nothing matched. Rather than send nothing, send the opening pages
            # of each document - usually scope and purpose - so the model can
            # at least say what the corpus covers.
            keys = sorted(scored.text.keys())[:max_pages]
            ranked = {k: 0.0 for k in keys}
        else:
            ranked = dict(sorted(top))

    out = []
    for (doc_id, page_no), score in sorted(ranked.items()):
        doc = meta.get(doc_id)
        if not doc:
            continue
        out.append(PageSelection(
            document_id=doc_id,
            filename=doc["filename"],
            title=doc["title"],
            page_no=page_no,
            text=scored.text[(doc_id, page_no)],
            score=score,
        ))
    return out
