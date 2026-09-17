"""Ask a question, log the result.

The model call itself lives in `backends`; this module owns the part that is
the same whichever model answered - dispatch and the audit trail. Everything a
support worker was told is recorded, with the sources behind it, because
"what did this tool tell someone in March" is a question an employer will
eventually need answered.
"""
import sqlite3

from . import backends, config
from .backends import BackendError  # re-exported: callers catch it from here
from .results import Answer, Citation, Segment  # noqa: F401  (public surface)

__all__ = ["ask", "log", "Answer", "Citation", "Segment", "BackendError"]


def ask(conn: sqlite3.Connection, org_id: int, question: str) -> Answer:
    return backends.get().answer(conn, org_id, question)


def log(conn: sqlite3.Connection, org_id: int, question: str, answer: Answer) -> int:
    cur = conn.execute(
        """INSERT INTO queries
           (org_id, question, answer, grounded, model, selected_doc_ids,
            input_tokens, output_tokens, cache_read_tokens, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            org_id, question, answer.text, int(answer.grounded),
            answer.model or config.MODEL,
            # One document may contribute several pages on the local backend;
            # the audit trail wants the set of documents, not a repeat per page.
            ",".join(sorted({str(s["document_id"]) for s in answer.sources})),
            answer.usage.get("input_tokens"), answer.usage.get("output_tokens"),
            answer.usage.get("cache_read_input_tokens"), answer.latency_ms,
        ),
    )
    query_id = cur.lastrowid
    conn.executemany(
        """INSERT INTO citations
           (query_id, document_id, page_start, page_end, cited_text, kind)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [(query_id, c.document_id, c.page_start, c.page_end, c.cited_text, c.kind)
         for c in answer.citations],
    )
    conn.commit()
    return query_id
