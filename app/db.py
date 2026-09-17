"""SQLite storage.

Documents belong to an org from the outset. This instance serves a single
tenant today (config.DEFAULT_ORG), but the column is here so adding real
tenants later is a routing change rather than a migration.
"""
import sqlite3
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
    id          INTEGER PRIMARY KEY,
    slug        TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    org_id      INTEGER NOT NULL REFERENCES orgs(id),
    filename    TEXT NOT NULL,
    title       TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    page_count  INTEGER NOT NULL,
    byte_size   INTEGER NOT NULL,
    -- Whether this document may be surfaced on a public deployment. Internal
    -- employer policy material defaults to 0; only set 1 with clear licence.
    publishable INTEGER NOT NULL DEFAULT 0,
    -- 0 when pypdf found no extractable text: a scanned PDF. Claude can still
    -- read it (it renders pages), but keyword selection cannot see it.
    has_text    INTEGER NOT NULL DEFAULT 1,
    added_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (org_id, sha256)
);

CREATE TABLE IF NOT EXISTS pages (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    PRIMARY KEY (document_id, page_no)
);

CREATE TABLE IF NOT EXISTS queries (
    id                INTEGER PRIMARY KEY,
    org_id            INTEGER NOT NULL REFERENCES orgs(id),
    question          TEXT NOT NULL,
    answer            TEXT NOT NULL,
    grounded          INTEGER NOT NULL,
    model             TEXT NOT NULL,
    selected_doc_ids  TEXT NOT NULL,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    cache_read_tokens INTEGER,
    latency_ms        INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS citations (
    id          INTEGER PRIMARY KEY,
    query_id    INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id),
    page_start  INTEGER,
    page_end    INTEGER,
    cited_text  TEXT NOT NULL,
    -- 'quote' = the exact span the model cited, verified by the API.
    -- 'page'  = the page was in the prompt; the model was not able to prove it
    --           quoted from it. See results.QUOTE / results.PAGE.
    kind        TEXT NOT NULL DEFAULT 'quote'
);

CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_queries_org ON queries(org_id, created_at);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that CREATE TABLE IF NOT EXISTS cannot add to an existing db.

    Additive only. Anything needing a rewrite should be a real migration file,
    not a line here.
    """
    have = {r["name"] for r in conn.execute("PRAGMA table_info(citations)")}
    if "kind" not in have:
        conn.execute(
            "ALTER TABLE citations ADD COLUMN kind TEXT NOT NULL DEFAULT 'quote'"
        )


def get_or_create_org(conn: sqlite3.Connection, slug: str, name: str | None = None) -> int:
    row = conn.execute("SELECT id FROM orgs WHERE slug = ?", (slug,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO orgs (slug, name) VALUES (?, ?)", (slug, name or slug)
    )
    conn.commit()
    return cur.lastrowid
