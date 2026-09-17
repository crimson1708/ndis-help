"""Scan documents/ for PDFs and build the selection index.

The extracted text is only used to decide *which* PDFs to send. The model
always reads the original PDF, so layout, tables and scanned pages survive
even when extraction here is poor.
"""
import hashlib
import re
import sqlite3
from pathlib import Path

from pypdf import PdfReader

from . import config, db


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _title(path: Path, reader: PdfReader) -> str:
    meta_title = (reader.metadata or {}).get("/Title") if reader.metadata else None
    if meta_title and str(meta_title).strip():
        return str(meta_title).strip()
    # "high-intensity_seizure-support.pdf" -> "High Intensity Seizure Support"
    stem = re.sub(r"[_\-]+", " ", path.stem)
    return re.sub(r"\s+", " ", stem).strip().title()


def ingest_file(conn: sqlite3.Connection, org_id: int, path: Path) -> dict:
    digest = _sha256(path)
    existing = conn.execute(
        "SELECT id, filename FROM documents WHERE org_id = ? AND sha256 = ?",
        (org_id, digest),
    ).fetchone()
    if existing:
        return {"path": path.name, "status": "unchanged", "document_id": existing["id"]}

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    extracted = sum(len(p.strip()) for p in pages)
    has_text = extracted > 50 * max(1, len(pages)) // 10

    # Same filename, new content: replace the old row so re-ingesting an
    # updated policy document does not leave the superseded version searchable.
    conn.execute(
        "DELETE FROM documents WHERE org_id = ? AND filename = ?", (org_id, path.name)
    )
    cur = conn.execute(
        """INSERT INTO documents
           (org_id, filename, title, sha256, page_count, byte_size, has_text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            org_id,
            path.name,
            _title(path, reader),
            digest,
            len(reader.pages),
            path.stat().st_size,
            int(has_text),
        ),
    )
    doc_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO pages (document_id, page_no, text) VALUES (?, ?, ?)",
        [(doc_id, i + 1, text) for i, text in enumerate(pages)],
    )
    conn.commit()
    return {
        "path": path.name,
        "status": "indexed",
        "document_id": doc_id,
        "pages": len(pages),
        "has_text": has_text,
    }


def ingest_all(conn: sqlite3.Connection, org_slug: str | None = None) -> list[dict]:
    org_id = db.get_or_create_org(conn, org_slug or config.DEFAULT_ORG)
    results = []
    for path in sorted(config.DOCUMENTS_DIR.glob("**/*.pdf")):
        results.append(ingest_file(conn, org_id, path))

    # Drop rows whose file has been deleted from documents/.
    on_disk = {p.name for p in config.DOCUMENTS_DIR.glob("**/*.pdf")}
    for row in conn.execute(
        "SELECT id, filename FROM documents WHERE org_id = ?", (org_id,)
    ).fetchall():
        if row["filename"] not in on_disk:
            conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
            results.append({"path": row["filename"], "status": "removed"})
    conn.commit()
    return results


if __name__ == "__main__":
    conn = db.connect()
    db.init(conn)
    for r in ingest_all(conn):
        line = f"  {r['status']:<10} {r['path']}"
        if r.get("pages"):
            line += f"  ({r['pages']} pages)"
        if r.get("has_text") is False:
            line += "  [!] no extractable text - scanned? selector cannot see it"
        print(line)
