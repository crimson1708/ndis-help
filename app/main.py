"""FastAPI app: chat page, ask endpoint, document listing, PDF serving."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import answer as answer_mod
from . import backends, config, db, ingest

app = FastAPI(title="NDIS Support Assistant")
STATIC = Path(__file__).parent / "static"


def _conn():
    conn = db.connect()
    db.init(conn)
    return conn


def _org(conn) -> int:
    # Single tenant for now. When auth lands, resolve the org from the session
    # here and nothing downstream changes.
    return db.get_or_create_org(conn, config.DEFAULT_ORG)


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")
    conn = _conn()
    try:
        result = answer_mod.ask(conn, _org(conn), question)
        query_id = answer_mod.log(conn, _org(conn), question, result)
        payload = result.to_dict()
        payload["query_id"] = query_id
        return payload
    except answer_mod.BackendError as exc:
        # Every cause is a setup problem - missing key, Ollama not running,
        # model not pulled - so say which one instead of a bare 500.
        raise HTTPException(503, str(exc)) from exc
    finally:
        conn.close()


@app.get("/api/health")
def health():
    """What this instance is configured to answer with, and whether it can."""
    conn = _conn()
    try:
        docs, pages = conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(page_count), 0)
               FROM documents WHERE org_id = ?""",
            (_org(conn),),
        ).fetchone()
    finally:
        conn.close()

    info = {"backend": config.BACKEND, "documents": docs, "pages": pages}
    try:
        backend = backends.get()
        info["model"] = backend.model_id
        info["ready"] = True
    except answer_mod.BackendError as exc:
        info["ready"] = False
        info["error"] = str(exc)
    return info


@app.get("/api/documents")
def documents():
    conn = _conn()
    try:
        rows = conn.execute(
            """SELECT id, filename, title, page_count, has_text, publishable, added_at
               FROM documents WHERE org_id = ? ORDER BY title""",
            (_org(conn),),
        ).fetchall()
        return {"documents": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/api/documents/{doc_id}/file")
def document_file(doc_id: int):
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT filename FROM documents WHERE id = ? AND org_id = ?",
            (doc_id, _org(conn)),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Document not found.")
        path = config.DOCUMENTS_DIR / row["filename"]
        if not path.exists():
            raise HTTPException(404, "File missing from documents/.")
        return FileResponse(path, media_type="application/pdf", filename=row["filename"])
    finally:
        conn.close()


@app.post("/api/reindex")
def reindex():
    conn = _conn()
    try:
        return {"results": ingest.ingest_all(conn)}
    finally:
        conn.close()
