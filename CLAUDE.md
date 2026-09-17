# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A document-grounded assistant for NDIS disability support workers. It answers questions **only** from the employer's own policy PDFs in `documents/`, with a citation behind every claim. The point of the product is the refusal: a worker acting on something that is not in their organisation's policy is the exact failure this exists to prevent. Treat "this isn't in your documents" as a correct answer, never a gap to fill.

## Commands

```bash
# Dependencies into .venv. pyproject sets [tool.uv] package = false, so this
# installs the declared deps without trying to build the project itself.
uv sync

# Run the app (Windows venv layout — .venv/Scripts, not .venv/bin)
./.venv/Scripts/uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload

# Index documents/ into SQLite. Prints per-file status and flags scanned PDFs
# that have no extractable text.
./.venv/Scripts/python.exe -m app.ingest

# Same thing over HTTP
curl -X POST http://127.0.0.1:8000/api/reindex

# What this instance is configured to answer with, and whether it can
curl http://127.0.0.1:8000/api/health

# End-to-end. On the ollama backend this takes minutes, not seconds.
curl -m 900 -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' -d '{"question":"..."}'
```

There is **no test suite, linter, or lockfile** in this repo, and no test/lint tooling in the venv. Verification is the loop above: reindex, `/api/health`, then a real question and read the JSON — `grounded`, `refusal`, and `usage.invented_citations` are the fields that tell you whether a change helped.

## Architecture

The pipeline is `documents/*.pdf → ingest → SQLite → selector → backend → Answer → FastAPI → static page`.

**Everything above `backends.py` is backend-agnostic.** Only that module knows which model answered. Both backends return the same `Answer` (`results.py`), so `answer.ask()`, the API layer, and the UI never branch on backend. Preserve this — it is the main structural invariant.

**Each backend owns its own retrieval**, because that is the real difference between them rather than an incidental one:

| | `ClaudeBackend` | `OllamaBackend` |
|---|---|---|
| Sends | whole PDFs (base64) | extracted page text |
| Selects with | `selector.select()` — per document | `selector.select_pages()` — per page |
| Citations | `kind=QUOTE`, exact span verified by the API | `kind=PAGE`, "this page was in the prompt" |
| Survives scanned PDFs | yes, the model renders pages | no, the selector cannot see them |

That `QUOTE`/`PAGE` distinction is deliberately surfaced all the way to the UI. A local model cannot prove it quoted anything, so the page-level case says so rather than dressing an unverified claim as a quotation. Do not collapse the two.

**`prompts.SYSTEM` is the product's main safety control**, not just prompt text. It is held byte-stable because it is the cached prefix of every Claude request — any edit invalidates the cache for all users.

**Every answer is logged** (`answer.log` → `queries` + `citations` tables) with sources, tokens and latency. The motivation is regulatory: "what did this tool tell someone in March" is a question an employer will eventually have to answer. Keep new fields flowing into the audit trail.

**`org_id` threads through every table and query** even though the app serves one tenant. `main._org()` is the single place a session would resolve a real org; nothing downstream changes when auth lands.

**Citation plumbing:** `page_end` is exclusive on both backends, matching the Anthropic API, so the UI and the `citations` table mean one thing. The local backend parses inline `[S3]` markers, drops any source number the model invented, and reports the count as `usage.invented_citations` so the damage stays measurable.

**Segments carry citations, one segment per cited span.** A numbered list where each step is cited arrives as one segment per step. The frontend has to stitch those back into a single list — see below.

## Local backend performance

The corpus (16 pages) fits inside `NDIS_LOCAL_MAX_PAGES`, so `select_pages()` sends all of it in deterministic document order and skips retrieval. That makes the prompt prefix **byte-identical between questions**, so a model still resident from the last request reuses its KV cache: measured 83.7 s of prompt processing cold versus 1.6 s warm.

Consequences worth knowing before you tune anything:

- **Do not lower `NDIS_LOCAL_MAX_PAGES` below the corpus page count.** It switches BM25 retrieval back on, varies the prompt per question, and costs a full cold prompt read every time.
- `NDIS_OLLAMA_KEEP_ALIVE` is what keeps that cache alive between questions. It holds ~6.5 GB resident; set it to `0` to trade the speed back for RAM.
- `num_ctx` is sized from a character estimate (~4.5 chars/token measured on this corpus) and capped by `NDIS_LOCAL_MAX_CTX`. Exceeding the cap raises a clear `BackendError` — Ollama silently truncates past `num_ctx`, which would drop sources the answer claims to cite, so refusing is deliberate.
- Ollama runs the model in a `llama-server.exe` child process. If it dies mid-request (allocation failure under memory pressure) the failure arrives as a 500 carrying a socket error that says nothing about the model. `_post` retries once, then reports it as the memory problem it is. Orphaned `llama-server.exe` processes whose parent is gone can hold gigabytes of commit charge — worth checking when things get slow or start crashing.

The comment in `prompts.py` above `LOCAL_SYSTEM` claiming there is "no prefix cache to invalidate" **is now stale**: with the whole corpus sent in stable order and the model kept warm, editing `LOCAL_SYSTEM` does invalidate a live KV cache.

## Frontend

`app/static/index.html` is one hand-written file — no build step, no framework, no dependencies. The renderer is not decorative:

- It parses the answer into block descriptors (`blocks()`) before emitting HTML, so a list split across several citation segments can be stitched into one list instead of restarting at "1." for every step.
- Lines are classified individually, so a lead-in (`"An advocate is allowed to:"`) followed by numbered steps does not swallow the first step into a paragraph.
- Lists honour the model's own starting number via `<ol start=…>`, and only continue an open list when that numbering actually continues.
- Citations are deduplicated by (document, pages, quoted text) for the source list, while markers reuse a number wherever the same source is cited again.

## Working in this repo

- Config is read once, at import, in `config.py`. `.env.example` documents every variable and why it is set where it is; keep it in step with `config.py`.
- `data/` and `documents/*` are gitignored. The PDFs are an employer's internal policy material — `documents.publishable` defaults to `0` and should only be set otherwise with a clear licence.
- Several comments reference a README that does not exist in this repo.
- Editing the JS in `index.html` through shell heredocs mangles backslash escapes (`\n` in a regex becomes a literal newline while `\s` survives). Use the Edit tool for that file.
