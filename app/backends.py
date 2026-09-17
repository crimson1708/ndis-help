"""Model backends.

Everything above this module - ingest, retrieval, storage, the web UI - is
backend-agnostic. Only the code in here knows which model answered, and each
backend owns its own retrieval strategy, because that is the real difference
between them rather than an incidental one:

    ClaudeBackend  sends whole PDFs. The model reads layout, tables and scanned
                   pages, and the API returns the exact span behind each claim.
    OllamaBackend  sends extracted page text to a model on this machine. Free,
                   slower, and it can only be given pages - never the PDF - so
                   attribution is limited to what we put in front of it.

Both return the same `Answer`, so `answer.ask()` and the UI do not branch.
"""
import base64
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, prompts, selector
from .results import PAGE, QUOTE, Answer, Citation, Segment


class BackendError(RuntimeError):
    """A backend could not run - misconfigured, unreachable, or model missing.

    Carries a message meant to be shown to whoever is running the app, since
    every realistic cause is a setup problem they can fix.
    """


def _is_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "authentication" in text or "api_key" in text


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

class ClaudeBackend:
    name = "claude"

    def __init__(self):
        self._client = None
        # Set False the first time the installed SDK rejects `fallbacks`, so we
        # stop retrying a parameter this version does not accept.
        self._supports_fallbacks = True

    @property
    def model_id(self) -> str:
        return config.MODEL

    def client(self):
        if self._client is None:
            import anthropic
            # Constructing never fails - the SDK resolves credentials lazily and
            # raises at request time instead. Missing credentials therefore have
            # to be caught around the call, in _call.
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _no_credentials() -> "BackendError":
        return BackendError(
            "No Anthropic credentials found. Put ANTHROPIC_API_KEY in "
            f"{config.ROOT / '.env'}, or switch to the free local backend "
            "with NDIS_BACKEND=ollama."
        )

    def _document_block(self, sel: selector.Selection, path: Path, cache: bool) -> dict:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
            "title": sel.title[:200],
            # `context` reaches the model but is never cited from - the right
            # place for provenance the model should know but shouldn't quote.
            "context": f"Source file: {sel.filename}. {sel.page_count} pages. "
                       f"Organisation policy document.",
            "citations": {"enabled": True},
        }
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return block

    def _call(self, **kwargs):
        """Call the API, dropping `fallbacks` if this SDK version rejects it.

        Two different failures arrive as TypeError here: an SDK too old for
        `fallbacks`, and missing credentials. Only the first is worth retrying,
        so they are told apart by message rather than by type.
        """
        import anthropic

        if self._supports_fallbacks:
            try:
                with self.client().beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **kwargs,
                ) as stream:
                    return stream.get_final_message()
            except TypeError as exc:
                if _is_auth_error(exc):
                    raise self._no_credentials() from exc
                self._supports_fallbacks = False
        try:
            with self.client().messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        except TypeError as exc:
            if _is_auth_error(exc):
                raise self._no_credentials() from exc
            raise
        except anthropic.AuthenticationError as exc:
            raise BackendError(
                "Anthropic rejected the API key in .env - check it is current "
                "and belongs to a workspace with credit."
            ) from exc

    def answer(self, conn: sqlite3.Connection, org_id: int, question: str) -> Answer:
        started = time.monotonic()
        selected = selector.select(conn, org_id, question)
        if not selected:
            return _no_documents(self, started)

        content = []
        for i, sel in enumerate(selected):
            path = config.DOCUMENTS_DIR / sel.filename
            if not path.exists():
                continue
            # Cache breakpoint on the final document: everything above it - the
            # system prompt and all documents - is reused when the same set is
            # selected again.
            content.append(self._document_block(sel, path, cache=(i == len(selected) - 1)))
        content.append({"type": "text", "text": prompts.QUESTION_PREFIX + question})

        response = self._call(
            model=config.MODEL,
            max_tokens=8000,
            system=[{
                "type": "text",
                "text": prompts.SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            thinking={"type": "adaptive"},
            output_config={"effort": config.EFFORT},
            messages=[{"role": "user", "content": content}],
        )

        latency_ms = int((time.monotonic() - started) * 1000)
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", None),
        }
        sources = [
            {"document_id": s.document_id, "title": s.title, "filename": s.filename,
             "score": round(s.score, 3)}
            for s in selected
        ]

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            return Answer(
                text="I wasn't able to answer that one. Please check the policy "
                     "document directly or ask your supervisor.",
                segments=[], citations=[], grounded=False, sources=sources,
                usage=usage, latency_ms=latency_ms,
                backend=self.name, model=config.MODEL,
                refusal=getattr(detail, "category", None) or "unknown",
            )

        # document_index is 0-indexed across the document blocks in request order.
        by_index = {i: s for i, s in enumerate(selected)}

        segments: list[Segment] = []
        all_citations: list[Citation] = []
        for block in response.content:
            if block.type != "text":
                continue
            cites = []
            for raw in (getattr(block, "citations", None) or []):
                src = by_index.get(getattr(raw, "document_index", -1))
                cite = Citation(
                    document_id=src.document_id if src else None,
                    document_title=getattr(raw, "document_title", None) or (src.title if src else "Unknown"),
                    filename=src.filename if src else None,
                    page_start=getattr(raw, "start_page_number", None),
                    page_end=getattr(raw, "end_page_number", None),
                    cited_text=getattr(raw, "cited_text", ""),
                    kind=QUOTE,
                )
                cites.append(cite)
                all_citations.append(cite)
            segments.append(Segment(text=block.text, citations=cites))

        return Answer(
            text="".join(s.text for s in segments),
            segments=segments, citations=all_citations,
            grounded=bool(all_citations), sources=sources,
            usage=usage, latency_ms=latency_ms,
            backend=self.name, model=config.MODEL,
        )


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

# [S3] / [S3, S7] / [S3][S7] - all shapes a small model produces in practice.
_CITE_RE = re.compile(r"\[\s*S\s*(\d+(?:\s*,\s*S?\s*\d+)*)\s*\]", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")
# Qwen and friends emit their reasoning inline. Never show it to a support worker.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class OllamaBackend:
    name = "ollama"

    @property
    def model_id(self) -> str:
        return config.OLLAMA_MODEL

    # Ollama runs the model in a llama-server child process and proxies to it.
    # If that child dies mid-request - on constrained hardware, an allocation
    # failure - the failure reaches us as a 500 carrying a socket error, saying
    # nothing about the model. The next request gets a fresh child, so one
    # retry turns a hard failure into a slow success.
    _RUNNER_DIED = ("forcibly closed", "wsarecv", "connection reset",
                    "broken pipe", "unexpected eof", "eof")

    def _post(self, path: str, payload: dict, _retries: int = 1) -> dict:
        url = config.OLLAMA_HOST.rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            if exc.code == 404:
                raise BackendError(
                    f"Ollama has no model called '{config.OLLAMA_MODEL}'. "
                    f"Pull it first:  ollama pull {config.OLLAMA_MODEL}"
                ) from exc
            if exc.code >= 500 and any(m in body.lower() for m in self._RUNNER_DIED):
                if _retries > 0:
                    time.sleep(2)
                    return self._post(path, payload, _retries - 1)
                raise BackendError(
                    "The Ollama model process crashed while answering, twice. "
                    "On this machine that is almost always memory: the model "
                    f"and its {config.LOCAL_MAX_CTX:,}-token context have to fit "
                    "in RAM alongside everything else. Close what you can, or "
                    "lower NDIS_LOCAL_MAX_CTX / use a smaller model."
                ) from exc
            raise BackendError(f"Ollama returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                f"Cannot reach Ollama at {config.OLLAMA_HOST} ({exc.reason}). "
                "Start it with:  ollama serve"
            ) from exc

    def _corpus(self, pages: list[selector.PageSelection]) -> tuple[str, dict[int, selector.PageSelection]]:
        """Render the pages as a numbered source list.

        Source numbers are flat and 1-based across the whole prompt. A single
        integer is dramatically easier for a small model to reproduce correctly
        than a document/page pair, and it is all we need - we hold the mapping.
        """
        by_number: dict[int, selector.PageSelection] = {}
        parts = []
        for n, page in enumerate(pages, start=1):
            by_number[n] = page
            parts.append(
                f"[S{n}] {page.title} — page {page.page_no}\n"
                f"{page.text.strip()}"
            )
        return "\n\n".join(parts), by_number

    def _split(self, text: str, by_number: dict[int, selector.PageSelection]):
        """Turn inline [S<n>] markers into segments with citations attached.

        Any source number the model invents is dropped rather than rendered -
        a fabricated citation is worse than none, because it looks checkable.
        The count is reported in usage so the damage is measurable.
        """
        segments: list[Segment] = []
        all_citations: list[Citation] = []
        invented = 0
        cursor = 0

        for match in _CITE_RE.finditer(text):
            chunk = text[cursor:match.start()]
            cites: list[Citation] = []
            for num in _NUM_RE.findall(match.group(1)):
                page = by_number.get(int(num))
                if page is None:
                    invented += 1
                    continue
                cite = Citation(
                    document_id=page.document_id,
                    document_title=page.title,
                    filename=page.filename,
                    page_start=page.page_no,
                    # Claude's page_end is exclusive; match that so the UI and
                    # the citations table mean one thing, not two.
                    page_end=page.page_no + 1,
                    cited_text="",
                    kind=PAGE,
                )
                cites.append(cite)
                all_citations.append(cite)
            if cites:
                chunk = chunk.rstrip()
            if chunk.strip() or cites:
                segments.append(Segment(text=chunk, citations=cites))
            cursor = match.end()

        tail = text[cursor:]
        if tail.strip():
            segments.append(Segment(text=tail, citations=[]))

        # Models write "...was not met [S6]." - the marker lands before the full
        # stop, so removing it strands the punctuation at the head of the next
        # segment and renders as "was not met .". Pull it back onto the sentence
        # it belongs to.
        for i in range(1, len(segments)):
            lead = re.match(r"^\s*([.,;:!?)\]]+)", segments[i].text)
            if lead:
                segments[i - 1].text = segments[i - 1].text.rstrip() + lead.group(1)
                segments[i].text = segments[i].text[lead.end():]

        return segments, all_citations, invented

    def answer(self, conn: sqlite3.Connection, org_id: int, question: str) -> Answer:
        started = time.monotonic()
        pages = selector.select_pages(conn, org_id, question)
        if not pages:
            return _no_documents(self, started)

        corpus, by_number = self._corpus(pages)
        user = f"{corpus}\n\n{prompts.LOCAL_QUESTION_PREFIX}{question}"

        # Ollama silently truncates anything past num_ctx, which would drop
        # sources the answer claims to cite. Size the window to the prompt, and
        # refuse rather than truncate if it will not fit.
        # Measured on this corpus: 4.54 chars per token. Dividing by 3 overshot
        # by half, so num_ctx was sized at 12k for a prompt that needed 8k -
        # a KV cache half again bigger than necessary, which is real RAM on a
        # 16 GB machine. 4 keeps a ~12% margin for text that tokenises denser.
        est_prompt = (len(prompts.LOCAL_SYSTEM) + len(user)) // 4 + 256
        needed = est_prompt + config.LOCAL_MAX_OUTPUT
        if needed > config.LOCAL_MAX_CTX:
            raise BackendError(
                f"This prompt needs roughly {needed:,} tokens of context but "
                f"NDIS_LOCAL_MAX_CTX is {config.LOCAL_MAX_CTX:,}. Lower "
                f"NDIS_LOCAL_MAX_PAGES (currently {config.LOCAL_MAX_PAGES}) or "
                "raise the context limit - raising it costs RAM."
            )
        num_ctx = min(config.LOCAL_MAX_CTX, max(2048, -(-needed // 1024) * 1024))

        data = self._post("/api/chat", {
            "model": config.OLLAMA_MODEL,
            "stream": False,
            # Stay resident: the corpus prefix is identical every request, so a
            # warm slot turns the next question's prompt processing into a
            # cache hit instead of another full read.
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            # Ignored by models without a thinking mode.
            "think": config.OLLAMA_THINK,
            "messages": [
                {"role": "system", "content": prompts.LOCAL_SYSTEM},
                {"role": "user", "content": user},
            ],
            "options": {
                # Near-greedy. This is a lookup tool; creative variation in a
                # policy answer is a defect, not a feature.
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": num_ctx,
                "num_predict": config.LOCAL_MAX_OUTPUT,
            },
        })

        latency_ms = int((time.monotonic() - started) * 1000)
        raw = (data.get("message") or {}).get("content", "")
        text = _THINK_RE.sub("", raw).strip()
        segments, citations, invented = self._split(text, by_number)

        sources = [
            {"document_id": p.document_id, "title": p.title, "filename": p.filename,
             "page_no": p.page_no, "source_no": n, "score": round(p.score, 3)}
            for n, p in by_number.items()
        ]
        usage = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "num_ctx": num_ctx,
            "pages_sent": len(pages),
            "invented_citations": invented,
            # Nanoseconds, as Ollama reports them. Prompt processing is the
            # number that decides whether this is usable on a given machine.
            "prompt_eval_ms": (data.get("prompt_eval_duration") or 0) // 1_000_000,
            "eval_ms": (data.get("eval_duration") or 0) // 1_000_000,
        }

        return Answer(
            text="".join(s.text for s in segments) or text,
            segments=segments, citations=citations,
            grounded=bool(citations), sources=sources,
            usage=usage, latency_ms=latency_ms,
            backend=self.name, model=config.OLLAMA_MODEL,
        )


# ---------------------------------------------------------------------------

def _no_documents(backend, started: float) -> Answer:
    return Answer(
        text="I don't have any documents indexed that relate to that question. "
             "Check with your supervisor, and let your administrator know if a "
             "policy document is missing from this tool.",
        segments=[], citations=[], grounded=False, sources=[],
        usage={}, latency_ms=int((time.monotonic() - started) * 1000),
        backend=backend.name, model=backend.model_id,
    )


_REGISTRY = {"claude": ClaudeBackend, "ollama": OllamaBackend}
_cache: dict[str, object] = {}


def get(name: str | None = None):
    """Return the configured backend, built once per process."""
    name = (name or config.BACKEND).strip().lower()
    if name not in _REGISTRY:
        raise BackendError(
            f"Unknown backend '{name}'. Set NDIS_BACKEND to one of: "
            f"{', '.join(sorted(_REGISTRY))}."
        )
    if name not in _cache:
        _cache[name] = _REGISTRY[name]()
    return _cache[name]
