"""Result types shared by the answer pipeline and every model backend.

These live apart from answer.py so a backend can build them without importing
the orchestrator that calls the backend.
"""
from dataclasses import dataclass, field, asdict


# How a citation was established. The distinction is the honest part of the
# local backend: Claude returns the exact span it drew on, verified server-side;
# a local model is merely told which pages it was given, so the strongest true
# claim is "this page was in front of it", not "it quoted this".
QUOTE = "quote"   # exact span the model cited, returned by the Claude API
PAGE = "page"     # page-level attribution: we know what we sent, not what it read


@dataclass
class Citation:
    document_id: int | None
    document_title: str
    filename: str | None
    page_start: int | None
    page_end: int | None   # exclusive, as returned by the API
    cited_text: str
    kind: str = QUOTE


@dataclass
class Segment:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    segments: list[Segment]
    citations: list[Citation]
    grounded: bool
    sources: list[dict]
    usage: dict
    latency_ms: int
    backend: str = "claude"
    model: str = ""
    refusal: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [
                {"text": s.text, "citations": [asdict(c) for c in s.citations]}
                for s in self.segments
            ],
            "citations": [asdict(c) for c in self.citations],
            "grounded": self.grounded,
            "sources": self.sources,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "model": self.model,
            "refusal": self.refusal,
        }
