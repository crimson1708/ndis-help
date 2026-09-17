"""Runtime configuration, read once from the environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DOCUMENTS_DIR = ROOT / "documents"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "ndis.sqlite3"

MODEL = os.getenv("NDIS_MODEL", "claude-opus-5")
EFFORT = os.getenv("NDIS_EFFORT", "high")
DEFAULT_ORG = os.getenv("NDIS_DEFAULT_ORG", "default")

# Which model backend answers questions: "claude" or "ollama".
#
# claude - sends the original PDFs and gets back verified citations. What the
#          demo and anything real should run on.
# ollama - sends extracted page text to a model on this machine. Free, slower,
#          and its attribution is page-level rather than quote-level: see
#          results.PAGE. Good for iterating on prompts and UI without a meter.
BACKEND = os.getenv("NDIS_BACKEND", "claude")

OLLAMA_HOST = os.getenv("NDIS_OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("NDIS_OLLAMA_MODEL", "qwen3:8b")
# Generation on an integrated GPU is minutes, not seconds. Well above any
# sensible HTTP default.
OLLAMA_TIMEOUT = int(os.getenv("NDIS_OLLAMA_TIMEOUT", "600"))
# Reasoning models (qwen3 and friends) think before answering. Measured on this
# corpus, leaving it ON is strictly better: same generation time, and the answer
# is a summary of the procedure rather than the policy copied out line by line
# (12 steps instead of 19). Turning it off did not buy speed.
OLLAMA_THINK = os.getenv("NDIS_OLLAMA_THINK", "1").strip().lower() in {"1", "true", "yes"}
# How long Ollama keeps the model resident after answering. The prompt is a
# stable prefix - the whole corpus, always in document order - so a model that
# is still warm reuses its KV cache and a follow-up question skips prompt
# processing entirely. On this machine that is the difference between four
# minutes and seconds. The cost is the model sitting in RAM between questions.
OLLAMA_KEEP_ALIVE = os.getenv("NDIS_OLLAMA_KEEP_ALIVE", "30m")

# How much of the corpus is allowed into a single request. The PDF page ceiling
# for a 1M-context model is 600; we stay well under it so answers stay focused
# and cheap. Raise MAX_PAGES if the selector is missing relevant material.
MAX_DOCS = int(os.getenv("NDIS_MAX_DOCS", "8"))
MAX_PAGES = int(os.getenv("NDIS_MAX_PAGES", "300"))
MAX_BYTES = int(os.getenv("NDIS_MAX_BYTES", "20000000"))

# The local budget is far tighter, and for a different reason: a local model
# must read the whole prompt before it emits a token, at a few hundred tokens
# per second. 24 pages is roughly 15k tokens - about a minute of reading on an
# integrated GPU, and enough to hold a small corpus whole.
LOCAL_MAX_PAGES = int(os.getenv("NDIS_LOCAL_MAX_PAGES", "24"))
# Ceiling on the context window requested from Ollama. Raising this costs RAM
# (KV cache); 16k fits an 8B model comfortably in 16 GB.
LOCAL_MAX_CTX = int(os.getenv("NDIS_LOCAL_MAX_CTX", "16384"))
LOCAL_MAX_OUTPUT = int(os.getenv("NDIS_LOCAL_MAX_OUTPUT", "1200"))

DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)
