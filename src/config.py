"""
Configuration module for the Customer Service Data Analyst Agent.

Loads environment variables, defines model IDs, API endpoints, paths,
and provides a Zscaler-safe HTTP client for corporate proxy environments.

Dual-mode design:
  - Dev mode:   combined_ca_bundle.pem present → Zscaler SSL bypass active
  - Grader mode: no .pem file → standard system SSL (works out of the box)
"""

import json
import logging
import os
import ssl
from pathlib import Path

import httpx
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()

os.environ.setdefault("MEM0_TELEMETRY", "false")

# --- API Configuration ---

NEBIUS_API_KEY: str = os.environ["NEBIUS_API_KEY"]
NEBIUS_BASE_URL: str = os.environ.get(
    "NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/"
)

# --- Model IDs ---
# Priority: model_selection.json > .env > defaults.
# Admin UI writes model_selection.json; .env is for grader overrides.

_MODEL_SELECTION_PATH: Path = Path(__file__).parent.parent / "model_selection.json"

def _load_model_selection() -> dict:
    """Load persisted model selection, if any."""
    try:
        if _MODEL_SELECTION_PATH.exists():
            with open(_MODEL_SELECTION_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_model_selection(data: dict) -> None:
    """Persist model selection to JSON."""
    with open(_MODEL_SELECTION_PATH, "w") as f:
        json.dump(data, f, indent=2)

_saved = _load_model_selection()

AGENT_MODEL: str = (
    _saved.get("agent_model")
    or os.environ.get("AGENT_MODEL")
    or "Qwen/Qwen3-235B-A22B-Instruct-2507"
)
ROUTER_MODEL: str = os.environ.get(
    "ROUTER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
JUDGE_MODEL: str = (
    _saved.get("judge_model")
    or os.environ.get("JUDGE_MODEL")
    or "meta-llama/Llama-3.3-70B-Instruct"
)
FALLBACK_AGENT_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def set_agent_model(model_id: str) -> None:
    """Update agent model at runtime and persist."""
    global AGENT_MODEL
    AGENT_MODEL = model_id
    data = _load_model_selection()
    data["agent_model"] = model_id
    _save_model_selection(data)
    log.info("Agent model changed to %s", model_id)


def set_judge_model(model_id: str) -> None:
    """Update judge model at runtime and persist."""
    global JUDGE_MODEL
    JUDGE_MODEL = model_id
    data = _load_model_selection()
    data["judge_model"] = model_id
    _save_model_selection(data)
    log.info("Judge model changed to %s", model_id)

# Mem0 models (semantic memory — fact extraction + embeddings)
MEM0_LLM_MODEL: str = os.environ.get(
    "MEM0_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

# --- Paths ---

PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_PATH: Path = DATA_DIR / "bitext_dataset.csv"
CHECKPOINTS_DB: Path = PROJECT_ROOT / "checkpoints.db"

# Mem0 storage: relative paths resolve against PROJECT_ROOT, absolute paths used as-is
_mem0_path_raw: str = os.environ.get("MEM0_STORAGE_PATH", "mem0_data")
MEM0_DATA_DIR: Path = (
    Path(_mem0_path_raw) if Path(_mem0_path_raw).is_absolute()
    else PROJECT_ROOT / _mem0_path_raw
)

# --- Agent Settings ---

MAX_ITERATIONS: int = 12
RECURSION_LIMIT: int = 2 * MAX_ITERATIONS + 4

# --- Quality Scoring ---

ENABLE_QUALITY_SCORING: bool = (
    os.environ.get("ENABLE_QUALITY_SCORING", "false").lower() == "true"
)
QUALITY_SCORE_THRESHOLD: int = int(
    os.environ.get("QUALITY_SCORE_THRESHOLD", "3")
)
AUTO_RETRY_ON_LOW_SCORE: bool = (
    os.environ.get("AUTO_RETRY_ON_LOW_SCORE", "false").lower() == "true"
)

# --- Reflection ---

ENABLE_REFLECTION: bool = (
    os.environ.get("ENABLE_REFLECTION", "true").lower() == "true"
)

# --- Query Decomposition ---

ENABLE_DECOMPOSITION: bool = (
    os.environ.get("ENABLE_DECOMPOSITION", "true").lower() == "true"
)

# --- Model Pricing (per 1M tokens) ---

MODEL_PRICING: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"input": 0.20, "output": 3.60},
    "Qwen/Qwen3.5-397B-A17B": {"input": 0.60, "output": 3.60},
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {"input": 0.10, "output": 0.30},
    "Qwen/Qwen3-32B": {"input": 0.10, "output": 0.30},
    "meta-llama/Llama-3.3-70B-Instruct": {"input": 0.13, "output": 0.40},
    "NousResearch/Hermes-4-70B": {"input": 0.13, "output": 0.48},
    "NousResearch/Hermes-4-405B": {"input": 0.60, "output": 1.80},
    "zai-org/GLM-5.1": {"input": 1.40, "output": 4.80},
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1": {"input": 0.60, "output": 1.80},
    "google/gemma-3-27b-it": {"input": 0.10, "output": 0.30},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a given model and token counts."""
    pricing = MODEL_PRICING.get(model, {"input": 0.20, "output": 0.60})
    return (
        prompt_tokens * pricing["input"]
        + completion_tokens * pricing["output"]
    ) / 1_000_000
SEED: int = 42
HEALTH_CHECK_INTERVAL_SECONDS: int = int(
    os.environ.get("HEALTH_CHECK_INTERVAL", "300")
)

# --- Zscaler / Corporate Proxy Auto-Detection ---
# If combined_ca_bundle.pem exists in the project root, we assume a corporate
# proxy (Zscaler) is intercepting TLS.  OpenSSL 3.x rejects Zscaler's
# intermediate CA because its Basic Constraints extension is not marked
# critical.  The fix: build an ssl.SSLContext with VERIFY_X509_STRICT
# cleared and pass it to httpx.  When the .pem is absent (grader's machine),
# get_http_client() returns None and all clients use default system SSL.

CA_BUNDLE: Path = PROJECT_ROOT / "combined_ca_bundle.pem"
USE_ZSCALER: bool = CA_BUNDLE.exists()

if USE_ZSCALER:
    os.environ.setdefault("SSL_CERT_FILE", str(CA_BUNDLE))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(CA_BUNDLE))

    _original_create_default_context = ssl.create_default_context

    def _zscaler_ssl_context(*args, **kwargs):
        ctx = _original_create_default_context(*args, **kwargs)
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        if "cafile" not in kwargs and "capath" not in kwargs:
            ctx.load_verify_locations(cafile=str(CA_BUNDLE))
        return ctx

    ssl.create_default_context = _zscaler_ssl_context
    log.info("Zscaler CA bundle detected — patched ssl.create_default_context globally")


def get_http_client() -> httpx.Client | None:
    """Return a Zscaler-safe httpx.Client, or None for standard SSL."""
    if not USE_ZSCALER:
        return None
    return httpx.Client()


def get_async_http_client() -> httpx.AsyncClient | None:
    """Async variant for Chainlit / async tool calls."""
    if not USE_ZSCALER:
        return None
    return httpx.AsyncClient()


_THINKING_MODELS = {"Qwen/Qwen3.5-397B-A17B", "zai-org/GLM-5.1"}


def get_llm(model: str, **kwargs):
    """Create a ChatOpenAI instance with automatic Zscaler handling.

    All LLM call sites should use this helper instead of constructing
    ChatOpenAI directly.  On a corporate machine the custom http_client
    is injected; on the grader's machine it is omitted.

    For models that default to "thinking" mode (e.g. Qwen3.5), we inject
    chat_template_kwargs to disable it — thinking wastes tokens in
    tool-calling flows where we need direct responses.
    """
    from langchain_openai import ChatOpenAI

    http_client = get_http_client()
    extra = {"http_client": http_client} if http_client else {}

    if model in _THINKING_MODELS:
        eb = kwargs.pop("extra_body", {})
        eb.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        kwargs["extra_body"] = eb

    return ChatOpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=NEBIUS_API_KEY,
        model=model,
        seed=SEED,
        **extra,
        **kwargs,
    )


# --- Summarizer Strategy ---
# "economy"  -> cheapest of AGENT_MODEL / ROUTER_MODEL (via live pricing)
# "quality"  -> AGENT_MODEL
# "router"   -> ROUTER_MODEL
# <model-id> -> explicit model ID

SUMMARIZER_STRATEGY: str = os.environ.get("SUMMARIZER_STRATEGY", "economy")

_model_prices: dict[str, dict[str, float]] | None = None


def get_model_prices() -> dict[str, dict[str, float]]:
    """Fetch per-token pricing from Nebius /v1/models?verbose=true.

    Lazy-loaded on first call, cached for the process lifetime.
    Returns a dict keyed by model ID with "prompt" and "completion"
    per-token prices as floats.
    """
    global _model_prices
    if _model_prices is not None:
        return _model_prices

    models_url = NEBIUS_BASE_URL.rstrip("/").rsplit("/v1", 1)[0] + "/v1/models"
    client = get_http_client() or httpx.Client()
    try:
        resp = client.get(
            models_url,
            params={"verbose": "true"},
            headers={"Authorization": f"Bearer {NEBIUS_API_KEY}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _model_prices = {}
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            _model_prices[m["id"]] = {
                "prompt": float(pricing.get("prompt", 0)),
                "completion": float(pricing.get("completion", 0)),
            }
        log.info("Fetched pricing for %d models from Nebius API", len(_model_prices))
    except Exception:
        log.warning("Failed to fetch model pricing; falling back to ROUTER_MODEL")
        _model_prices = {}
    return _model_prices


def get_summarizer_model() -> str:
    """Resolve SUMMARIZER_STRATEGY to a concrete model ID."""
    if SUMMARIZER_STRATEGY == "quality":
        return AGENT_MODEL
    if SUMMARIZER_STRATEGY == "router":
        return ROUTER_MODEL
    if SUMMARIZER_STRATEGY == "economy":
        prices = get_model_prices()
        candidates = {AGENT_MODEL: prices.get(AGENT_MODEL), ROUTER_MODEL: prices.get(ROUTER_MODEL)}
        priced = {m: p for m, p in candidates.items() if p is not None}
        if priced:
            return min(priced, key=lambda m: priced[m]["completion"])
        return ROUTER_MODEL
    return SUMMARIZER_STRATEGY


# --- Mem0 Configuration ---

MEM0_CONFIG: dict = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": MEM0_LLM_MODEL,
            "api_key": NEBIUS_API_KEY,
            "openai_base_url": NEBIUS_BASE_URL,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": EMBEDDING_MODEL,
            "api_key": NEBIUS_API_KEY,
            "openai_base_url": NEBIUS_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": str(MEM0_DATA_DIR),
            "embedding_model_dims": 4096,
        },
    },
    "history_db_path": str(MEM0_DATA_DIR / "history.db"),
}

# --- Ensure directories exist ---

MEM0_DATA_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
