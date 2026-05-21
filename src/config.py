"""
Configuration module for the Customer Service Data Analyst Agent.

Loads environment variables, defines model IDs, API endpoints, paths,
and provides a Zscaler-safe HTTP client for corporate proxy environments.

Dual-mode design:
  - Dev mode:   combined_ca_bundle.pem present → Zscaler SSL bypass active
  - Grader mode: no .pem file → standard system SSL (works out of the box)
"""

import logging
import os
import ssl
from pathlib import Path

import httpx
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()

# --- API Configuration ---

NEBIUS_API_KEY: str = os.environ["NEBIUS_API_KEY"]
NEBIUS_BASE_URL: str = os.environ.get(
    "NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/"
)

# --- Model IDs ---
# Configurable via .env — grader can override if their account differs

AGENT_MODEL: str = os.environ.get("AGENT_MODEL", "deepseek-ai/DeepSeek-V3.2")
ROUTER_MODEL: str = os.environ.get(
    "ROUTER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
JUDGE_MODEL: str = os.environ.get(
    "JUDGE_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"
)
FALLBACK_AGENT_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct"

# --- Paths ---

PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_PATH: Path = DATA_DIR / "bitext_dataset.csv"
PROFILES_DIR: Path = PROJECT_ROOT / "profiles"
CHECKPOINTS_DB: Path = PROJECT_ROOT / "checkpoints.db"

# --- Agent Settings ---

MAX_ITERATIONS: int = 12
SEED: int = 42

# --- Zscaler / Corporate Proxy Auto-Detection ---
# If combined_ca_bundle.pem exists in the project root, we assume a corporate
# proxy (Zscaler) is intercepting TLS.  OpenSSL 3.x rejects Zscaler's
# intermediate CA because its Basic Constraints extension is not marked
# critical.  The fix: build an ssl.SSLContext with VERIFY_X509_STRICT
# cleared and pass it to httpx.  When the .pem is absent (grader's machine),
# get_http_client() returns None and all clients use default system SSL.

CA_BUNDLE: Path = PROJECT_ROOT / "combined_ca_bundle.pem"
USE_ZSCALER: bool = CA_BUNDLE.exists()


def get_http_client() -> httpx.Client | None:
    """Return a Zscaler-safe httpx.Client, or None for standard SSL."""
    if not USE_ZSCALER:
        return None
    ssl_ctx = ssl.create_default_context(cafile=str(CA_BUNDLE))
    ssl_ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return httpx.Client(verify=ssl_ctx)


def get_async_http_client() -> httpx.AsyncClient | None:
    """Async variant for Chainlit / async tool calls."""
    if not USE_ZSCALER:
        return None
    ssl_ctx = ssl.create_default_context(cafile=str(CA_BUNDLE))
    ssl_ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return httpx.AsyncClient(verify=ssl_ctx)


def get_llm(model: str, **kwargs):
    """Create a ChatOpenAI instance with automatic Zscaler handling.

    All LLM call sites should use this helper instead of constructing
    ChatOpenAI directly.  On a corporate machine the custom http_client
    is injected; on the grader's machine it is omitted.
    """
    from langchain_openai import ChatOpenAI

    http_client = get_http_client()
    extra = {"http_client": http_client} if http_client else {}
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


# --- Ensure directories exist ---

PROFILES_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
