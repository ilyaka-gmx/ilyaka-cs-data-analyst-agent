# Customer Service Data Analyst Agent

A LangGraph-based ReAct agent that answers structured and unstructured questions about the [Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26,872 rows, 11 categories, 27 intents).

The agent classifies queries via a dedicated router, uses 11 typed tools with Pydantic schemas, persists conversations via SQLite, maintains per-user semantic memory via [mem0](https://github.com/mem0ai/mem0) (Qdrant + LLM-based fact extraction), exposes tools through MCP, and includes a cross-model quality judge. All LLM inference runs on [Nebius Token Factory](https://tokenfactory.nebius.com/).

## Table of Contents

- [Agent Quick Setup](#agent-quick-setup)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running the Application](#running-the-application)
- [Tests](#tests)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)
- [Requirements Mapping](#requirements-mapping)
- [Token Optimization](#token-optimization)
- [Models Observation](#models-observation)
- [Future Improvements](#future-improvements)

## Agent Quick Setup

> **This section is for AI agents** (Cursor, Claude Code, Codex, etc.) picking up this project.
> Humans — skip to [Quick Start](#quick-start).

**Step 1 — Get the API key:**
**STOP. Ask the user for their Nebius API key. Do not proceed until you have it.**

**Step 2 — Install:**
```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

**Step 3 — Configure (write the key from step 1):**
```bash
cp .env.example .env
# Replace <key> with the actual key the user gave you in step 1
echo 'NEBIUS_API_KEY=<key>' >> .env
```

**Step 4 — Fast sanity (runs without API key — but complete step 3 first):**
```bash
uv run pytest tests/test_tools.py tests/test_ui.py tests/test_quality.py tests/test_mcp.py -q
```
If any tests fail with SSL errors, see [SSL / Corporate Proxy](#ssl--corporate-proxy-zscaler-case) before retrying.

**Step 5 — Health check (needs the API key from step 1):**
```bash
uv run python main.py --health
```

**Step 6 — Start the web server:**
```bash
uv run python api_server.py
# → http://localhost:8000
```

**If step 3 passes**: dependencies are correct, dataset loads, tools work.
**If step 4 passes**: Nebius API key is valid, models respond.
**If the server starts**: the project is fully operational.

Key files: `src/agent.py` (graph), `src/tools.py` (11 tools), `src/prompts.py` (all prompts), `src/config.py` (all config), `api_server.py` (REST API).

## Prerequisites

| Tool                 | Version   | Required              | Install                                                                                   |
| -------------------- | --------- | --------------------- | ----------------------------------------------------------------------------------------- |
| **Python**           | 3.10–3.13 | yes                   | [python.org](https://www.python.org/downloads/) or `brew install python@3.13`             |
| **uv** (recommended) | latest    | no (pip works too)    | `curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh)            |
| **pip**              | latest    | yes (if not using uv) | Bundled with Python                                                                       |
| **git**              | any       | yes                   | `brew install git` or [git-scm.com](https://git-scm.com/)                                 |
| **Nebius API key**   | —         | yes                   | Sign up at [tokenfactory.nebius.com](https://tokenfactory.nebius.com/), create an API key |

## Quick Start

```bash
# Clone and enter the project
git clone <repo-url> && cd <repo-name>

# Create virtual environment
uv venv && source .venv/bin/activate
# Alternative without uv:
#   python -m venv .venv && source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# Alternative without uv:
#   pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and set NEBIUS_API_KEY=<your-key>

# Verify setup
uv run python main.py --health
```

The health check validates dataset integrity, API connectivity (agent and router models), and persistence.

## Running the Application

### CLI

The CLI is the simplest interface for interacting with the agent. It prints each reasoning step (tool calls and results) before the final answer, plus per-query token usage.

```bash
uv run python main.py --session demo --user alice
```

| Parameter   | Required | Default     | Description                                                                         |
| ----------- | -------- | ----------- | ----------------------------------------------------------------------------------- |
| `--session` | no       | random UUID | Conversation thread ID. Reuse to continue a previous conversation.                  |
| `--user`    | no       | `"default"` | User ID for profile isolation. Each user has their own profile and session history. |
| `--health`  | no       | —           | Run diagnostics and exit (no interactive session).                                  |

Type `quit`, `exit`, or press Ctrl+C to exit.

### Web Frontend (primary)

```bash
uv run python api_server.py
```

Open [http://localhost:8000](http://localhost:8000). The custom HTML frontend provides:

- SSE streaming with live reasoning steps (including decomposition plan)
- Suggestion chips (contextual after each response)
- Admin tab with per-query execution traces and runtime model selector
- Memory Insights tab (4-panel dashboard with mem0 health)
- Judge card showing quality scores (when enabled)
- Theme selector, conversation management, user picker
- Chat search, filtering, and tag management (auto + manual)
- Feature toggles: past sessions, quality scoring, reflection, decomposition
- Status bar with active model, dataset stats, and token/cost counters

### Streamlit (legacy fallback)

```bash
uv run streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). Same agent, simpler UI.

### MCP Server

```bash
# Start the MCP server (stdio transport)
uv run fastmcp run src/mcp_server.py:mcp

# Inspect available tools
uv run fastmcp inspect src/mcp_server.py
```

Exposes 7 data tools. Connect with any MCP-compatible client (Claude Desktop, Cursor, etc.). Memory tools are excluded (they require agent-level user context).

## Tests

138 tests across 8 files. Tests marked `@pytest.mark.slow` call real LLMs and require a valid API key.

```bash
# Fast tests only (mocked LLM, ~10 seconds)
uv run pytest -m "not slow"

# Full suite including LLM integration tests (~2 minutes)
uv run pytest

# Run a specific test file
uv run pytest tests/test_tools.py

# Run a specific test by name
uv run pytest -k "test_count_rows"

# Run tests with verbose output
uv run pytest -v -m "not slow"
```

| File                   | Tests | What it covers                                |
| ---------------------- | ----- | --------------------------------------------- |
| `test_tools.py`        | 27    | All 11 tools, metadata, dynamic tool exposure |
| `test_router.py`       | 17    | Classification accuracy (slow — real LLM)     |
| `test_agent.py`        | 21    | Full graph integration, persistence, mem0 memory |
| `test_api.py`          | 19    | FastAPI endpoints, SSE parsing                |
| `test_ui.py`           | 21    | UI helpers, SessionStore, tagging             |
| `test_mcp.py`          | 12    | MCP tool registration and invocation          |
| `test_quality.py`      | 14    | Judge scoring (mocked LLM)                    |
| `test_connectivity.py` | 7     | Nebius API connectivity (slow)                |

## Architecture

See [docs/architecture.md](docs/architecture.md) for full diagrams and design details.

### High-level flow

```
User Query → Router (Qwen3-30B) → classification
  ├── structured/unstructured/recommend → Decompose (query plan) → Agent ReAct Loop (Qwen3-235B) ↔ Tools → Reflect (self-check) → Answer
  └── out_of_scope → Static decline message (no LLM call)
```

### Models

All models accessed via [Nebius Token Factory](https://tokenfactory.nebius.com/) (OpenAI-compatible API). IDs configurable via `.env`.

| Role       | Model                    | Why this model                                                |
| ---------- | ------------------------ | ------------------------------------------------------------- |
| **Agent**  | Qwen3-235B-A22B-Instruct | Reliable tool calling in multi-step ReAct loops               |
| **Router** | Qwen3-30B-A3B-Instruct   | Fast, cheap classification (3.3× cheaper than agent)          |
| **Judge**  | Llama-3.3-70B-Instruct   | Cross-family evaluation — Llama judging Qwen avoids self-bias |
| **Mem0 LLM** | Qwen3-30B-A3B-Instruct | Fact extraction and normalization for semantic memory (same model as router — cost-effective) |
| **Mem0 Embedder** | Qwen3-Embedding-8B  | Embedding model for vectorizing and searching user memories (4096-dim vectors)  |

**Why Llama works as judge but not as agent**: Llama-3.3-70B enters infinite tool-calling loops inside a ReAct loop (repeats the same tool call until the iteration limit), but works perfectly for single-shot evaluation tasks where no tool calling is needed. DeepSeek-V3.2 was also rejected as agent due to a DSML tool-call parsing bug on the Nebius backend. See [architecture doc, Section 9](docs/architecture.md#9-models) for details.

### Tools (11)

| Tool                   | Purpose                                        |
| ---------------------- | ---------------------------------------------- |
| `list_categories`      | List all 11 categories                         |
| `list_intents`         | List intents (optionally filtered by category) |
| `count_rows`           | Count rows with optional filters               |
| `get_distribution`     | Frequency breakdown by category or intent      |
| `get_examples`         | Random sample of N rows                        |
| `search_instructions`  | Keyword search in customer text                |
| `summarize_responses`  | LLM-powered summary of agent responses         |
| `remember_fact`        | Save a fact to user profile                    |
| `recall_profile`       | Read user profile (supports semantic search via optional `query`)  |
| `update_profile`       | Replace user profile (after confirmation)      |
| `recall_past_sessions` | Search past session history                    |

Tools are **dynamically exposed** based on the router's query classification — structured queries don't see `summarize_responses`; recommend queries see only memory tools. This reduces token overhead by ~40%.

### Memory

| Layer          | Storage                        | What it stores                                                                     |
| -------------- | ------------------------------ | ---------------------------------------------------------------------------------- |
| **Short-term** | `checkpoints.db` (SqliteSaver) | Active conversation history for the current thread                                 |
| **Episodic**   | `session_store.json`           | Records of past interactions: queries asked, tools used, durations, quality scores |
| **Semantic**   | `mem0_data/` (Qdrant + history.db via [mem0](https://github.com/mem0ai/mem0)) | Distilled user facts with semantic deduplication and embedding-based recall |
| **Procedural** | Computed from session store    | Aggregated patterns: tool usage frequency, query type distribution                 |

### Middleware

Real `AgentMiddleware` subclasses (requires `langchain>=1.3.0`, not just `langchain-core`):

- **TokenTrackingMiddleware**: tracks prompt/completion tokens per query and session
- **ToolTimingMiddleware**: records wall-clock time per tool call
- Tool input bounds enforced via Pydantic `Field(ge=, le=)` on schemas
- Message trimming (last 30) replaces LLM-based summarization

### Quality Judge

Optional cross-model scoring: a Llama judge evaluates each Qwen agent response on three dimensions (1–5 each):

- **data_grounded** — does the response faithfully represent tool output? (penalizes fabrication and data relabeling)
- **addresses_question** — does it actually answer what was asked?
- **conciseness** — is it appropriately brief?

Overall = average; passes if ≥ 3. Toggle via settings or `ENABLE_QUALITY_SCORING=true` in `.env`.

## Configuration

All configuration is in `.env` (copy from `.env.example`):

| Variable                  | Required | Default                              | Description                                                 |
| ------------------------- | -------- | ------------------------------------ | ----------------------------------------------------------- |
| `NEBIUS_API_KEY`          | **yes**  | —                                    | Nebius Token Factory API key                                |
| `NEBIUS_BASE_URL`         | no       | `https://api.studio.nebius.ai/v1/`   | API endpoint                                                |
| `AGENT_MODEL`             | no       | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Agent model ID                                              |
| `ROUTER_MODEL`            | no       | `Qwen/Qwen3-30B-A3B-Instruct-2507`   | Router model ID                                             |
| `JUDGE_MODEL`             | no       | `meta-llama/Llama-3.3-70B-Instruct`  | Judge model ID                                              |
| `ENABLE_QUALITY_SCORING`  | no       | `false`                              | Enable quality judge. JUDGE_MODEL is required when enabled. |
| `QUALITY_SCORE_THRESHOLD` | no       | `3`                                  | Minimum pass score (1–5)                                    |
| `SUMMARIZER_STRATEGY`     | no       | `economy`                            | `economy` / `quality` / `router` / explicit model ID        |
| `ENABLE_REFLECTION`       | no       | `true`                               | Self-check before final response; nudges lazy answers       |
| `ENABLE_DECOMPOSITION`    | no       | `true`                               | Pre-execution query planning for ambiguous queries          |
| `MEM0_LLM_MODEL`          | no       | `Qwen/Qwen3-30B-A3B-Instruct-2507`   | LLM for mem0 fact extraction                                |
| `EMBEDDING_MODEL`         | no       | `Qwen/Qwen3-Embedding-8B`            | Embedding model for semantic memory search                  |
| `MEM0_STORAGE_PATH`       | no       | `mem0_data`                          | Local path for Qdrant vector store                          |

## Project Structure

```
├── src/
│   ├── config.py           # Config, LLM factory, Zscaler handling
│   ├── data.py             # Dataset loading and validation
│   ├── prompts.py          # All prompt strings
│   ├── router.py           # Query classifier
│   ├── agent.py            # LangGraph StateGraph (ReAct loop)
│   ├── tools.py            # 11 tools with Pydantic schemas
│   ├── toon.py             # TOON output formatter
│   ├── memory.py           # Semantic memory via mem0 (fact extraction, vector recall)
│   ├── middleware.py        # Token + timing middleware
│   ├── session_store.py     # Chat metadata and traces
│   ├── quality.py          # Cross-model judge
│   ├── recommender.py      # Query suggestion engine
│   ├── health.py           # Startup diagnostics
│   ├── model_catalog.py    # Nebius model catalog (quality, pricing, pros/cons)
│   ├── ui_helpers.py        # Shared UI formatting
│   └── mcp_server.py       # FastMCP server (7 tools)
├── frontend/
│   ├── index.html          # Lean HTML shell (primary web UI)
│   ├── styles.css          # All CSS (theme, layout, components)
│   └── js/                 # 11 JS modules (shared App namespace)
├── tests/                  # 138 tests across 8 files
├── data/
│   └── bitext_dataset.csv  # Dataset (~19 MB, committed to git)
├── docs/
│   ├── architecture.md         # Architecture document
│   ├── requirements_mapping.md # Requirements-to-implementation mapping
│   └── models-observation.md   # Open-source LLM quality analysis and findings
├── main.py                 # CLI entry point
├── api_server.py           # FastAPI backend
├── streamlit_app.py        # Streamlit UI (legacy)
├── langgraph.json          # LangGraph graph config (for LangSmith / langgraph CLI)
├── pyproject.toml          # Pytest configuration
├── requirements.txt        # Python dependencies
└── .env.example            # Environment template
```

## Debugging and Troubleshooting

### LangGraph configuration

The project includes `langgraph.json` which registers the agent graph entry point (`src/agent.py:build_graph`). This file is used by:

- **LangGraph CLI** — `langgraph dev` launches a local LangGraph Studio for interactive graph inspection and testing
- **LangSmith tracing** — when tracing is enabled, LangSmith uses this config to resolve the graph structure and visualize step-by-step execution
- **Graph validation** — `langgraph test` can validate the graph compiles and runs correctly

### LangSmith

The agent supports [LangSmith](https://smith.langchain.com/) tracing out of the box. Set these environment variables to enable per-step trace visualization, token counts, and latency breakdowns:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-langsmith-key>
export LANGCHAIN_PROJECT="cs-data-analyst"
```

These can also be placed in `.env` — this way all UI and CLI activity is automatically traced in LangSmith without exporting variables each time.

LangSmith is useful for diagnosing why the agent chose a particular tool, inspecting tool call arguments and results, and identifying where tokens are being consumed.

### Health checks

`src/health.py` validates three things at startup: dataset integrity (11 categories, 27 intents), API connectivity (both agent and router models respond), and persistence (SQLite is writable). Run diagnostics:

```bash
uv run python main.py --health
```

Or via the API: `GET /api/health`.

### Dataset fallback

The dataset CSV (~19 MB) is committed to the repo. If it's missing (e.g., after a shallow clone), `src/data.py` automatically downloads it from HuggingFace using the `datasets` library and caches it in `.cache/` (gitignored).

### Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'NEBIUS_API_KEY'` | Missing `.env` file | Copy `.env.example` to `.env` and set your API key |
| `SSLCertVerificationError` | Corporate proxy (Zscaler) | Place `combined_ca_bundle.pem` in project root; auto-detected |
| `sqlite3.OperationalError: database is locked` | Concurrent access to `checkpoints.db` | Normal under load — the `threading.Lock` handles this. If persistent, restart the server. |
| Agent returns raw XML instead of calling tools | DeepSeek DSML leak | The DSML repair in `agent.py` handles most cases. If persistent, switch `AGENT_MODEL` to Qwen. |
| Infinite tool call loop in CLI | Model repeating same tool call | Loop detection forces a text response after 2 identical calls. If it persists, the model may not be suitable (see rejected models). |
| `ModuleNotFoundError: langchain.agents.middleware` | `langchain-core` installed instead of full `langchain` | Install `langchain>=1.3.0` (the full package, not just `langchain-core`). |

## Requirements Mapping

See [docs/requirements_mapping.md](docs/requirements_mapping.md) for a detailed table mapping every assignment requirement to the corresponding implementation.

## Token Optimization

Following token optimization techniques are implemented in the project:

| Optimization Technique    | Description and expected savings                                                                                                                                                                                                                           | Implementation                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **TOON format**           | Multi-record tool outputs use field names once as a header, then pipe-delimited rows instead of repeating JSON keys per record. 30–60% fewer tokens for tabular results.                                                                                   | `src/toon.py` — `to_toon()` formatter, used by `get_examples` and `search_instructions` |
| **Dynamic tool exposure** | Instead of binding all 11 tools on every LLM call, the router classification selects a subset. The LLM never sees tool descriptions it won't need. ~40% fewer tool description tokens per call.                                                            | `src/tools.py` — `get_tools_for_query_type()`, driven by `src/router.py` classification |
| **Deterministic caching** | Tools with static results are cached — repeated calls return the same result without recomputation or re-sending to the LLM. Eliminates redundant tool invocations.                                                                                        | `@lru_cache` on `list_categories` and `list_intents` in `src/tools.py`                  |
| **Output trimming**       | Tool outputs exclude irrelevant columns and cap result size to prevent unbounded context consumption. `get_examples` returns only `instruction`, `intent`, `response` (excludes the empty `flags` column). `get_distribution` returns top 15 entries only. | `src/tools.py` — column selection in `get_examples`, `.head(15)` in `get_distribution`  |
| **Cheap router**          | Query classification uses a smaller, cheaper model (Qwen3-30B at $0.10/$0.30 per 1M tokens) instead of the full agent model (Qwen3-235B at $0.20/$0.60). Classification doesn't need 235B parameters. 3.3× cheaper per classification call.                | `src/router.py` uses `ROUTER_MODEL`, separate from `AGENT_MODEL`                        |
| **Message trimming**      | Conversation history is truncated to the last 30 messages instead of LLM-based summarization. Avoids an extra LLM call per turn. Trade-off: early context is lost in long conversations rather than being summarized.                                      | `src/agent.py` — `_trim_messages(max_messages=30)`                                      |

## SSL / Corporate Proxy (Zscaler case)

The project auto-detects Zscaler corporate proxy: if `combined_ca_bundle.pem` exists in the project root, it builds a custom SSL context with `VERIFY_X509_STRICT` cleared. If absent (grader's machine), standard system SSL is used automatically. No configuration needed.

## Models Observation

Open-source LLMs in agentic settings show a significant quality gap vs frontier models. We tried four approaches — only one combination worked:

| Approach | Result |
|---|---|
| Prompt tuning | No effect — both Qwen models ignored behavioral instructions after tool calls |
| Reflection alone | No effect — agent retried but repeated the same failing strategy |
| Judge scoring | Unreliable — false negatives on complex conversations (lost track of many tool results) |
| **Decomposition + Tool enrichment** | **Significant improvement** — agent explores 3–5 search strategies instead of 1, scores ~4.1 avg vs ~3.2 |

The trade-off: latency increases from ~15s to ~35s per query, and token usage roughly doubles due to more tool calls (which is the desired behavior).

See [docs/models-observation.md](docs/models-observation.md) for detailed analysis with examples, models evaluated, and further improvements roadmap.

## Future Improvements

This project can be considered as a POC towards a real project. If we would continue to a real project, the following points would be on the roadmap.

- **Streaming**: current SSE is step-level (events per tool call). True token-level streaming would improve perceived responsiveness.
- **Scalability**: replace `SqliteSaver` with `PostgresSaver` for concurrent access and horizontal scaling.
- **PostgreSQL consolidation**: replace SQLite (checkpoints), JSON (session store), and Qdrant in-process (mem0) with a single PostgreSQL instance using pgvector — unified deployment, backup, and concurrent access. This consolidates three separate storage backends into one, simplifying operations and enabling proper concurrent access.
- **Auto-retry on low quality**: the judge infrastructure is in place; wiring auto-retry when the score is below threshold would improve response quality automatically.
- **Multi-model fallback**: if the primary model is slow or unavailable, automatically switch to a cheaper fallback model.
- **Authentication**: add user authentication for multi-user deployments.
- **Deployment**: containerize with Docker for Oracle Cloud VPS, Nebius Serverless, or AWS Lambda deployment.
