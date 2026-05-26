# Models Observation

Lessons learned from building a ReAct agent with open-source LLMs on [Nebius Token Factory](https://tokenfactory.nebius.com/).

## The Findings

Open-source LLMs in agentic settings (multi-step tool calling, free-text query handling) demonstrate quite low quality in this project. They require explicit guidance to achieve comparable behavior. Further improvements such as decomposition, help to increase the quality while also increasing the request processing time and tokens demand.

## What We Tried

We tested four approaches to improve agent quality. Here's what worked and what didn't.

```
                    Quality Impact
                    ─────────────────────────────────────────
Prompt tuning       ░░░░░░░░░░░░░░░░░░░░  No measurable effect
Reflection step     ░░░░░░░░░░░░░░░░░░░░  No measurable effect alone
Judge scoring       ▓▓░░░░░░░░░░░░░░░░░░  Unreliable — not usable as feedback signal
Decomposition +     ████████████████░░░░  Significant improvement
Tool enrichment                           (with latency/cost trade-off)
```

### 1. Prompt Tuning — Did Not Help

We iteratively refined the agent system prompt for both Qwen3-235B and Qwen3-30B:

- Added instructions to "try synonyms when search returns empty"
- Added "explore related categories before giving up"
- Added "never conclude 'not found' without trying at least 3 approaches"

**Result**: No observable improvement. Both models ignored the behavioral guidance in the system prompt when their tool call returned empty results. The prompt text is processed once at the start; by the time the agent sees an empty result 3–4 turns later, the prompt instruction has faded from attention.

### 2. Agent Reflection — Did Not Help Alone

A post-response self-check using the cheap router model (Qwen3-30B) evaluates whether the agent's answer looks "lazy" (e.g., "I couldn't find data about X" after minimal effort) and nudges it to retry.

**Result**: In isolation, reflection did not improve quality. The agent would retry but repeat the same failing strategy — calling `search_instructions` with the same keyword that already returned nothing. Reflection without a *plan for what to do differently* just wastes tokens.

**However**: reflection becomes useful *in combination* with tool enrichment, because the enriched docstrings give the agent specific fallback strategies to follow on retry.

### 3. Judge Scoring — Unreliable

A cross-model judge (Llama-3.3-70B evaluating Qwen3-235B responses) scores each answer on three dimensions: `data_grounded`, `addresses_question`, `conciseness`.

**Result**: The judge is not reliable enough to use as a feedback signal.

**Example from the evaluation run** (user: "Explain categories, intents, and record counts"):

> The agent called `list_intents(category=X)` for all 11 categories — 15+ tool calls with correct results. The judge scored it **4/5** with the complaint: *"The analyst fabricated intent details for most categories. The tool output only provided intents for the 'ACCOUNT' category."*
>
> **The judge was wrong.** It lost track of 11 separate tool results and concluded the data was fabricated.

Another example (user: "How do you know 'card declined' is top 5?"):

> The agent searched for "card declined" and got zero results (literal substring match), but then claimed "I found 10 examples." **The judge correctly caught this** contradiction — score 3.3/5.

**Pattern**: The judge works on simple conversations (1–2 tool calls) but fails on complex multi-tool interactions where it can't correlate many tool results with the response. This makes it unreliable as an automated quality gate.

### 4. Decomposition + Tool Enrichment — The Winner

The combination of two features produced the only significant quality improvement:

**Tool definition enrichment** — WHAT/WHY/WHEN annotations in every tool docstring:

```
WHEN: User describes a topic in natural language
IMPORTANT: This is a LITERAL substring match, not semantic search.
FALLBACK: If multiple search terms return nothing, switch strategy:
call list_intents on related categories to find matching intent
names, then use get_examples on those intents.
```

**Query decomposition** — a planning step (using cheap Qwen3-30B) that generates a search strategy before the ReAct loop:

```
1. Search for: "payment issue", "payment failed", "card declined"
2. Check intents in PAYMENT: payment_issue, check_payment_methods
3. Pull examples from matching intents and count keyword frequency
```

**Example from the evaluation run** (user: "Yes, do it" — count payment issue types):

Without decomposition + enrichment, the agent would call `search_instructions("payment issues")`, get limited results, and say "I can't count specific issue types."

With both features enabled, the agent ran **10 search calls** with different keywords ("card declined", "payment failed", "cannot make payment", "transfer", "payment error", "card", "declined", "failed", etc.), presented a table of results, and honestly acknowledged the 20-result cap limitation. **Score: 4.7/5.**

## The Trade-Off

```
                  Without              With
                  decomposition        decomposition + enrichment
─────────────────────────────────────────────────────────────────
Avg latency       10–20s               30–40s
Avg tokens/query  ~15K                 ~25K
Tool calls/query  2–4                  8–15
Quality score     3.2 avg              4.1 avg
Search strategies 1 (give up)          3–5 (systematic)
```

Decomposition adds ~1 cheap LLM call ($0.10/1M tokens). The real cost increase comes from the agent making more tool calls — which is the desired behavior.

## Models Evaluated


| Model             | Role Tested        | Outcome                                                                 |
| ----------------- | ------------------ | ----------------------------------------------------------------------- |
| **Qwen3-235B**    | Agent              | Best available. Reliable tool calling, follows decomposition plans      |
| **Qwen3.5-397B**  | Agent              | Stronger but requires thinking mode disabled; higher cost               |
| **Qwen3-30B**     | Router, Decomposer | Cost-effective for classification and planning                          |
| **Llama-3.3-70B** | Agent, Judge       | Infinite loops as agent; unreliable as judge on complex conversations   |
| **DeepSeek-V3.2** | Agent              | DSML tool-call parsing bug on Nebius backend — unusable                 |
| **Kimi-K2.6**     | Agent              | Returns `content=None` (thinking mode that can't be disabled) — removed |


## Further Improvements Roadmap


| Improvement                            | Expected Impact                                                                                                                                                   | Effort |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| **Semantic search tool**               | High — would eliminate the "literal match" limitation that causes most failed searches. Replace `search_instructions` substring match with embedding-based search | Medium |
| **Smarter judge context**              | Medium — send only relevant tool results to the judge instead of the full trace, reducing context-window confusion                                                | Low    |
| **Few-shot examples in decomposition** | Medium — include 2–3 worked examples of good search strategies in the decomposition prompt                                                                        | Low    |
| **Evaluation pipeline**                | High — benchmark fixed test suite across models with automated scoring, replacing ad-hoc manual testing                                                           | Medium |
| **Frontier model fallback**            | High — use a frontier model (GPT-4o via API) for complex queries, open-source for simple ones                                                                     | Low    |
| **Fine-tuning on tool traces**         | High — fine-tune Qwen3-30B on successful multi-step tool-calling traces to internalize search strategies                                                          | High   |


