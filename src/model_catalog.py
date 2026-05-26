"""
Model catalog for the admin model selector.

Each entry contains metadata shown in the UI: strength rating,
cost per 1M tokens, pros/cons, and the full Nebius model ID.
"""

AGENT_MODELS = [
    {
        "id": "Qwen/Qwen3.5-397B-A17B",
        "short_name": "Qwen3.5-397B",
        "strength": 5,
        "cost_input": 0.60,
        "cost_output": 3.60,
        "pros": [
            "Best instruction following (IFBench 76.5)",
            "Top agentic benchmarks (BFCL v4: 72.9)",
            "Same MoE architecture, proven efficient",
        ],
        "cons": [
            "Thinking mode disabled (wastes tokens in tool-calling)",
        ],
    },
    {
        "id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "short_name": "Qwen3-235B",
        "strength": 4,
        "cost_input": 0.20,
        "cost_output": 3.60,
        "pros": [
            "Good reasoning, proven in our stack",
            "Cost-effective input pricing",
        ],
        "cons": [
            "Weaker instruction following",
            "Tends to give up on empty searches",
        ],
    },
    {
        "id": "zai-org/GLM-5.1",
        "short_name": "GLM-5.1",
        "strength": 4,
        "cost_input": 1.40,
        "cost_output": 4.80,
        "pros": [
            "Strong bilingual reasoning",
            "Agent-oriented design",
        ],
        "cons": [
            "Most expensive option",
            "Chinese-English focused",
        ],
    },
    {
        "id": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
        "short_name": "Nemotron-253B",
        "strength": 4,
        "cost_input": 0.60,
        "cost_output": 1.80,
        "pros": [
            "NVIDIA quality, strong reasoning",
            "Lowest output cost among large models",
        ],
        "cons": [
            "Dense 253B — higher latency than MoE models",
        ],
    },
]

JUDGE_MODELS = [
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "short_name": "Llama-3.3-70B",
        "strength": 3,
        "cost_input": 0.13,
        "cost_output": 0.40,
        "pros": [
            "Proven, cheap, reliable JSON output",
        ],
        "cons": [
            "Misses nuance in memory/recall scenarios",
        ],
    },
    {
        "id": "NousResearch/Hermes-4-70B",
        "short_name": "Hermes-4-70B",
        "strength": 4,
        "cost_input": 0.13,
        "cost_output": 0.48,
        "pros": [
            "High-quality reasoning",
            "Similar cost to Llama",
        ],
        "cons": [
            "May have different JSON formatting",
        ],
    },
    {
        "id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "short_name": "Qwen3-30B",
        "strength": 3,
        "cost_input": 0.20,
        "cost_output": 0.30,
        "pros": [
            "Very cheap, good for simple scoring",
        ],
        "cons": [
            "May be too small for nuanced evaluation",
        ],
    },
    {
        "id": "Qwen/Qwen3-32B",
        "short_name": "Qwen3-32B",
        "strength": 3,
        "cost_input": 0.10,
        "cost_output": 0.30,
        "pros": [
            "Cheapest option",
            "Decent quality for the price",
        ],
        "cons": [
            "May miss subtle grounding issues",
        ],
    },
    {
        "id": "NousResearch/Hermes-4-405B",
        "short_name": "Hermes-4-405B",
        "strength": 5,
        "cost_input": 0.60,
        "cost_output": 1.80,
        "pros": [
            "Excellent reasoning at 405B scale",
            "Best judge accuracy",
        ],
        "cons": [
            "Expensive for per-query judge calls",
        ],
    },
    {
        "id": "google/gemma-3-27b-it",
        "short_name": "Gemma-3-27B",
        "strength": 3,
        "cost_input": 0.10,
        "cost_output": 0.30,
        "pros": [
            "Google quality at low cost",
        ],
        "cons": [
            "Small model, may struggle with complex evaluations",
        ],
    },
]

AGENT_MODEL_IDS = {m["id"] for m in AGENT_MODELS}
JUDGE_MODEL_IDS = {m["id"] for m in JUDGE_MODELS}
