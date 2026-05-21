"""
Dataset loading and validation for the Bitext Customer Service dataset.

Loads the CSV into a pandas DataFrame and validates the expected schema.
Auto-downloads from HuggingFace if the CSV is not found locally.
Derives constants (CATEGORIES, INTENTS, CATEGORY_INTENT_MAP) and builds
DatasetMetadata for the agent's system prompt.
"""

import os
from dataclasses import dataclass, field

import pandas as pd

from src.config import DATA_DIR, DATASET_PATH, PROJECT_ROOT

EXPECTED_COLUMNS: list[str] = ["flags", "instruction", "category", "intent", "response"]
EXPECTED_NUM_CATEGORIES: int = 11
EXPECTED_NUM_INTENTS: int = 27

HUGGINGFACE_DATASET_ID: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def _auto_download() -> None:
    """Download the dataset from HuggingFace if not present locally.

    Redirects HF_HOME to a local .cache/ directory to avoid permission
    issues in sandboxed or restricted environments.
    """
    print(f"Dataset not found at {DATASET_PATH}. Downloading from HuggingFace...")
    try:
        local_cache = PROJECT_ROOT / ".cache" / "huggingface"
        local_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(local_cache))

        from datasets import load_dataset as hf_load

        ds = hf_load(HUGGINGFACE_DATASET_ID)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ds["train"].to_csv(str(DATASET_PATH), index=False)
        print(f"Downloaded and saved to {DATASET_PATH}")
    except ImportError:
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH} and 'datasets' package not installed. "
            f"Either place the CSV manually or: pip install datasets"
        )


def load_dataset() -> pd.DataFrame:
    """Load the Bitext dataset from CSV and validate schema.

    Auto-downloads from HuggingFace if the file is missing.

    Returns:
        pd.DataFrame: Validated dataset.

    Raises:
        FileNotFoundError: If download fails.
        ValueError: If the dataset schema doesn't match expectations.
    """
    if not DATASET_PATH.exists():
        _auto_download()

    df = pd.read_csv(DATASET_PATH)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")

    num_categories = df["category"].nunique()
    num_intents = df["intent"].nunique()

    print(f"Dataset loaded: {len(df)} rows, {num_categories} categories, {num_intents} intents")

    if num_categories != EXPECTED_NUM_CATEGORIES:
        print(f"  WARNING: Expected {EXPECTED_NUM_CATEGORIES} categories, got {num_categories}")

    if num_intents != EXPECTED_NUM_INTENTS:
        print(f"  WARNING: Expected {EXPECTED_NUM_INTENTS} intents, got {num_intents}")

    return df


dataset: pd.DataFrame = load_dataset()

# --- Derived constants — single source of truth for the loaded dataset ---

CATEGORIES: list[str] = sorted(dataset["category"].unique().tolist())
INTENTS: list[str] = sorted(dataset["intent"].unique().tolist())
CATEGORY_INTENT_MAP: dict[str, list[str]] = {
    cat: sorted(dataset[dataset["category"] == cat]["intent"].unique().tolist())
    for cat in CATEGORIES
}


@dataclass
class DatasetMetadata:
    """Validated metadata about the loaded dataset, used to generate
    dynamic system prompt context so the agent always has accurate
    dataset awareness even if the underlying data changes."""

    row_count: int
    num_categories: int
    num_intents: int
    categories: list[str]
    intents: list[str]
    category_intent_map: dict[str, list[str]]
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> "DatasetMetadata":
        if self.num_categories != EXPECTED_NUM_CATEGORIES:
            self.warnings.append(
                f"Expected {EXPECTED_NUM_CATEGORIES} categories, got {self.num_categories}"
            )
        if self.num_intents != EXPECTED_NUM_INTENTS:
            self.warnings.append(
                f"Expected {EXPECTED_NUM_INTENTS} intents, got {self.num_intents}"
            )
        return self

    def to_system_prompt_context(self) -> str:
        mapping_lines = []
        for cat, intents in self.category_intent_map.items():
            mapping_lines.append(f"  {cat}: {', '.join(intents)}")
        return (
            f"The dataset contains {self.row_count:,} customer service records "
            f"across {self.num_categories} categories and {self.num_intents} intents.\n"
            f"Categories and their intents:\n" + "\n".join(mapping_lines)
        )


def build_metadata(df: pd.DataFrame) -> DatasetMetadata:
    return DatasetMetadata(
        row_count=len(df),
        num_categories=df["category"].nunique(),
        num_intents=df["intent"].nunique(),
        categories=CATEGORIES,
        intents=INTENTS,
        category_intent_map=CATEGORY_INTENT_MAP,
    ).validate()


metadata: DatasetMetadata = build_metadata(dataset)

if metadata.warnings:
    for w in metadata.warnings:
        print(f"  WARNING: {w}")
