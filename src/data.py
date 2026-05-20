"""
Dataset loading and validation for the Bitext Customer Service dataset.

Loads the CSV into a pandas DataFrame and validates the expected schema.
Auto-downloads from HuggingFace if the CSV is not found locally.
"""

import os

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
