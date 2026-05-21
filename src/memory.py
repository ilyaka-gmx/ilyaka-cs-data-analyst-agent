"""
User profile management — persistent per-user facts stored as JSON.

The agent uses remember_fact / recall_profile tools to interact with
this module.  Profiles persist across sessions and restarts.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import PROFILES_DIR


class UserProfile(BaseModel):
    """Schema for a persistent user profile."""

    user_id: str
    facts: list[str] = Field(default_factory=list)
    last_updated: str = ""


def _profile_path(user_id: str) -> Path:
    return PROFILES_DIR / f"{user_id}.json"


def load_profile(user_id: str) -> UserProfile:
    """Load a user profile from disk, or return an empty profile."""
    path = _profile_path(user_id)
    if path.exists():
        data = json.loads(path.read_text())
        return UserProfile(**data)
    return UserProfile(user_id=user_id)


def save_profile(profile: UserProfile) -> None:
    """Save a user profile to disk."""
    profile.last_updated = datetime.now(timezone.utc).isoformat()
    path = _profile_path(profile.user_id)
    path.write_text(json.dumps(profile.model_dump(), indent=2))


def add_fact(user_id: str, fact: str) -> str:
    """Add a fact to the user's profile. Returns confirmation."""
    profile = load_profile(user_id)
    if fact.strip() not in profile.facts:
        profile.facts.append(fact.strip())
        save_profile(profile)
        return f"Remembered: {fact.strip()}"
    return f"Already known: {fact.strip()}"


def get_facts(user_id: str) -> str:
    """Retrieve all facts about a user. Returns formatted string."""
    profile = load_profile(user_id)
    if not profile.facts:
        return "No profile information stored yet."
    return "User profile:\n" + "\n".join(f"- {f}" for f in profile.facts)
