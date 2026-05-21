"""
Startup and on-demand health checks for the agent system.

Validates dataset, API connectivity, persistence, and model availability
before accepting user queries.
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from src.config import (
    AGENT_MODEL,
    CHECKPOINTS_DB,
    PROFILES_DIR,
    ROUTER_MODEL,
    get_llm,
)
from src.data import metadata


@dataclass
class CheckResult:
    name: str
    status: Literal["pass", "warn", "fail"]
    message: str
    duration_ms: int = 0


@dataclass
class HealthReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warn" for c in self.checks)

    def summary(self) -> str:
        lines = ["Health Check Report", "=" * 40]
        for c in self.checks:
            icon = {"pass": "+", "warn": "!", "fail": "x"}[c.status]
            lines.append(f"  [{icon}] {c.name}: {c.message} ({c.duration_ms}ms)")
        passed = sum(1 for c in self.checks if c.status == "pass")
        total = len(self.checks)
        lines.append(f"\n{passed}/{total} checks passed")
        return "\n".join(lines)


def check_dataset() -> CheckResult:
    """Validate dataset is loaded with expected schema."""
    start = time.time()
    try:
        if metadata.warnings:
            return CheckResult(
                "Dataset",
                "warn",
                f"Loaded with warnings: {metadata.warnings}",
                int((time.time() - start) * 1000),
            )
        return CheckResult(
            "Dataset",
            "pass",
            f"{metadata.row_count:,} rows, {metadata.num_categories} categories, "
            f"{metadata.num_intents} intents",
            int((time.time() - start) * 1000),
        )
    except Exception as e:
        return CheckResult("Dataset", "fail", str(e), int((time.time() - start) * 1000))


def check_api_connectivity(model: str, role: str) -> CheckResult:
    """Ping a model with a minimal request."""
    start = time.time()
    try:
        llm = get_llm(model, max_tokens=5, temperature=0)
        llm.invoke("Reply with ok.")
        duration = int((time.time() - start) * 1000)
        return CheckResult(f"API ({role})", "pass", f"{model} responded in {duration}ms", duration)
    except Exception as e:
        duration = int((time.time() - start) * 1000)
        return CheckResult(f"API ({role})", "fail", f"{model}: {e}", duration)


def check_persistence() -> CheckResult:
    """Check that SQLite checkpoint path and profiles dir are writable."""
    start = time.time()
    issues = []

    if not PROFILES_DIR.exists():
        issues.append(f"Profiles dir missing: {PROFILES_DIR}")
    elif not PROFILES_DIR.is_dir():
        issues.append(f"Profiles path is not a directory: {PROFILES_DIR}")

    try:
        test_file = CHECKPOINTS_DB.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        issues.append(f"Cannot write to checkpoint dir: {e}")

    duration = int((time.time() - start) * 1000)
    if issues:
        return CheckResult("Persistence", "warn", "; ".join(issues), duration)
    return CheckResult("Persistence", "pass", "SQLite + profiles writable", duration)


def run_startup_checks() -> HealthReport:
    """Run all startup health checks. Called before accepting queries."""
    report = HealthReport()
    report.add(check_dataset())
    report.add(check_api_connectivity(AGENT_MODEL, "agent"))
    report.add(check_api_connectivity(ROUTER_MODEL, "router"))
    report.add(check_persistence())
    return report


def run_diagnostics() -> HealthReport:
    """Run full diagnostics including optional checks."""
    report = run_startup_checks()

    if CHECKPOINTS_DB.exists():
        size_mb = CHECKPOINTS_DB.stat().st_size / (1024 * 1024)
        report.add(CheckResult("Checkpoint DB", "pass", f"{size_mb:.1f} MB", 0))
    else:
        report.add(CheckResult("Checkpoint DB", "pass", "Not yet created (first run)", 0))

    profiles = list(PROFILES_DIR.glob("*.json"))
    report.add(CheckResult("Profiles", "pass", f"{len(profiles)} user profiles on disk", 0))

    return report
