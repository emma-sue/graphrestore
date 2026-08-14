"""Structured audit results with deterministic JSON and Markdown rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .io import utc_now_iso

AuditStatus = Literal["PASS", "WARN", "FAIL"]


@dataclass(frozen=True)
class AuditCheck:
    """One independently interpretable audit assertion."""

    name: str
    status: AuditStatus
    detail: str


class AuditTrail:
    """Collect hard failures and warnings without losing later diagnostics."""

    def __init__(self, *, protocol: str) -> None:
        self.protocol = protocol
        self.created_utc = utc_now_iso()
        self.checks: list[AuditCheck] = []
        self.facts: dict[str, Any] = {}

    def record(self, name: str, status: AuditStatus, detail: str) -> None:
        if status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"invalid audit status: {status}")
        self.checks.append(AuditCheck(name=name, status=status, detail=detail))

    def require(self, condition: bool, name: str, success: str, failure: str) -> bool:
        self.record(name, "PASS" if condition else "FAIL", success if condition else failure)
        return condition

    def warn(self, condition: bool, name: str, success: str, warning: str) -> bool:
        self.record(name, "PASS" if condition else "WARN", success if condition else warning)
        return condition

    @property
    def failure_count(self) -> int:
        return sum(check.status == "FAIL" for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(check.status == "WARN" for check in self.checks)

    @property
    def passed(self) -> bool:
        return self.failure_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "protocol": self.protocol,
            "created_utc": self.created_utc,
            "passed": self.passed,
            "failure_count": self.failure_count,
            "warning_count": self.warning_count,
            "checks": [asdict(check) for check in self.checks],
            "facts": self.facts,
        }

    def to_markdown(self, *, title: str) -> str:
        result = "PASS" if self.passed else "FAIL"
        lines = [
            f"# {title}",
            "",
            f"- protocol: `{self.protocol}`",
            f"- created_utc: `{self.created_utc}`",
            f"- result: **{result}**",
            f"- failures: `{self.failure_count}`",
            f"- warnings: `{self.warning_count}`",
            "",
            "## Checks",
            "",
        ]
        for check in self.checks:
            lines.append(f"- **{check.status}** `{check.name}` — {check.detail}")
        lines.extend(
            [
                "",
                "## Machine-readable facts",
                "",
                "```json",
            ]
        )
        import json

        lines.append(json.dumps(self.facts, ensure_ascii=False, indent=2, sort_keys=True))
        lines.extend(["```", ""])
        return "\n".join(lines)
