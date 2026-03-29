from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SaverEnvironmentState:
    visited_windows: List[Dict[str, Any]] = field(default_factory=list)
    evidence_ledger: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    verification_records: List[Dict[str, Any]] = field(default_factory=list)
    finalized_case: Optional[Dict[str, Any]] = None
    last_claim: Optional[Dict[str, Any]] = None
    active_evidence_window_ids: List[str] = field(default_factory=list)
    verifier_cache: List[Dict[str, Any]] = field(default_factory=list)
    next_evidence_id: int = 1
    next_window_id: int = 1
    next_alert_id: int = 1


def validate_required_fields(payload: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> None:
    if not schema:
        return
    required = schema.get("required", [])
    missing = [field_name for field_name in required if field_name not in payload]
    if missing:
        raise ValueError(f"Missing required finalize_case fields: {missing}")
