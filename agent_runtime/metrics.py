from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_RECORD_BYTES = 16_384
MAX_FILE_BYTES = 5 * 1024 * 1024
_ALLOWED_NUMBERS = {
    "duration_ms", "sample_count", "tool_round_count", "tool_count", "input_tokens_estimate",
    "persistence_ms", "snapshot_ms", "prompt_tokens_estimate", "actual_prompt_tokens",
    "actual_completion_tokens", "actual_cached_tokens", "actual_reasoning_tokens",
    "attempt", "attempt_count", "first_chunk_ms", "first_text_ms", "last_chunk_ms",
    "completion_ms", "backoff_ms", "catalog_ms", "prompt_ms", "estimate_ms", "compaction_ms",
    "system_tokens_estimate", "history_tokens_estimate", "current_tokens_estimate",
    "tool_tokens_estimate", "estimated_tokens_before", "estimated_tokens_after",
    "delta_count", "text_bytes",
}
_ALLOWED_LABELS = {"kind", "turn_id", "sample_id", "status", "provider", "model", "outcome", "finish_reason", "error_class", "tool_id", "tool_name", "reason", "mode"}
_warned = False


def _safe_record(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        if key in _ALLOWED_NUMBERS and value is None:
            safe[key] = None
        elif key in _ALLOWED_NUMBERS and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            safe[key] = value
        elif key in _ALLOWED_LABELS and isinstance(value, str) and len(value) <= 128 and all(c.isalnum() or c in "._:/-" for c in value):
            safe[key] = value
    return safe


def append_runtime_metric(record: dict[str, Any], workspace_root: str | Path = ".") -> None:
    """Append bounded, allowlisted diagnostics; failures never affect a turn."""
    global _warned
    try:
        safe = _safe_record(record)
        safe["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
        payload = (json.dumps(safe, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_RECORD_BYTES:
            return
        path = Path(workspace_root) / ".vimax" / "logs" / "agent_performance.runtime.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Separate runtime/bridge files give each rotation a single process owner.
        if path.exists() and path.stat().st_size + len(payload) > MAX_FILE_BYTES:
            path.replace(path.with_suffix(".jsonl.1"))
        with path.open("ab") as handle:
            handle.write(payload)
    except Exception as exc:
        if not _warned:
            _warned = True
            logging.warning("Runtime metrics write failed (%s)", type(exc).__name__)
