from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings


def _log_path(cfg: Settings, when: datetime) -> Path:
    date_key = when.date().isoformat()
    return Path(cfg.data_dir) / "logs" / f"{date_key}.jsonl"


def log_event(cfg: Settings, event_type: str, payload: dict[str, Any], correlation_id: str) -> None:
    now = datetime.now(timezone.utc)
    path = _log_path(cfg, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": now.isoformat(),
        "event_type": event_type,
        "correlation_id": correlation_id,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
