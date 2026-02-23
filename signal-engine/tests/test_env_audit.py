from __future__ import annotations

from pathlib import Path

from tools.env_audit import collect_env_audit, render_markdown


def test_env_audit_generation_metadata_only() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = collect_env_audit(root)
    markdown = render_markdown(audit)

    assert "RUN_MODE" in {entry["key"] for entry in audit["canonical"]}
    keys = {entry["key"] for entry in audit["canonical"]}
    alias_keys = {entry["legacy_key"] for entry in audit["aliases"]}
    assert "REPLAY_SPEED" in keys or "REPLAY_SPEED" in alias_keys

    assert "DATABASE_URL=sqlite:///" not in markdown
    assert "sqlite:///data/trades.db" not in markdown
    assert "| DATABASE_URL |" in markdown
