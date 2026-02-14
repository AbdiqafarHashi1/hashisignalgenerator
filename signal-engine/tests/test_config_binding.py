from pathlib import Path


def test_no_magic_regime_entry_zone_literal() -> None:
    content = Path("app/strategy/decision.py").read_text()
    assert "0.9998" not in content
    assert "1.0002" not in content
