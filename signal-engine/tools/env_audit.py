from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any, get_args, get_origin


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "app" / "config.py"
ENV_EXAMPLE_PATH = ROOT / ".env.example"
ENV_TEMPLATE_PATHS = [ROOT / ".env.example", ROOT / ".env.example.crypto", ROOT / ".env.example.ftmo"]
DOC_PATH = ROOT / "docs" / "ENV_KEYS.md"


@dataclass
class StaticKeyRef:
    key: str
    file: str
    line: int
    kind: str


def _load_settings_class():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.config import Settings  # pylint: disable=import-outside-toplevel

    return Settings


def _format_default(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return value or '""'
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)


def _type_label(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is None:
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is str:
            return "str"
        return "str"
    if origin in (list, tuple, set):
        return "list"
    args = [a for a in get_args(annotation) if a is not type(None)]
    if len(args) == 1:
        return _type_label(args[0])
    labels = {_type_label(a) for a in args}
    if len(labels) == 1:
        return next(iter(labels))
    return "str"


def _settings_field_lines(config_text: str) -> dict[str, int]:
    lines: dict[str, int] = {}
    tree = ast.parse(config_text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    lines[item.target.id] = item.lineno
    return lines


def _parse_profile_defaults(config_text: str) -> dict[str, dict[str, Any]]:
    tree = ast.parse(config_text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_profile_defaults":
                    for stmt in fn.body:
                        if isinstance(stmt, ast.Assign):
                            if any(isinstance(t, ast.Name) and t.id == "profiles" for t in stmt.targets):
                                return ast.literal_eval(stmt.value)
    return {}


def _parse_legacy_keys(config_text: str) -> list[str]:
    tree = ast.parse(config_text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == "LEGACY_ENV_KEYS":
                    return list(ast.literal_eval(item.value))
    return []


def _collect_static_env_refs(root: Path) -> list[StaticKeyRef]:
    refs: list[StaticKeyRef] = []
    patterns = [
        (re.compile(r"os\.getenv\(\s*['\"]([A-Z0-9_]+)['\"]"), "os.getenv"),
        (re.compile(r"os\.environ\.get\(\s*['\"]([A-Z0-9_]+)['\"]"), "os.environ.get"),
        (re.compile(r"environ\.get\(\s*['\"]([A-Z0-9_]+)['\"]"), "environ.get"),
        (re.compile(r"os\.environ\[\s*['\"]([A-Z0-9_]+)['\"]\s*\]"), "os.environ[]"),
        (re.compile(r"load_dotenv\("), "dotenv"),
    ]
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("dashboard/"):
            continue
        text = path.read_text(encoding="utf-8-sig")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, kind in patterns:
                for match in pattern.finditer(line):
                    key = match.group(1) if match.groups() else "load_dotenv"
                    refs.append(StaticKeyRef(key=key, file=rel, line=lineno, kind=kind))
    return refs


def _collect_setting_usage(root: Path) -> dict[str, list[str]]:
    usage: dict[str, list[str]] = {}
    attr_re = re.compile(r"\b(?:cfg|settings)\.([a-zA-Z_][a-zA-Z0-9_]*)")
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in {"app/config.py", "tools/env_audit.py"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in attr_re.finditer(line):
                usage.setdefault(m.group(1), []).append(f"{rel}:{lineno}")
    return usage


def collect_env_audit(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    settings_cls = _load_settings_class()
    config_text = CONFIG_PATH.read_text(encoding="utf-8-sig")
    field_lines = _settings_field_lines(config_text)
    static_refs = _collect_static_env_refs(root)
    static_map: dict[str, list[str]] = {}
    for ref in static_refs:
        static_map.setdefault(ref.key, []).append(f"{ref.file}:{ref.line} ({ref.kind})")
    usage_map = _collect_setting_usage(root)
    profile_defaults = _parse_profile_defaults(config_text)
    legacy_forbidden = _parse_legacy_keys(config_text)

    canonical: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    canonical_key_set: set[str] = set()

    for field_name, field in settings_cls.model_fields.items():
        alias = field.validation_alias
        choices = []
        if alias is not None and hasattr(alias, "choices"):
            choices = [str(x) for x in alias.choices]
        if not choices:
            choices = [field_name.upper()]

        canonical_key = choices[0]
        canonical_key_set.add(canonical_key)
        default = field.default_factory() if field.default_factory else field.default
        locations = [f"app/config.py:{field_lines.get(field_name, 0)} (Settings.{field_name})"]
        locations.extend(static_map.get(canonical_key, []))
        note_parts: list[str] = []
        if field_name in usage_map:
            note_parts.append("active")
        else:
            note_parts.append("dead/legacy")

        canonical.append(
            {
                "key": canonical_key,
                "type": _type_label(field.annotation),
                "default": _format_default(default),
                "component": "Settings",
                "used_for": f"Config field `{field_name}`",
                "source_locations": sorted(set(locations)),
                "notes": ", ".join(note_parts),
                "active": field_name in usage_map,
                "field_name": field_name,
            }
        )

        for legacy_key in choices[1:]:
            status = "accepted"
            notes = f"Alias for {canonical_key}; precedence order: {' > '.join(choices)}"
            if legacy_key in legacy_forbidden:
                status = "forbidden"
                notes += "; rejected when SETTINGS_ENABLE_LEGACY=false"
            aliases.append(
                {
                    "legacy_key": legacy_key,
                    "maps_to": canonical_key,
                    "status": status,
                    "notes": notes,
                    "locations": sorted(set([f"app/config.py:{field_lines.get(field_name, 0)} (Settings.{field_name})"] + static_map.get(legacy_key, []))),
                }
            )

    unknown_runtime_keys: dict[str, dict[str, Any]] = {}
    for ref in static_refs:
        if ref.key in {"load_dotenv", *canonical_key_set}:
            continue
        if any(ref.key == item["legacy_key"] for item in aliases):
            continue
        bucket = unknown_runtime_keys.setdefault(
            ref.key,
            {
                "key": ref.key,
                "type": "str",
                "default": "None",
                "component": "Runtime",
                "used_for": "Direct environment lookup",
                "source_locations": [],
                "notes": "active",
                "active": True,
                "field_name": None,
            },
        )
        bucket["source_locations"].append(f"{ref.file}:{ref.line} ({ref.kind})")

    for item in unknown_runtime_keys.values():
        item["source_locations"] = sorted(set(item["source_locations"]))
        canonical.append(item)

    canonical.sort(key=lambda item: item["key"])
    aliases.sort(key=lambda item: item["legacy_key"])

    env_example_keys: set[str] = set()
    template_files: list[str] = []
    missing_template_files: list[str] = []
    for template_path in ENV_TEMPLATE_PATHS:
        if not template_path.exists():
            missing_template_files.append(str(template_path.relative_to(root)))
            continue
        template_files.append(str(template_path.relative_to(root)))
        for line in template_path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            env_example_keys.add(stripped.split("=", 1)[0].strip())

    used_keys = canonical_key_set | {a["legacy_key"] for a in aliases} | {r.key for r in static_refs if r.key != "load_dotenv"}
    docs_not_used = sorted(env_example_keys - used_keys)
    used_not_docs = sorted(used_keys - env_example_keys)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loading_order": [
            "BaseSettings reads .env file first (env_file=.env)",
            "Process environment variables override matching keys",
            "Field defaults apply when env keys are absent",
            "Profile defaults (_profile_defaults) fill None fields",
            "Runtime overrides in apply_mode_defaults adjust derived values",
        ],
        "canonical": canonical,
        "aliases": aliases,
        "profiles": profile_defaults,
        "forbidden_when_legacy_disabled": sorted(legacy_forbidden),
        "templates": {"loaded": template_files, "missing": missing_template_files},
        "drift": {
            "docs_not_used": docs_not_used,
            "used_not_documented": used_not_docs,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Environment Keys Audit")
    lines.append("")
    lines.append(f"Generated at: `{audit['generated_at']}`")
    lines.append("")
    lines.append("## How env is loaded")
    for step in audit["loading_order"]:
        lines.append(f"1. {step}")
    lines.append("")

    lines.append("## Table 1: Canonical Keys")
    lines.append("")
    lines.append("| Key | Type | Default | Component | Used For | Source Locations | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for item in audit["canonical"]:
        locs = "<br>".join(item["source_locations"])
        lines.append(
            f"| {item['key']} | {item['type']} | `{item['default']}` | {item['component']} | {item['used_for']} | {locs} | {item['notes']} |"
        )
    lines.append("")

    lines.append("## Table 2: Legacy/Alias Keys")
    lines.append("")
    lines.append("| Legacy Key | Maps To | Status (accepted/ignored/forbidden) | Notes | Locations |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in audit["aliases"]:
        locs = "<br>".join(item["locations"])
        lines.append(f"| {item['legacy_key']} | {item['maps_to']} | {item['status']} | {item['notes']} | {locs} |")
    lines.append("")

    lines.append("## Table 3: Profile-driven defaults")
    lines.append("")
    for profile, values in sorted(audit["profiles"].items()):
        lines.append(f"### {profile}")
        for key, value in sorted(values.items()):
            lines.append(f"- `{key}` = `{value}`")
        lines.append("")

    lines.append("## Table 4: Keys referenced in .env.example but not used (drift report)")
    lines.append("")
    for key in audit["drift"]["docs_not_used"]:
        lines.append(f"- {key}")
    if not audit["drift"]["docs_not_used"]:
        lines.append("- _None_")
    lines.append("")

    lines.append("## Table 5: Keys used in code but missing from .env.example (missing doc report)")
    lines.append("")
    for key in audit["drift"]["used_not_documented"]:
        lines.append(f"- {key}")
    if not audit["drift"]["used_not_documented"]:
        lines.append("- _None_")
    lines.append("")

    lines.append("## Forbidden when SETTINGS_ENABLE_LEGACY=false")
    for key in audit["forbidden_when_legacy_disabled"]:
        lines.append(f"- {key}")
    return "\n".join(lines) + "\n"


def regenerate_docs(root: Path | None = None) -> tuple[dict[str, Any], str]:
    audit = collect_env_audit(root)
    markdown = render_markdown(audit)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(markdown, encoding="utf-8")
    return audit, markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate env key audit docs")
    parser.add_argument("--check", action="store_true", help="Exit non-zero if drift is detected")
    args = parser.parse_args()

    audit, _ = regenerate_docs(ROOT)
    if args.check and (audit["drift"]["docs_not_used"] or audit["drift"]["used_not_documented"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
