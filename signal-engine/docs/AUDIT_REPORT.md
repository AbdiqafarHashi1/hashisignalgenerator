# Audit Report

## Strategy entrypoints
- `app/strategy/decision.py:decide` is the canonical signal/entry/exit decision path.
- `app/services/scheduler.py:_evaluate_symbol` calls decision pipeline and routes execution through paper trader.
- `app/services/paper_trader.py` handles execution/fills and stop/tp lifecycle.

## Accounting / metrics computation points
- `app/services/metrics.py:compute_metrics` (canonical).
- `app/services/dashboard_metrics.py:build_dashboard_metrics` (compat aliases only).
- `app/main.py:/dashboard/overview` and `/dashboard/metrics` consume canonical metrics payload.

## Governor/blocker codes and triggers
- Governor evaluation: `app/services/prop_governor.py:block_reason` + `signal_engine/governor/prop.py:evaluate_prop_block`.
- Runtime blocker aggregation: `app/services/scheduler.py` (terminal/governor/risk/strategy blockers).
- State risk blockers: `app/state.py:check_limits` (daily dd, global dd, consecutive losses, cooldown).

## Environment keys read
- Full active key list generated in `docs/ENV_USED_KEYS.md`.
- Source of truth: `app/config.py` Settings fields + direct `os.environ` lookups in config/main.

## Duplicate/overlap findings
- Legacy profile names (`PROP_PASS`, `INSTANT_FUNDED`, `CRYPTO_SCALP_25`, etc.) overlap risk templates; consolidated runtime to one strategy profile (`CORE_STRATEGY`) plus `RISK_MODE`.
- Metrics were duplicated between dashboard endpoint formatting and accounting helpers; consolidated through `compute_metrics` and alias projection.
- Fee logic had crypto-specific assumptions inside paper trader; extracted to unified cost model with crypto+forex implementations.

## Recommended cleanup (safe deletions next pass)
- Remove unused legacy profile branches from `_profile_defaults` after env migration window.
- Remove deprecated legacy strategy names from docs/templates once compatibility period ends.
- Keep endpoint field aliases until dashboard frontend no longer requests legacy names.
