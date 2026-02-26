import { DashboardOverview } from "./api";

// Compatibility normalizer for evolving /dashboard/overview payloads.
// The trade table can keep updating from paginated endpoints even when KPI tiles break
// if overview keys move between flat and nested contracts.
export type NormalizedOverview = {
  equity_now: number;
  equity_start: number;
  realized_pnl_net: number;
  realized_pnl_gross: number;
  unrealized_pnl: number;
  fees_today: number;
  fees_total: number;
  daily_dd_pct: number;
  global_dd_pct: number;
  trades_today: number;
  status: string | null;
  reason: string | null;
  blocker_code: string | null;
  blocker_reason: string | null;
  run_mode: "replay" | "live";
  heartbeat_ts: string | null;
  replay_ts: string | null;
};

const asObject = (value: unknown): Record<string, unknown> => {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
};

const asNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const asString = (value: unknown): string | null => {
  if (typeof value === "string" && value.trim().length > 0) return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return null;
};

const lower = (value: unknown): string | null => {
  const text = asString(value);
  return text ? text.toLowerCase() : null;
};

const pickNumber = (obj: Record<string, unknown>, root: Record<string, unknown>, ...keys: string[]): number => {
  for (const key of keys) {
    const nested = asNumber(obj[key]);
    if (nested != null) return nested;
    const flat = asNumber(root[key]);
    if (flat != null) return flat;
  }
  return 0;
};

const pickString = (
  primary: Record<string, unknown>,
  fallback: Record<string, unknown>,
  root: Record<string, unknown>,
  ...keys: string[]
): string | null => {
  for (const key of keys) {
    const first = asString(primary[key]);
    if (first) return first;
    const second = asString(fallback[key]);
    if (second) return second;
    const third = asString(root[key]);
    if (third) return third;
  }
  return null;
};

export function normalizeOverview(raw: DashboardOverview | Record<string, unknown>): NormalizedOverview {
  const root = asObject(raw);
  const account = asObject(root.account);
  const challenge = asObject(root.challenge);
  const governor = asObject(root.governor);
  const meta = asObject(root.meta);
  const settings = asObject(root.settings);
  const runtime = asObject(root.runtime);

  const modeRaw =
    lower(settings.run_mode) ??
    lower(meta.run_mode) ??
    lower(meta.mode) ??
    lower(account.run_mode) ??
    lower(root.run_mode) ??
    lower(runtime.run_mode);

  const run_mode: "replay" | "live" = modeRaw === "replay" ? "replay" : "live";

  return {
    equity_now: pickNumber(account, root, "equity_now", "live_equity", "equity", "equity_now_usd"),
    equity_start: pickNumber(account, root, "equity_start", "starting_equity", "equity_start_usd"),
    realized_pnl_net: pickNumber(account, root, "realized_net_usd", "realized_pnl_net", "realized_pnl"),
    realized_pnl_gross: pickNumber(account, root, "realized_gross_usd", "realized_pnl_gross"),
    unrealized_pnl: pickNumber(account, root, "unrealized_pnl", "unrealized_pnl_usd", "unrealized"),
    fees_today: pickNumber(account, root, "fees_today"),
    fees_total: pickNumber(account, root, "fees_total_usd", "fees_total"),
    daily_dd_pct: pickNumber(account, root, "daily_drawdown_pct", "daily_dd_pct"),
    global_dd_pct: pickNumber(account, root, "global_drawdown_pct", "global_dd_pct"),
    trades_today: pickNumber(account, root, "trades_today"),
    status: pickString(challenge, account, root, "status", "challenge_status"),
    reason: pickString(challenge, account, root, "status_reason", "challenge_status_reason", "reason"),
    blocker_code: pickString(governor, account, root, "blocker_code", "blocker_name", "governor_blocker_code"),
    blocker_reason: pickString(governor, account, root, "blocker_reason", "blocker_detail", "governor_blocker_reason"),
    run_mode,
    heartbeat_ts: pickString(meta, account, root, "now_ts", "last_tick_time"),
    replay_ts: pickString(meta, account, root, "replay_ts", "replay_clock"),
  };
}

export function validateNormalizeOverviewRuntime(): void {
  if (process.env.NODE_ENV === "production") return;
  const nested = normalizeOverview({ account: { equity_now: 25100, fees_total_usd: 12 }, challenge: { status_reason: "ok" } });
  const flat = normalizeOverview({ equity_now: 25200, trades_today: 3 });
  const missing = normalizeOverview({});
  if (nested.equity_now !== 25100 || flat.equity_now !== 25200 || missing.equity_now !== 0) {
    throw new Error("normalizeOverview runtime assertion failed");
  }
}
