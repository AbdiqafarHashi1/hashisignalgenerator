export type RiskTone = "neutral" | "amber" | "red";

const STATUS_REASON_LABELS: Record<string, string> = {
  none: "No active reason",
  prop_time_cooldown: "Prop cooldown active",
  prop_daily_stop_after_net_r: "Daily stop: net R reached",
  prop_daily_stop_after_losses: "Daily stop: max losses",
  funding_blackout_entries_blocked: "Funding blackout (entries blocked)",
  cooldown_active: "Cooldown active",
  funding_blackout: "Funding blackout",
  max_trades_reached: "Max trades reached",
  global_drawdown_limit: "Global drawdown limit reached",
  daily_drawdown_limit: "Daily drawdown limit reached",
  RUNNING: "Challenge running",
  STOPPED_DAILY_TARGET: "Stopped: daily target reached",
  STOPPED_COOLDOWN: "Stopped: cooldown active",
  FAILED_DRAWDOWN: "Failed: global drawdown breached",
  FAILED_DAILY: "Failed: daily loss breached",
  PASSED: "Passed",
};

const BLOCKER_LABELS: Record<string, string> = {
  none: "No active blocker",
  prop_time_cooldown: "Prop cooldown",
  funding_blackout_entries_blocked: "Funding blackout",
  max_trades_reached: "Trade cap reached",
  cooldown_active: "Cooldown",
};

export function getRiskTone(ddPct: number | null, limitPct: number | null): RiskTone {
  if (ddPct == null || limitPct == null || limitPct <= 0) return "neutral";
  const ratio = ddPct / limitPct;
  if (ratio >= 0.8) return "red";
  if (ratio >= 0.5) return "amber";
  return "neutral";
}

export function getRiskHeat(dailyUsedPct: number | null, dailyLimitPct: number | null, globalUsedPct: number | null, globalLimitPct: number | null): number | null {
  const daily = dailyUsedPct != null && dailyLimitPct != null && dailyLimitPct > 0 ? dailyUsedPct / dailyLimitPct : null;
  const global = globalUsedPct != null && globalLimitPct != null && globalLimitPct > 0 ? globalUsedPct / globalLimitPct : null;
  if (daily == null && global == null) return null;
  return Math.max(daily ?? 0, global ?? 0);
}

export function formatStatusReason(reason: string): string {
  if (!reason) return "No active reason";
  if (STATUS_REASON_LABELS[reason]) return STATUS_REASON_LABELS[reason];
  return toTitleCase(reason.replaceAll("_", " "));
}

export function formatBlocker(code: string): string {
  if (!code) return "No active blocker";
  if (BLOCKER_LABELS[code]) return BLOCKER_LABELS[code];
  return toTitleCase(code.replaceAll("_", " "));
}

function toTitleCase(value: string): string {
  return value
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1).toLowerCase()}`)
    .join(" ");
}

export function toneClasses(tone: RiskTone): string {
  if (tone === "red") return "border-rose-500/65";
  if (tone === "amber") return "border-amber-400/60";
  return "border-slate-700/45";
}
