import { apiFetch, DashboardOverview, DebugConfigResponse, EngineState } from "./api";

export type NormalizedTrade = {
  id: string;
  symbol: string;
  side: string;
  entry: number | null;
  exit: number | null;
  size: number | null;
  fee: number | null;
  pnlUsd: number | null;
  result: "win" | "loss" | "breakeven" | "unknown";
  resultLabel: string;
  opened: string;
  closed: string;
  reason: string;
  tp: number | null;
  sl: number | null;
  rMultiple: number | null;
  feeBreakdown: string;
  trigger: string;
  skipReason: string;
  mode: string;
  raw: Record<string, unknown>;
};

export type NormalizedSymbol = {
  symbol: string;
  regime: string;
  regimeLabel: string;
  lastDecision: string;
  lastSkipReason: string;
  atrPct: number | null;
  trendStrength: number | null;
  signalScore: number | null;
  unrealizedPnl: number | null;
  openPosition: string;
};

export type DashboardBundle = {
  account: Record<string, unknown>;
  risk: Record<string, number>;
  activity: Record<string, number>;
  symbols: NormalizedSymbol[];
  equitySeries: number[];
  skipGlobal: Record<string, number>;
  trades: NormalizedTrade[];
  debugConfig: DebugConfigResponse;
};

const DEFAULT_DEBUG: DebugConfigResponse = {
  effective: {},
  sources: {},
  env_keys: {},
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

const asString = (value: unknown, fallback = "--"): string => {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
};

const toNumericRecord = (input: unknown): Record<string, number> => {
  const obj = asObject(input);
  return Object.entries(obj).reduce<Record<string, number>>((acc, [key, value]) => {
    const parsed = asNumber(value);
    if (parsed !== null) acc[key] = parsed;
    return acc;
  }, {});
};

const tradeResult = (pnl: number | null): NormalizedTrade["result"] => {
  if (pnl == null) return "unknown";
  if (pnl > 0) return "win";
  if (pnl < 0) return "loss";
  return "breakeven";
};

const normalizeTrades = (input: unknown): NormalizedTrade[] => {
  if (!Array.isArray(input)) return [];
  return input.map((item, index) => {
    const row = asObject(item);
    const fee = asNumber(row.fee ?? row.fees ?? row.commission);
    const pnlUsd = asNumber(row.pnl_usd ?? row.pnl ?? row.realized_pnl);
    const rawResult = asString(row.result ?? row.close_reason ?? row.exit_reason, "unknown");
    return {
      id: asString(row.trade_id ?? row.id ?? `${index}`),
      symbol: asString(row.symbol),
      side: asString(row.side),
      entry: asNumber(row.entry_price ?? row.open_price ?? row.avg_entry ?? row.entry),
      exit: asNumber(row.exit_price ?? row.close_price ?? row.avg_exit ?? row.exit),
      size: asNumber(row.size ?? row.quantity),
      fee,
      pnlUsd,
      result: tradeResult(pnlUsd),
      resultLabel: rawResult,
      opened: asString(row.entry_time ?? row.opened_at ?? row.opened, "--"),
      closed: asString(row.exit_time ?? row.closed_at ?? row.closed ?? row.timestamp, "--"),
      reason: asString(row.reason),
      tp: asNumber(row.tp ?? row.take_profit ?? row.tp_price),
      sl: asNumber(row.sl ?? row.stop ?? row.stop_loss ?? row.stop_price ?? row.sl_price),
      rMultiple: asNumber(row.r_multiple),
      feeBreakdown: asString(row.fee_breakdown ?? row.fees_detail, "--"),
      trigger: asString(row.trigger ?? row.entry_trigger, "--"),
      skipReason: asString(row.skip_reason, "--"),
      mode: asString(row.trade_mode ?? row.mode, "paper"),
      raw: row,
    };
  });
};

const normalizeSymbols = (input: unknown): NormalizedSymbol[] => {
  const symbols = asObject(input);
  return Object.entries(symbols).map(([symbol, value]) => {
    const data = asObject(value);
    const openPosition = asObject(data.open_position);
    const openPositionSummary = Object.keys(openPosition).length
      ? `${asString(openPosition.side, "?")} ${asString(openPosition.size, "?")} @ ${asString(openPosition.entry, "?")}`
      : "No open position";

    return {
      symbol,
      regime: asString(data.regime),
      regimeLabel: asString(data.regime_label ?? data.regime),
      lastDecision: asString(data.last_decision),
      lastSkipReason: asString(data.last_skip_reason),
      atrPct: asNumber(data.atr_pct),
      trendStrength: asNumber(data.trend_strength),
      signalScore: asNumber(data.signal_score),
      unrealizedPnl: asNumber(data.unrealized_pnl_usd ?? data.unrealized_pnl),
      openPosition: openPositionSummary,
    };
  });
};

const normalizeOverview = (payload: DashboardOverview): Omit<DashboardBundle, "debugConfig"> => {
  const account = asObject(payload.account);
  const risk = toNumericRecord(payload.risk);
  const activity = toNumericRecord(payload.activity);
  const symbols = normalizeSymbols(payload.symbols);
  const trades = normalizeTrades(payload.recent_trades);
  const equitySeries = Array.isArray(payload.equity_curve)
    ? payload.equity_curve
        .map((point) => asNumber(asObject(point).equity))
        .filter((value): value is number => value !== null)
    : [];
  const skipGlobal = toNumericRecord(asObject(payload.skip_reasons).global);

  return { account, risk, activity, symbols, trades, equitySeries, skipGlobal };
};

export async function fetchDashboardBundle(): Promise<DashboardBundle> {
  const [overview, debugConfig] = await Promise.all([
    apiFetch<DashboardOverview>("/dashboard/overview"),
    apiFetch<DebugConfigResponse>("/debug/config").catch(() => DEFAULT_DEBUG),
  ]);

  return { ...normalizeOverview(overview), debugConfig };
}

export async function fetchEngineStateSafe(): Promise<EngineState | null> {
  try {
    return await apiFetch<EngineState>("/state");
  } catch {
    return null;
  }
}

export async function triggerEngineAction(path: string): Promise<string> {
  const response = await apiFetch<{ status?: string; detail?: string }>(path);
  return response.status ?? response.detail ?? "ok";
}
