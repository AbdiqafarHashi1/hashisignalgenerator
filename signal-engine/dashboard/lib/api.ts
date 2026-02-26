const DEFAULT_DOCKER_API_BASE = "http://api:8000";
const DEFAULT_DEV_API_BASE = "http://localhost:8000";

const resolveApiBase = (): string => {
  const envBase = process.env.NEXT_PUBLIC_API_BASE ?? "";
  if (envBase) {
    return envBase;
  }

  if (typeof window === "undefined") {
    return process.env.INTERNAL_API_BASE_URL ?? DEFAULT_DOCKER_API_BASE;
  }

  const isLocalhost =
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  return isLocalhost ? DEFAULT_DEV_API_BASE : "https://hashibot-api.fly.dev";
};

export const API_BASE = resolveApiBase();

export type ApiError = {
  status: number;
  message: string;
  url: string;
};

export type EngineStatus = {
  running?: boolean;
  status: string;
  mode?: string | null;
  symbols?: string[];
  last_heartbeat_ts?: string | null;
  last_action?:
    | {
        type?: string;
        ts?: string | null;
        detail?: string;
      }
    | string
    | null;
  last_tick_ts?: number | null;
  uptime_seconds?: number;
};


export type EngineBlocker = {
  code: string;
  layer: "terminal" | "governor" | "risk" | "strategy" | "none";
  detail: string;
  until_ts?: string | null;
};

export type EngineState = {
  blocker_code?: string | null;
  blocker_detail?: string | null;
  blocker_layer?: "terminal" | "governor" | "risk" | "strategy" | "none";
  blocker_until_ts?: string | null;
  blockers?: EngineBlocker[];

  timestamp: string;
  server_ts?: string | null;
  candle_ts?: string | null;
  replay_cursor_ts?: string | null;
  last_tick_age_seconds: number | null;
  running: boolean;
  balance: number;
  equity: number;
  unrealized_pnl_usd: number;
  realized_pnl_today_usd: number;
  trades_today: number;
  wins: number;
  losses: number;
  win_rate: number;
  profit_factor: number;
  max_dd_today_pct: number;
  daily_loss_remaining_usd: number;
  daily_loss_pct: number;
  open_positions: Array<Record<string, unknown>>;
  recent_trades: Array<Record<string, unknown>>;
  cooldown_active: boolean;
  funding_blackout: boolean;
  swings_enabled: boolean;
  current_mode: string;
  consecutive_losses?: number;
  last_decision?: string | null;
  last_skip_reason?: string | null;
  final_entry_gate?: string | null;
  regime_label?: string | null;
  allowed_side?: string | null;
  atr_pct?: number | null;
  ema_fast?: number | null;
  ema_slow?: number | null;
  ema_trend?: number | null;
};

export type PerformanceTrade = {
  trade_id: number;
  symbol: string;
  entry_time: string;
  exit_time: string;
  side: string;
  entry_price: number;
  exit_price: number;
  pnl_usd: number;
  pnl_pct: number;
  r_multiple: number;
  hold_seconds: number;
  reason: string;
};

export type PerformanceMetrics = {
  generated_at: string;
  trades: PerformanceTrade[];
  trades_today: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  expectancy_r: number;
  profit_factor: number;
  max_drawdown_pct: number;
  consecutive_wins: number;
  consecutive_losses: number;
  avg_hold_time: number;
  sharpe_like: number;
  skip_reason_counts: Record<string, number>;
  equity_curve: number[];
  drawdown_curve_pct: number[];
  win_loss_distribution: Record<string, number>;
};

export type DashboardOverview = {
  account: Record<string, unknown>;
  challenge?: Record<string, unknown>;
  governor?: Record<string, unknown>;
  meta?: Record<string, unknown>;
  risk: Record<string, unknown>;
  activity: Record<string, unknown>;
  symbols: Record<string, Record<string, unknown>>;
  recent_trades: Array<Record<string, unknown>>;
  equity_curve: Array<{ index: number; equity: number }>;
  skip_reasons: Record<string, unknown>;
};

export type PaginatedResponse<T> = {
  items: T[];
  limit: number;
  offset: number;
  total?: number | null;
};

export const stateEventsUrl = `${API_BASE}/events/state`;


let lastKnownOverviewRunMode: "replay" | "live" = "live";

const inferRunModeFromOverview = (payload: unknown): "replay" | "live" => {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return lastKnownOverviewRunMode;
  const root = payload as Record<string, unknown>;
  const meta = root.meta && typeof root.meta === "object" && !Array.isArray(root.meta) ? (root.meta as Record<string, unknown>) : {};
  const settings = root.settings && typeof root.settings === "object" && !Array.isArray(root.settings) ? (root.settings as Record<string, unknown>) : {};
  const account = root.account && typeof root.account === "object" && !Array.isArray(root.account) ? (root.account as Record<string, unknown>) : {};
  const values = [settings.run_mode, meta.run_mode, meta.mode, account.run_mode, root.run_mode];
  for (const value of values) {
    if (typeof value === "string" && value.toLowerCase() === "replay") return "replay";
  }
  return "live";
};

const timeoutForPath = (path: string): number => {
  if (path.startsWith("/dashboard/overview")) {
    return lastKnownOverviewRunMode === "replay" ? 15000 : 8000;
  }
  if (path.startsWith("/dashboard/trades") || path.startsWith("/dashboard/executions") || path.startsWith("/dashboard/open_orders")) {
    return 8000;
  }
  return 8000;
};
export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const timeoutMs = timeoutForPath(path);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers ?? {}),
      },
    });
  } catch (error) {
    const aborted = error instanceof Error && error.name === "AbortError";
    const message = aborted ? `Request timed out after ${timeoutMs}ms` : error instanceof Error ? error.message : "Failed to fetch";
    const apiError: ApiError = { status: 0, message, url };
    throw apiError;
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload?.detail ?? payload?.message ?? message;
    } catch (error) {
      message = response.statusText;
    }
    const apiError: ApiError = { status: response.status, message, url };
    throw apiError;
  }

  const payload = (await response.json()) as T;
  if (path.startsWith("/dashboard/overview")) {
    lastKnownOverviewRunMode = inferRunModeFromOverview(payload);
  }
  return payload;
}


export async function fetchEngineStatus(): Promise<EngineStatus> {
  return apiFetch<EngineStatus>("/engine/status");
}


export type DebugConfigResponse = {
  effective: Record<string, unknown>;
  sources: Record<string, string>;
  env_keys: Record<string, string>;
};


export type RuntimeDiagnostics = Record<string, unknown>;

export type DebugResetRequest = {
  reset_replay_state: boolean;
  reset_governor_state: boolean;
  reset_trades_db: boolean;
  reset_performance: boolean;
  dry_run: boolean;
};
