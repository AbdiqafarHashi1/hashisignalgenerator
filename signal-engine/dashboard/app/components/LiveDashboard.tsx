"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, API_BASE, ApiError, fetchEngineStatus, EngineStatus } from "../../lib/api";

const CONTROL_BUTTONS = [
  { label: "Start", path: "/start" },
  { label: "Stop", path: "/stop" },
  { label: "Run Once (Force)", path: "/run" },
  { label: "Test Telegram", path: "/test/telegram" },
];

type DecisionResponse = {
  symbol: string;
  decision: Record<string, unknown> | null;
};

type DailyStateResponse = {
  symbol: string;
  state: Record<string, unknown> | null;
};

type PositionsResponse = { positions: Array<Record<string, unknown>> };

type TradesResponse = { trades: Array<Record<string, unknown>> };

type StatsResponse = Record<string, unknown>;

type SymbolsResponse = { symbols: string[] };

type AccountSummary = {
  starting_balance_usd: number | null;
  balance_usd: number | null;
  equity_usd: number | null;
  realized_pnl_usd: number | null;
  unrealized_pnl_usd: number | null;
  total_pnl_usd: number | null;
  pnl_pct: number | null;
  open_positions: number | null;
  trades_today: number | null;
  wins_today: number | null;
  losses_today: number | null;
  win_rate_today: number | null;
  profit_factor: number | null;
  expectancy: number | null;
  max_drawdown_pct: number | null;
  last_updated_ts: string | null;
  equity_curve?: number[];
};

type AccountSummary = {
  starting_balance_usd: number | null;
  balance_usd: number | null;
  equity_usd: number | null;
  realized_pnl_usd: number | null;
  unrealized_pnl_usd: number | null;
  total_pnl_usd: number | null;
  pnl_pct: number | null;
  open_positions: number | null;
  trades_today: number | null;
  wins_today: number | null;
  losses_today: number | null;
  win_rate_today: number | null;
  profit_factor: number | null;
  expectancy: number | null;
  max_drawdown_pct: number | null;
  last_updated_ts: string | null;
  equity_curve?: number[];
};

export default function LiveDashboard() {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [decision, setDecision] = useState<Record<string, unknown> | null>(null);
  const [decisionError, setDecisionError] = useState<string>("");
  const [positions, setPositions] = useState<Array<Record<string, unknown>>>([]);
  const [trades, setTrades] = useState<Array<Record<string, unknown>>>([]);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [todayState, setTodayState] = useState<Record<string, unknown> | null>(null);
  const [accountSummary, setAccountSummary] = useState<AccountSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string>("");
  const [accountSummary, setAccountSummary] = useState<AccountSummary | null>(null);
  const [summaryError, setSummaryError] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  const [statusError, setStatusError] = useState<string>("");
  const [actionInFlight, setActionInFlight] = useState<string | null>(null);
  const [heartbeatAgeSeconds, setHeartbeatAgeSeconds] = useState<number | null>(null);
  const [activityMessages, setActivityMessages] = useState<
    Array<{ id: string; message: string; tone: "info" | "error" }>
  >([]);

  const symbolLabel = useMemo(() => selectedSymbol || "--", [selectedSymbol]);
  const statusLastHeartbeat = useMemo(() => {
    if (!engineStatus?.last_heartbeat_ts) {
      return null;
    }
    const ts = new Date(engineStatus.last_heartbeat_ts).getTime();
    if (Number.isNaN(ts)) {
      return null;
    }
    return ts;
  }, [engineStatus?.last_heartbeat_ts]);

  const isRunning = engineStatus?.status === "RUNNING";
  const isStopped = engineStatus?.status === "STOPPED";
  const statusBadge = statusError
    ? "DISCONNECTED/API ERROR"
    : isRunning
    ? "RUNNING"
    : isStopped
    ? "STOPPED"
    : "UNKNOWN";
  const statusClass = statusError ? "yellow" : isRunning ? "green" : "red";

  const formatApiError = useCallback((label: string, apiError: ApiError) => {
    return `${label}: ${apiError.message} (Status: ${apiError.status} URL: ${apiError.url})`;
  }, []);

  const loadSymbols = useCallback(async () => {
    try {
      const payload = await apiFetch<SymbolsResponse>("/symbols");
      setSymbols(payload.symbols);
      if (!selectedSymbol && payload.symbols.length > 0) {
        setSelectedSymbol(payload.symbols[0]);
      }
    } catch (err) {
      const apiError = err as ApiError;
      setError(formatApiError("Symbols", apiError));
    }
  }, [formatApiError, selectedSymbol]);

  const addActivityMessage = useCallback((message: string, tone: "info" | "error" = "info") => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setActivityMessages((prev) => {
      const next = [{ id, message, tone }, ...prev];
      return next.slice(0, 10);
    });
  }, []);

  const loadFast = useCallback(async () => {
    if (!selectedSymbol) {
      return;
    }
    try {
      const decisionPayload = await apiFetch<DecisionResponse>(
        `/decision/latest?symbol=${selectedSymbol}`
      );
      setDecision(decisionPayload.decision ?? null);
      setDecisionError("");
    } catch (err) {
      const apiError = err as ApiError;
      if (apiError.status === 404 && apiError.message === "no_decision") {
        setDecision(null);
        setDecisionError("");
      } else {
        setDecision(null);
        setDecisionError(formatApiError("Decision", apiError));
      }
    }

    try {
      const positionsPayload = await apiFetch<PositionsResponse>("/positions");
      setPositions(positionsPayload.positions ?? []);
    } catch (err) {
      const apiError = err as ApiError;
      setError(formatApiError("Positions", apiError));
    }

    try {
      const todayPayload = await apiFetch<DailyStateResponse>(
        `/state/today?symbol=${selectedSymbol}`
      );
      setTodayState(todayPayload.state ?? null);
    } catch (err) {
      const apiError = err as ApiError;
      setError(formatApiError("State today", apiError));
    }

    try {
      const summaryPayload = await apiFetch<AccountSummary>("/account/summary");
      setAccountSummary(summaryPayload);
      setSummaryError("");
    } catch (err) {
      const apiError = err as ApiError;
      setSummaryError(formatApiError("Account summary", apiError));
    }

    try {
      const summaryPayload = await apiFetch<AccountSummary>("/account/summary");
      setAccountSummary(summaryPayload);
      setSummaryError("");
    } catch (err) {
      const apiError = err as ApiError;
      setSummaryError(formatApiError("Account summary", apiError));
    }
  }, [formatApiError, selectedSymbol]);

  const loadSlow = useCallback(async () => {
    try {
      const tradesPayload = await apiFetch<TradesResponse>("/trades");
      setTrades(tradesPayload.trades ?? []);
    } catch (err) {
      const apiError = err as ApiError;
      setError(formatApiError("Trades", apiError));
    }

    try {
      const statsPayload = await apiFetch<StatsResponse>("/stats");
      setStats(statsPayload);
    } catch (err) {
      const apiError = err as ApiError;
      setError(formatApiError("Stats", apiError));
    }

  }, [formatApiError]);

  const loadEngineStatus = useCallback(async () => {
    try {
      const payload = await fetchEngineStatus();
      setEngineStatus(payload);
      setStatusError("");
    } catch (err) {
      const apiError = err as ApiError;
      const message = formatApiError("Engine status", apiError);
      setStatusError((prev) => {
        if (prev !== message) {
          addActivityMessage(message, "error");
        }
        return message;
      });
    }
  }, [addActivityMessage, formatApiError]);

  useEffect(() => {
    loadSymbols();
  }, [loadSymbols]);

  useEffect(() => {
    if (!selectedSymbol) {
      return;
    }
    loadFast();
    loadSlow();
    const fastTimer = setInterval(loadFast, 2000);
    const slowTimer = setInterval(loadSlow, 10000);
    return () => {
      clearInterval(fastTimer);
      clearInterval(slowTimer);
    };
  }, [selectedSymbol, loadFast, loadSlow]);

  useEffect(() => {
    loadEngineStatus();
    const statusTimer = setInterval(loadEngineStatus, 3000);
    return () => clearInterval(statusTimer);
  }, [loadEngineStatus]);

  useEffect(() => {
    if (!statusLastHeartbeat) {
      setHeartbeatAgeSeconds(null);
      return;
    }
    const updateAge = () => {
      setHeartbeatAgeSeconds(
        Math.max(0, Math.floor((Date.now() - statusLastHeartbeat) / 1000))
      );
    };
    updateAge();
    const heartbeatTimer = setInterval(updateAge, 1000);
    return () => clearInterval(heartbeatTimer);
  }, [statusLastHeartbeat]);

  const handleAction = async (path: string, label: string) => {
    setError("");
    try {
      setActionInFlight(label);
      await apiFetch(path);
      if (label === "Start") {
        addActivityMessage("Engine started");
      } else if (label === "Stop") {
        addActivityMessage("Engine stopped");
      } else if (label.startsWith("Run Once")) {
        addActivityMessage("Manual run executed");
      } else {
        addActivityMessage(`${label} completed`);
      }
      await loadEngineStatus();
    } catch (err) {
      const apiError = err as ApiError;
      const message = formatApiError(label, apiError);
      setError(message);
      addActivityMessage(message, "error");
    } finally {
      setActionInFlight(null);
    }
  };

  const decisionDisplay =
    decision ?? (decisionError ? { status: "error" } : { status: "waiting" });
  const lastActionText = useMemo(() => {
    if (!engineStatus?.last_action) {
      return "--";
    }
    if (typeof engineStatus.last_action === "string") {
      return engineStatus.last_action;
    }
    const type = engineStatus.last_action.type ?? "unknown";
    const ts = engineStatus.last_action.ts ?? "--";
    const detail = engineStatus.last_action.detail ? ` • ${engineStatus.last_action.detail}` : "";
    return `${type} • ${ts}${detail}`;
  }, [engineStatus?.last_action]);

  const summaryUpdatedAt = useMemo(() => {
    if (!accountSummary?.last_updated_ts) {
      return "--";
    }
    const ts = new Date(accountSummary.last_updated_ts);
    if (Number.isNaN(ts.getTime())) {
      return accountSummary.last_updated_ts;
    }
    return ts.toLocaleTimeString();
  }, [accountSummary?.last_updated_ts]);

  const formatNumber = (value: number | null | undefined) =>
    value === null || value === undefined ? "--" : String(value);
  const formatCurrency = (value: number | null | undefined) =>
    value === null || value === undefined ? "--" : value.toFixed(2);
  const formatPercent = (value: number | null | undefined) =>
    value === null || value === undefined ? "--" : `${value.toFixed(2)}%`;

  const toneForValue = (value: number | null | undefined) => {
    if (value === null || value === undefined) {
      return "text-slate-300";
    }
    if (value > 0) {
      return "text-emerald-400";
    }
    if (value < 0) {
      return "text-red-400";
    }
    return "text-slate-200";
  };

  const winRatePercent =
    accountSummary?.win_rate_today == null ? null : accountSummary.win_rate_today * 100;
  const pnlPercent = accountSummary?.pnl_pct ?? null;
  const maxDrawdownPercent = accountSummary?.max_drawdown_pct ?? null;

  const accountKpis = [
    { label: "Balance (USD)", value: formatCurrency(accountSummary?.balance_usd), tone: toneForValue(accountSummary?.balance_usd) },
    { label: "Equity (USD)", value: formatCurrency(accountSummary?.equity_usd), tone: toneForValue(accountSummary?.equity_usd) },
    { label: "PnL", value: formatCurrency(accountSummary?.total_pnl_usd), tone: toneForValue(accountSummary?.total_pnl_usd) },
    { label: "Realized PnL", value: formatCurrency(accountSummary?.realized_pnl_usd), tone: toneForValue(accountSummary?.realized_pnl_usd) },
    { label: "Unrealized PnL", value: formatCurrency(accountSummary?.unrealized_pnl_usd), tone: toneForValue(accountSummary?.unrealized_pnl_usd) },
    { label: "Total PnL", value: formatCurrency(accountSummary?.total_pnl_usd), tone: toneForValue(accountSummary?.total_pnl_usd) },
    {
      label: "PnL %",
      value: formatPercent(pnlPercent),
      tone: toneForValue(accountSummary?.pnl_pct),
    },
  ];

  const tradingKpis = [
    {
      label: "Open Positions",
      value: accountSummary?.open_positions,
      format: formatNumber,
    },
    {
      label: "Trades Today",
      value: accountSummary?.trades_today,
      format: formatNumber,
    },
    {
      label: "Wins",
      value: accountSummary?.wins_today,
      format: formatNumber,
    },
    {
      label: "Losses",
      value: accountSummary?.losses_today,
      format: formatNumber,
    },
    {
      label: "Win Rate %",
      value: winRatePercent,
      format: formatPercent,
    },
    {
      label: "Profit Factor",
      value: accountSummary?.profit_factor,
      format: formatCurrency,
    },
    {
      label: "Expectancy",
      value: accountSummary?.expectancy,
      format: formatCurrency,
    },
    {
      label: "Max Drawdown %",
      value: maxDrawdownPercent,
      format: formatPercent,
    },
  ];

  const equityData = accountSummary?.equity_curve ?? [];

  return (
    <div className="min-h-screen px-6 py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-binance-border bg-binance-card/80 p-6 shadow-panel">
          <div className="space-y-3">
            <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
              Signal Engine
            </p>
            <h1 className="text-3xl font-semibold text-white">Live Control Dashboard</h1>
            <p className="text-sm text-slate-400">
              API Base: <span className="text-slate-200">{API_BASE}</span>
            </p>
            <div className="flex flex-wrap items-center gap-3 text-sm text-slate-300">
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-white ${statusClass}`}
              >
                {statusBadge}
              </span>
              <span>
                Last heartbeat:{" "}
                {heartbeatAgeSeconds === null ? "never" : `${heartbeatAgeSeconds}s ago`}
              </span>
              <span className="text-slate-400">Mode: {engineStatus?.mode ?? "--"}</span>
            </div>
            {statusError ? (
              <p className="text-xs text-yellow-400">{statusError}</p>
            ) : null}
            <div className="text-xs text-slate-400">
              <span className="font-semibold text-slate-300">Last action:</span>{" "}
              {lastActionText}
            </div>
          </div>
          <div className="flex flex-col items-end gap-2 text-right">
            <span className="text-sm text-slate-400">Active symbol</span>
            <div className="rounded-lg bg-binance-dark px-4 py-2 text-lg font-semibold text-binance-accent shadow-panel">
              {symbolLabel}
            </div>
          </div>
        </header>

        <section className="grid gap-4 rounded-2xl border border-binance-border bg-binance-card p-6 shadow-panel">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold text-white">Account Summary</h2>
              <p className="text-sm text-slate-400">Last update: {summaryUpdatedAt}</p>
            </div>
            {summaryError ? (
              <span className="rounded-full border border-yellow-400/60 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-yellow-300">
                API error
              </span>
            ) : null}
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Account</p>
              <div className="grid gap-4 sm:grid-cols-2">
                {accountKpis.map((item) => (
                  <div
                    key={item.label}
                    className="rounded-xl border border-binance-border bg-binance-dark/60 p-4"
                  >
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      {item.label}
                    </p>
                    <p className={`mt-2 text-2xl font-semibold ${item.tone}`}>{item.value}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Trading</p>
              <div className="grid gap-4 sm:grid-cols-2">
                {tradingKpis.map((item) => (
                  <div
                    key={item.label}
                    className="rounded-xl border border-binance-border bg-binance-dark/60 p-4"
                  >
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      {item.label}
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-slate-200">
                      {item.format(item.value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-binance-border bg-binance-dark/60 p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Equity Curve</p>
              <span className="text-xs text-slate-500">Auto-refreshing</span>
            </div>
            <EquityChart data={equityData} />
          </div>
        </section>

        <section className="grid gap-4 rounded-2xl border border-binance-border bg-binance-card p-6 shadow-panel">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold text-white">Account Summary</h2>
              <p className="text-sm text-slate-400">Last update: {summaryUpdatedAt}</p>
            </div>
            {summaryError ? (
              <span className="rounded-full border border-yellow-400/60 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-yellow-300">
                API error
              </span>
            ) : null}
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Account</p>
              <div className="grid gap-4 sm:grid-cols-2">
                {accountKpis.map((item) => (
                  <div
                    key={item.label}
                    className="rounded-xl border border-binance-border bg-binance-dark/60 p-4"
                  >
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      {item.label}
                    </p>
                    <p className={`mt-2 text-2xl font-semibold ${item.tone}`}>{item.value}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Trading</p>
              <div className="grid gap-4 sm:grid-cols-2">
                {tradingKpis.map((item) => (
                  <div
                    key={item.label}
                    className="rounded-xl border border-binance-border bg-binance-dark/60 p-4"
                  >
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      {item.label}
                    </p>
                    <p className="mt-2 text-2xl font-semibold text-slate-200">
                      {item.format(item.value)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="rounded-xl border border-binance-border bg-binance-dark/60 p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Equity Curve</p>
              <span className="text-xs text-slate-500">Auto-refreshing</span>
            </div>
            <EquityChart data={equityData} />
          </div>
        </section>

        <section className="grid gap-4 rounded-2xl border border-binance-border bg-binance-card p-6 shadow-panel">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-white">Quick Controls</h2>
              <p className="text-sm text-slate-400">One-click operations</p>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-slate-400">Symbol</label>
              <select
                className="rounded-lg border border-binance-border bg-binance-dark px-3 py-2 text-sm text-slate-200"
                value={selectedSymbol}
                onChange={(event) => setSelectedSymbol(event.target.value)}
              >
                {symbols.map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            {CONTROL_BUTTONS.map((button) => {
              const disabled =
                (button.label === "Start" && isRunning) ||
                (button.label === "Stop" && !isRunning) ||
                Boolean(actionInFlight);
              return (
                <button
                  key={button.label}
                  onClick={() => handleAction(button.path, button.label)}
                  disabled={disabled}
                  className={`rounded-lg border border-binance-border px-4 py-2 text-sm font-semibold transition ${
                    disabled
                      ? "cursor-not-allowed bg-slate-800 text-slate-500"
                      : "bg-binance-dark text-slate-200 hover:border-binance-accent hover:text-white"
                  }`}
                >
                  {button.label}
                </button>
              );
            })}
          </div>
          {error ? <p className="text-sm text-red-400">{error}</p> : null}
          <div className="rounded-xl border border-binance-border bg-binance-dark/60 p-3">
            <p className="mb-2 text-xs uppercase tracking-[0.2em] text-slate-400">
              Activity
            </p>
            {activityMessages.length ? (
              <ul className="space-y-1 text-sm">
                {activityMessages.map((item) => (
                  <li
                    key={item.id}
                    className={item.tone === "error" ? "text-red-400" : "text-slate-200"}
                  >
                    {item.message}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500">No recent activity.</p>
            )}
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-6">
            <Panel title="Latest Decision">
              {decisionError ? (
                <p className="mb-3 text-sm text-red-400">{decisionError}</p>
              ) : null}
              <JsonBlock data={decisionDisplay} />
            </Panel>
            <Panel title="Today State">
              <JsonBlock data={todayState ?? { status: "waiting" }} />
            </Panel>
            <Panel title="Positions">
              <TableBlock rows={positions} emptyLabel="No open positions" />
            </Panel>
          </div>
          <div className="space-y-6">
            <Panel title="Stats">
              <JsonBlock data={stats ?? { status: "waiting" }} />
            </Panel>
            <Panel title="Trades">
              <TableBlock rows={trades} emptyLabel="No trades yet" />
            </Panel>
          </div>
        </section>
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-binance-border bg-binance-card p-5 shadow-panel">
      <h3 className="mb-3 text-lg font-semibold text-white">{title}</h3>
      {children}
    </div>
  );
}

function JsonBlock({ data }: { data: Record<string, unknown> }) {
  return (
    <pre className="max-h-80 overflow-auto rounded-xl bg-binance-dark p-4 text-xs text-slate-200">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

function TableBlock({
  rows,
  emptyLabel,
}: {
  rows: Array<Record<string, unknown>>;
  emptyLabel: string;
}) {
  if (!rows.length) {
    return <p className="text-sm text-slate-400">{emptyLabel}</p>;
  }

  const columns = Object.keys(rows[0] ?? {});

  return (
    <div className="max-h-80 overflow-auto rounded-xl border border-binance-border">
      <table className="min-w-full text-left text-xs text-slate-200">
        <thead className="sticky top-0 bg-binance-dark">
          <tr>
            {columns.map((column) => (
              <th key={column} className="px-3 py-2 text-slate-400">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={`${row.id ?? index}`}
              className={index % 2 === 0 ? "bg-binance-card" : "bg-binance-dark"}
            >
              {columns.map((column) => (
                <td key={column} className="px-3 py-2 align-top">
                  <Cell value={row[column]} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Cell({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-slate-500">--</span>;
  }
  if (typeof value === "object") {
    return (
      <pre className="whitespace-pre-wrap break-words text-[11px]">
        {JSON.stringify(value, null, 2)}
      </pre>
    );
  }
  return <span>{String(value)}</span>;
}

function EquityChart({ data }: { data: number[] }) {
  if (!data.length) {
    return <p className="text-sm text-slate-400">No equity data yet.</p>;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data
    .map((value, index) => {
      const x = (index / Math.max(1, data.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="h-40 w-full">
      <svg viewBox="0 0 100 100" className="h-full w-full" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke="#22c55e"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="mt-2 flex justify-between text-xs text-slate-500">
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  );
}

function EquityChart({ data }: { data: number[] }) {
  if (!data.length) {
    return <p className="text-sm text-slate-400">No equity data yet.</p>;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data
    .map((value, index) => {
      const x = (index / Math.max(1, data.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="h-40 w-full">
      <svg viewBox="0 0 100 100" className="h-full w-full" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke="#22c55e"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="mt-2 flex justify-between text-xs text-slate-500">
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  );
}
