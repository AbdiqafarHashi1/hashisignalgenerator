"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, API_BASE, ApiError, EngineState, PerformanceMetrics, stateEventsUrl } from "../../lib/api";

const CONTROL_BUTTONS = [
  { label: "Start", path: "/start" },
  { label: "Stop", path: "/stop" },
  { label: "Run Once (Force)", path: "/run?force=true" },
];

const MAX_POINTS = 200;

export default function LiveDashboard() {
  const [state, setState] = useState<EngineState | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
  const [equityCurve, setEquityCurve] = useState<number[]>([]);
  const [pnlCurve, setPnlCurve] = useState<number[]>([]);
  const [isStreamLive, setIsStreamLive] = useState(false);
  const [lastUpdateAtMs, setLastUpdateAtMs] = useState<number | null>(null);
  const [error, setError] = useState<string>("");
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const formatApiError = useCallback((label: string, apiError: ApiError) => {
    return `${label}: ${apiError.message} (Status: ${apiError.status})`;
  }, []);

  const loadPerformance = useCallback(async () => {
    try {
      const payload = await apiFetch<PerformanceMetrics>("/metrics/performance");
      setPerformance(payload);
    } catch (err) {
      setError(formatApiError("Performance", err as ApiError));
    }
  }, [formatApiError]);

  const pushChartPoints = useCallback((snapshot: EngineState) => {
    setEquityCurve((prev) => [...prev, snapshot.equity].slice(-MAX_POINTS));
    setPnlCurve((prev) => [...prev, snapshot.realized_pnl_today_usd].slice(-MAX_POINTS));
  }, []);

  const applySnapshot = useCallback((snapshot: EngineState) => {
    setState(snapshot);
    setLastUpdateAtMs(Date.now());
    pushChartPoints(snapshot);
  }, [pushChartPoints]);

  useEffect(() => {
    const loadInitial = async () => {
      try {
        const payload = await apiFetch<EngineState>("/state");
        setEquityCurve([payload.equity]);
        setPnlCurve([payload.realized_pnl_today_usd]);
        applySnapshot(payload);
        setError("");
      } catch (err) {
        setError(formatApiError("State", err as ApiError));
      }
    };

    loadInitial();
    loadPerformance();
  }, [applySnapshot, formatApiError, loadPerformance]);

  useEffect(() => {
    const timer = setInterval(() => {
      loadPerformance();
    }, 2000);
    return () => clearInterval(timer);
  }, [loadPerformance]);

  useEffect(() => {
    let source: EventSource | null = null;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      source = new EventSource(stateEventsUrl);

      source.onopen = () => {
        setIsStreamLive(true);
      };

      source.onerror = () => {
        setIsStreamLive(false);
        source?.close();
        if (!cancelled) {
          reconnectTimer.current = setTimeout(connect, 1500);
        }
      };

      source.addEventListener("state", (event) => {
        try {
          const payload = JSON.parse((event as MessageEvent<string>).data) as EngineState;
          applySnapshot(payload);
          setIsStreamLive(true);
          setError("");
        } catch {
          setError("State stream parse error");
        }
      });
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      source?.close();
    };
  }, [applySnapshot]);

  const handleAction = async (path: string) => {
    try {
      await apiFetch(path);
      const payload = await apiFetch<EngineState>("/state");
      applySnapshot(payload);
      loadPerformance();
    } catch (err) {
      setError(formatApiError("Action", err as ApiError));
    }
  };

  const runningBadge = useMemo(() => {
    const isRunning = Boolean(state?.running);
    const classes = isRunning ? "bg-emerald-600" : "bg-red-600";
    return <span className={`rounded-full px-3 py-1 text-xs font-semibold ${classes}`}>{isRunning ? "RUNNING" : "STOPPED"}</span>;
  }, [state?.running]);

  const streamBadge = useMemo(() => {
    const classes = isStreamLive ? "bg-emerald-700" : "bg-amber-600";
    return <span className={`rounded-full px-3 py-1 text-xs font-semibold ${classes}`}>{isStreamLive ? "LIVE" : "DISCONNECTED"}</span>;
  }, [isStreamLive]);

  const fmtNum = (v: number | undefined | null) => (v === undefined || v === null ? "--" : Number(v).toFixed(2));
  const fmtUsd = (v: number | undefined | null) => (v === undefined || v === null ? "--" : `$${Number(v).toFixed(2)}`);
  const fmtPct = (v: number | undefined | null, asFraction = false) => {
    if (v === undefined || v === null) return "--";
    const num = asFraction ? v * 100 : v;
    return `${num.toFixed(2)}%`;
  };

  const lastTickAgeSeconds = state?.last_tick_age_seconds == null ? null : Math.max(0, Math.floor(state.last_tick_age_seconds));
  const lastUpdateAgeSeconds = lastUpdateAtMs == null ? null : Math.max(0, Math.floor((Date.now() - lastUpdateAtMs) / 1000));

  return (
    <div className="min-h-screen bg-slate-950 px-5 py-6 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="rounded-xl border border-binance-border bg-binance-card p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold">Professional Trading Dashboard</h1>
              <p className="text-xs text-slate-400">API: {API_BASE}</p>
              <p className="text-xs text-slate-400">Last tick: {lastTickAgeSeconds === null ? "--" : `${lastTickAgeSeconds} seconds ago`}</p>
              <p className="text-xs text-slate-400">Last update age: {lastUpdateAgeSeconds === null ? "--" : `${lastUpdateAgeSeconds} seconds ago`}</p>
            </div>
            <div className="flex gap-2">{streamBadge}{runningBadge}</div>
          </div>
        </header>

        {error ? <p className="text-sm text-red-400">{error}</p> : null}

        <section>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-300">Account Status</h2>
          <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Kpi label="Equity" value={fmtUsd(state?.equity)} />
          <Kpi label="Balance" value={fmtUsd(state?.balance)} />
          <Kpi label="Unrealized PnL (Live)" value={fmtUsd(state?.unrealized_pnl_usd)} />
          <Kpi label="Realized PnL Today" value={fmtUsd(state?.realized_pnl_today_usd)} />
          <Kpi label="Open Positions" value={String(state?.open_positions?.length ?? "--")} />
          <Kpi label="Trades Today" value={String(state?.trades_today ?? "--")} />
          </div>
        </section>

        <section>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-300">Performance Analytics</h2>
          <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-7">
          <Kpi label="Expectancy (R)" value={fmtNum(performance?.expectancy_r)} />
          <Kpi label="Profit Factor" value={fmtNum(performance?.profit_factor)} />
          <Kpi label="Avg Win" value={fmtUsd(performance?.avg_win)} />
          <Kpi label="Avg Loss" value={fmtUsd(performance?.avg_loss)} />
          <Kpi label="Trades Today" value={String(performance?.trades_today ?? "--")} />
          <Kpi label="Win Rate" value={fmtPct(performance?.win_rate, true)} />
          <Kpi label="Max DD %" value={fmtPct(performance?.max_drawdown_pct)} />
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Panel title="Equity Curve"><Spark data={performance?.equity_curve || equityCurve} color="#22c55e" /></Panel>
          <Panel title="Drawdown Curve"><Spark data={performance?.drawdown_curve_pct || pnlCurve} color="#f97316" /></Panel>
        </section>

        <section className="grid gap-4 lg:grid-cols-3">
          <Panel title="Skip Reason Histogram"><Bars data={performance?.skip_reason_counts || {}} /></Panel>
          <Panel title="Win/Loss Distribution"><Bars data={performance?.win_loss_distribution || {}} /></Panel>
          <Panel title="Avg Hold Time"><Kpi label="Seconds" value={fmtNum(performance?.avg_hold_time)} /></Panel>
        </section>

        <section className="rounded-xl border border-binance-border bg-binance-card p-4">
          <div className="mb-3 flex flex-wrap gap-2">
            {CONTROL_BUTTONS.map((button) => (
              <button key={button.label} onClick={() => handleAction(button.path)} className="rounded border border-binance-border bg-binance-dark px-3 py-2 text-sm hover:border-binance-accent">
                {button.label}
              </button>
            ))}
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Panel title="Live Trades / Open Positions"><Table rows={state?.open_positions || []} emptyLabel="No open positions" /></Panel>
          <Panel title="Recent Trades"><Table rows={state?.recent_trades || []} emptyLabel="No trades" /></Panel>
        </section>
      </div>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-binance-border bg-binance-card p-3">
      <p className="text-xs uppercase text-slate-400">{label}</p>
      <p className="mt-1 break-all text-lg font-semibold text-slate-100">{value}</p>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-binance-border bg-binance-card p-4">
      <h3 className="mb-3 text-lg font-semibold">{title}</h3>
      {children}
    </div>
  );
}

function Table({ rows, emptyLabel }: { rows: Array<Record<string, unknown>>; emptyLabel: string }) {
  if (!rows.length) return <p className="text-sm text-slate-400">{emptyLabel}</p>;
  const columns = Object.keys(rows[0]);
  return (
    <div className="max-h-72 overflow-auto rounded border border-binance-border">
      <table className="min-w-full text-xs">
        <thead className="bg-binance-dark"><tr>{columns.map((c) => <th key={c} className="px-2 py-1 text-left">{c}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i} className={i % 2 ? "bg-binance-dark" : "bg-binance-card"}>{columns.map((c) => <td key={c} className="px-2 py-1">{String(r[c] ?? "--")}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function Spark({ data, color }: { data: number[]; color: string }) {
  if (!data.length) return <p className="text-sm text-slate-400">No data.</p>;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data
    .map((v, i) => `${(i / Math.max(1, data.length - 1)) * 100},${100 - ((v - min) / range) * 100}`)
    .join(" ");
  return <svg viewBox="0 0 100 100" className="h-40 w-full"><polyline points={points} fill="none" stroke={color} strokeWidth="2" /></svg>;
}

function Bars({ data }: { data: Record<string, number> }) {
  const entries = Object.entries(data);
  if (!entries.length) return <p className="text-sm text-slate-400">No data.</p>;
  const max = Math.max(...entries.map(([, v]) => v), 1);
  return (
    <div className="space-y-2 text-xs">
      {entries.map(([label, value]) => (
        <div key={label}>
          <div className="mb-1 flex justify-between text-slate-300"><span>{label}</span><span>{value}</span></div>
          <div className="h-2 rounded bg-slate-800"><div className="h-2 rounded bg-cyan-500" style={{ width: `${(value / max) * 100}%` }} /></div>
        </div>
      ))}
    </div>
  );
}
