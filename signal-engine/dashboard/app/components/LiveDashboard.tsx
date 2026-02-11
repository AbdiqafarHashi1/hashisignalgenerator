"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, API_BASE, ApiError, fetchEngineStatus, EngineStatus } from "../../lib/api";

const CONTROL_BUTTONS = [
  { label: "Start", path: "/start" },
  { label: "Stop", path: "/stop" },
  { label: "Run Once (Force)", path: "/run?force=true" },
];

type AccountSummary = {
  equity_usd: number;
  balance_usd: number;
  unrealized_pnl_usd: number;
  realized_pnl_today_usd: number;
  daily_pct: number;
  trades_today: number;
  wins_today: number;
  losses_today: number;
  win_rate_today: number;
  profit_factor: number;
  max_drawdown_pct: number;
  engine_status: string;
  last_trade_ts: string | null;
  equity_curve: number[];
  pnl_curve: number[];
};

type RiskSummary = {
  daily_loss_remaining_usd: number;
  trades_remaining: number;
  consecutive_losses: number;
  cooldown_active: boolean;
  funding_blackout_active: boolean;
  sweet8_enabled?: boolean;
  sweet8_current_mode?: string;
  open_positions?: number;
  blocked_premature_exits?: number;
  daily_loss_pct?: number;
  risk_per_trade_pct?: number;
};

type PositionsResponse = { positions: Array<Record<string, unknown>> };

type TradesResponse = { trades: Array<Record<string, unknown>> };

export default function LiveDashboard() {
  const [summary, setSummary] = useState<AccountSummary | null>(null);
  const [risk, setRisk] = useState<RiskSummary | null>(null);
  const [positions, setPositions] = useState<Array<Record<string, unknown>>>([]);
  const [trades, setTrades] = useState<Array<Record<string, unknown>>>([]);
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  const [error, setError] = useState<string>("");

  const formatApiError = useCallback((label: string, apiError: ApiError) => {
    return `${label}: ${apiError.message} (Status: ${apiError.status})`;
  }, []);

  const load = useCallback(async () => {
    try {
      const [summaryPayload, riskPayload, positionsPayload, tradesPayload] = await Promise.all([
        apiFetch<AccountSummary>("/account/summary"),
        apiFetch<RiskSummary>("/risk/summary"),
        apiFetch<PositionsResponse>("/positions"),
        apiFetch<TradesResponse>("/trades"),
      ]);
      setSummary(summaryPayload);
      setRisk(riskPayload);
      setPositions(positionsPayload.positions || []);
      setTrades((tradesPayload.trades || []).slice(0, 50));
      setError("");
    } catch (err) {
      setError(formatApiError("Dashboard", err as ApiError));
    }
  }, [formatApiError]);

  useEffect(() => {
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, [load]);


  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const statusPayload = await fetchEngineStatus();
        setEngineStatus(statusPayload);
      } catch (err) {
        setError(formatApiError("Engine status", err as ApiError));
      }
    };

    fetchStatus();
    const id = setInterval(fetchStatus, 5000);
    return () => clearInterval(id);
  }, [formatApiError]);

  const handleAction = async (path: string) => {
    try {
      await apiFetch(path);
      await load();
    } catch (err) {
      setError(formatApiError("Action", err as ApiError));
    }
  };

  const statusBadge = useMemo(() => {
    const isRunning = Boolean(engineStatus?.running);
    const status = isRunning ? "RUNNING" : "STOPPED";
    const classes = isRunning ? "bg-emerald-600" : "bg-red-600";
    return <span className={`rounded-full px-3 py-1 text-xs font-semibold ${classes}`}>{status}</span>;
  }, [engineStatus?.running]);

  const fmt = (v: number | undefined | null) => (v === undefined || v === null ? "--" : v.toFixed(2));

  const lastTickAgeSeconds =
    typeof engineStatus?.last_tick_ts === "number"
      ? Math.max(0, Math.floor(Date.now() / 1000 - engineStatus.last_tick_ts))
      : null;
  const engineStale = lastTickAgeSeconds !== null && lastTickAgeSeconds > 30;

  return (
    <div className="min-h-screen bg-slate-950 px-5 py-6 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="rounded-xl border border-binance-border bg-binance-card p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold">Professional Trading Dashboard</h1>
              <p className="text-xs text-slate-400">API: {API_BASE}</p>
              <p className="text-xs text-slate-400">Last tick: {lastTickAgeSeconds === null ? "--" : `${lastTickAgeSeconds} seconds ago`}</p>
              {engineStale ? <p className="text-xs font-semibold text-amber-400">Engine stale</p> : null}
            </div>
            {statusBadge}
          </div>
        </header>

        {error ? <p className="text-sm text-red-400">{error}</p> : null}

        <section className="grid gap-3 md:grid-cols-3 xl:grid-cols-5">
          <Kpi label="Equity" value={fmt(summary?.equity_usd)} />
          <Kpi label="Balance" value={fmt(summary?.balance_usd)} />
          <Kpi label="Unrealized PnL" value={fmt(summary?.unrealized_pnl_usd)} />
          <Kpi label="Realized PnL Today" value={fmt(summary?.realized_pnl_today_usd)} />
          <Kpi label="Daily %" value={`${fmt(summary?.daily_pct)}%`} />
          <Kpi label="Trades Today" value={String(summary?.trades_today ?? "--")} />
          <Kpi label="Wins" value={String(summary?.wins_today ?? "--")} />
          <Kpi label="Losses" value={String(summary?.losses_today ?? "--")} />
          <Kpi label="Win Rate" value={`${fmt((summary?.win_rate_today ?? 0) * 100)}%`} />
          <Kpi label="Profit Factor" value={fmt(summary?.profit_factor)} />
          <Kpi label="Max DD Today" value={`${fmt(summary?.max_drawdown_pct)}%`} />
          <Kpi label="Last Trade" value={summary?.last_trade_ts ?? "--"} />
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Panel title="Equity Curve"><Spark data={summary?.equity_curve || []} color="#22c55e" /></Panel>
          <Panel title="PnL Curve"><Spark data={summary?.pnl_curve || []} color="#60a5fa" /></Panel>
        </section>

        <section className="grid gap-3 md:grid-cols-2 lg:grid-cols-5">
          <Kpi label="Daily Loss Remaining" value={fmt(risk?.daily_loss_remaining_usd)} />
          <Kpi label="Trades Remaining" value={String(risk?.trades_remaining ?? "--")} />
          <Kpi label="Consecutive Losses" value={String(risk?.consecutive_losses ?? "--")} />
          <Kpi label="Cooldown Active" value={risk?.cooldown_active ? "YES" : "NO"} />
          <Kpi label="Funding Blackout" value={risk?.funding_blackout_active ? "YES" : "NO"} />
          <Kpi label="Sweet8 Enabled" value={risk?.sweet8_enabled ? "YES" : "NO"} />
          <Kpi label="Current Mode" value={risk?.sweet8_current_mode?.toUpperCase() ?? ((engineStatus as any)?.sweet8_current_mode?.toUpperCase?.() || "--")} />
          <Kpi label="Open Positions" value={String(risk?.open_positions ?? positions.length)} />
          <Kpi label="Blocked Premature Exits" value={String(risk?.blocked_premature_exits ?? "--")} />
          <Kpi label="Daily Loss %" value={`${fmt((risk?.daily_loss_pct ?? 0) * 100)}%`} />
          <Kpi label="Risk per Trade" value={`${fmt((risk?.risk_per_trade_pct ?? 0) * 100)}%`} />
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
          <Panel title="Live Trades / Open Positions"><Table rows={positions} emptyLabel="No open positions" /></Panel>
          <Panel title="Recent Trades"><Table rows={trades} emptyLabel="No trades" /></Panel>
        </section>
      </div>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-binance-border bg-binance-card p-3">
      <p className="text-xs uppercase text-slate-400">{label}</p>
      <p className="mt-1 text-lg font-semibold text-slate-100 break-all">{value}</p>
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
