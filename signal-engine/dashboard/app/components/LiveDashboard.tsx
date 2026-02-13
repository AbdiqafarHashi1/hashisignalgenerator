"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";
import { apiFetch, API_BASE, DashboardOverview, EngineState, stateEventsUrl } from "../../lib/api";

const UI_REFRESH_MS = 3000;
const HEAVY_REFRESH_MS = 15000;

const CONTROLS = [
  { label: "Start", path: "/start" },
  { label: "Stop", path: "/stop" },
  { label: "Run Once", path: "/run?force=true" },
];

export default function LiveDashboard() {
  const [state, setState] = useState<EngineState | null>(null);
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [error, setError] = useState("");
  const [selectedSymbol, setSelectedSymbol] = useState<string>("ALL");

  const loadOverview = useCallback(async () => {
    try {
      const payload = await apiFetch<DashboardOverview>("/dashboard/overview");
      setOverview(payload);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  const handleAction = useCallback(async (path: string) => {
    await apiFetch(path);
    await Promise.all([loadOverview(), apiFetch<EngineState>("/state").then(setState)]);
  }, [loadOverview]);

  useEffect(() => {
    loadOverview();
    const timer = setInterval(loadOverview, HEAVY_REFRESH_MS);
    return () => clearInterval(timer);
  }, [loadOverview]);

  useEffect(() => {
    const source = new EventSource(stateEventsUrl);
    source.addEventListener("state", (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as EngineState;
      setState(payload);
    });
    source.onerror = () => source.close();
    return () => source.close();
  }, []);

  useEffect(() => {
    const timer = setInterval(async () => {
      const payload = await apiFetch<EngineState>("/state");
      setState(payload);
    }, UI_REFRESH_MS);
    return () => clearInterval(timer);
  }, []);

  const account = (overview?.account ?? {}) as Record<string, number | string | boolean | string[]>;
  const risk = (overview?.risk ?? {}) as Record<string, number>;
  const activity = (overview?.activity ?? {}) as Record<string, number>;
  const symbols = overview?.symbols ?? {};
  const skipGlobal = ((overview?.skip_reasons as Record<string, unknown> | undefined)?.global ?? {}) as Record<string, number>;
  const trades = (overview?.recent_trades ?? []).filter((row) => selectedSymbol === "ALL" || String(row.symbol) === selectedSymbol);
  const symbolKeys = Object.keys(symbols);

  const statusReasons = ((account.pause_reasons as string[] | undefined) ?? []).join(", ");
  const tickAge = Number(account.last_tick_age_seconds ?? state?.last_tick_age_seconds ?? 0);
  const tickInterval = Number(account.tick_interval_seconds ?? 60);
  const isStale = Number.isFinite(tickAge) && tickAge > tickInterval * 2;
  const running = Boolean(account.engine_running ?? state?.running ?? false);
  const liveLabel = running ? (isStale ? "STALE" : "LIVE") : "STOPPED";

  const equitySeries = (overview?.equity_curve ?? []).map((p) => p.equity);
  const drawdownSeries = toDrawdownCurve(equitySeries);

  return (
    <div className="min-h-screen bg-slate-950 px-5 py-6 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="rounded-xl border border-slate-700 bg-slate-900 p-4">
          <h1 className="text-2xl font-semibold">Prop Paper Dashboard</h1>
          <p className="text-xs text-slate-400">API: {API_BASE}</p>
          <p className="text-xs text-slate-400">Status: {liveLabel} ({running ? "RUNNING" : "STOPPED"})</p>
          <p className="text-xs text-slate-400">Last tick age: {Number.isFinite(tickAge) ? `${tickAge.toFixed(1)}s` : "--"}</p>
        </header>

        {error ? <p className="text-sm text-red-400">{error}</p> : null}

        <section className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Kpi label="Equity" value={fmt(account.live_equity, true)} />
          <Kpi label="PnL Today" value={fmt(account.realized_pnl_today, true)} />
          <Kpi label="Daily DD %" value={pct(account.daily_drawdown_pct)} />
          <Kpi label="Global DD %" value={pct(account.global_drawdown_pct)} />
          <Kpi label="Trades Today" value={String(activity.trades_today ?? "--")} />
          <Kpi label="Status" value={`${account.status ?? "--"}${statusReasons ? ` (${statusReasons})` : ""}`} />
        </section>

        <section className="rounded-xl border border-slate-700 bg-slate-900 p-4">
          <div className="flex flex-wrap gap-2">
            {CONTROLS.map((control) => (
              <button key={control.label} className="rounded border border-slate-600 px-3 py-2 text-sm" onClick={() => handleAction(control.path)}>
                {control.label}
              </button>
            ))}
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Panel title="Risk Panel">
            <p>Daily loss limit: {pct(risk.daily_loss_limit_pct)}</p>
            <p>Global DD limit: {pct(risk.global_dd_limit_pct)}</p>
            <p>Consecutive losses: {risk.consecutive_losses ?? 0}</p>
            <p>Cooldown remaining: {risk.cooldown_remaining_seconds ?? 0}s</p>
            <p>Max trades/day: {risk.max_trades_per_day ?? 0} (used {risk.trades_today ?? 0})</p>
          </Panel>
          <Panel title="Activity">
            <p>Win rate today: {pct(activity.win_rate_today)}</p>
            <p>Profit factor: {num(activity.profit_factor)}</p>
            <p>Expectancy: {num(activity.expectancy)}</p>
            <p>Avg win/loss: {fmt(activity.avg_win, true)} / {fmt(activity.avg_loss, true)}</p>
            <p>Fees today/rolling: {fmt(activity.fees_today, true)} / {fmt(activity.fees_rolling, true)}</p>
          </Panel>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Panel title="Equity Curve"><Spark data={equitySeries} color="#22c55e" /></Panel>
          <Panel title="Drawdown Curve"><Spark data={drawdownSeries} color="#f97316" /></Panel>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          {Object.entries(symbols).map(([symbol, data]) => {
            const d = data as Record<string, unknown>;
            const openPos = d.open_position as Record<string, unknown> | null;
            return (
              <Panel key={symbol} title={symbol}>
                <p>Regime: {String(d.regime ?? "--")}</p>
                <p>Last decision: {String(d.last_decision ?? "--")}</p>
                <p>Last skip reason: {String(d.last_skip_reason ?? "--")}</p>
                <p>ATR%: {pct(d.atr_pct as number)}</p>
                <p>Trend strength: {num(d.trend_strength as number)}</p>
                <p>Signal score: {num(d.signal_score as number)}</p>
                {openPos ? <p>Open {String(openPos.side)} size {num(openPos.size as number)} entry {num(openPos.entry as number)}</p> : <p>No open position</p>}
              </Panel>
            );
          })}
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Panel title="Skip Reason Histogram"><Bars data={skipGlobal} /></Panel>
          <Panel title="Trade Log">
            <div className="mb-2">
              <select className="rounded bg-slate-800 px-2 py-1 text-xs" value={selectedSymbol} onChange={(e) => setSelectedSymbol(e.target.value)}>
                <option value="ALL">ALL</option>
                {symbolKeys.map((symbol) => <option key={symbol} value={symbol}>{symbol}</option>)}
              </select>
            </div>
            <Table rows={trades.slice(0, 100)} />
          </Panel>
        </section>
      </div>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string }) { return <div className="rounded-xl border border-slate-700 bg-slate-900 p-3"><p className="text-xs text-slate-400">{label}</p><p>{value}</p></div>; }
function Panel({ title, children }: { title: string; children: ReactNode }) { return <div className="rounded-xl border border-slate-700 bg-slate-900 p-4"><h3 className="mb-2">{title}</h3>{children}</div>; }
function Table({ rows }: { rows: Array<Record<string, unknown>> }) { if (!rows.length) return <p className="text-sm text-slate-400">No trades</p>; const cols = Object.keys(rows[0]); return <div className="overflow-auto"><table className="text-xs"><thead><tr>{cols.map((c) => <th key={c} className="px-2 text-left">{c}</th>)}</tr></thead><tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c} className="px-2">{String(r[c] ?? "--")}</td>)}</tr>)}</tbody></table></div>; }
function Spark({ data, color }: { data: number[]; color: string }) { if (!data.length) return <p className="text-sm text-slate-400">No data</p>; const min = Math.min(...data); const max = Math.max(...data); const range = max - min || 1; const points = data.map((v, i) => `${(i / Math.max(1, data.length - 1)) * 100},${100 - ((v - min) / range) * 100}`).join(" "); return <svg viewBox="0 0 100 100" className="h-40 w-full"><polyline points={points} fill="none" stroke={color} strokeWidth="2" /></svg>; }
function Bars({ data }: { data: Record<string, number> }) { const entries = Object.entries(data); if (!entries.length) return <p className="text-sm text-slate-400">No data</p>; const max = Math.max(...entries.map(([, v]) => v), 1); return <div className="space-y-2 text-xs">{entries.map(([label, value]) => <div key={label}><div className="mb-1 flex justify-between"><span>{label}</span><span>{value}</span></div><div className="h-2 rounded bg-slate-800"><div className="h-2 rounded bg-cyan-500" style={{ width: `${(value / max) * 100}%` }} /></div></div>)}</div>; }

const num = (v: number | undefined | null) => (v == null || Number.isNaN(v) ? "--" : v.toFixed(2));
const fmt = (v: number | string | boolean | undefined, usd = false) => (typeof v === "number" ? `${usd ? "$" : ""}${v.toFixed(2)}` : String(v ?? "--"));
const pct = (v: number | undefined | null) => (v == null || Number.isNaN(v) ? "--" : `${(v * 100).toFixed(2)}%`);

function toDrawdownCurve(equity: number[]): number[] {
  let peak = Number.NEGATIVE_INFINITY;
  return equity.map((value) => {
    peak = Math.max(peak, value);
    return peak > 0 ? ((peak - value) / peak) * 100 : 0;
  });
}
