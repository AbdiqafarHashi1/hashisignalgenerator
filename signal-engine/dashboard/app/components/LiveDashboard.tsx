"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useState } from "react";
import { API_BASE, EngineState, stateEventsUrl } from "../../lib/api";
import {
  fetchDashboardBundle,
  fetchEngineStateSafe,
  NormalizedSymbol,
  NormalizedTrade,
  triggerEngineAction,
} from "../../lib/dashboardClient";

const UI_REFRESH_MS = 3000;
const HEAVY_REFRESH_MS = 15000;

type Toast = { id: number; message: string; type: "success" | "error" };
type ViewMode = "professional" | "minimal";
type ResultFilter = "all" | "win" | "loss" | "breakeven";
type ControlKey = "start" | "stop" | "runOnce" | "forceRun";

const CONTROLS: Array<{
  key: ControlKey;
  label: string;
  path: string;
  activeWhenRunning: boolean;
  kind: "primary" | "secondary" | "danger";
}> = [
  { key: "start", label: "Start", path: "/start", activeWhenRunning: false, kind: "primary" },
  { key: "stop", label: "Stop", path: "/stop", activeWhenRunning: true, kind: "secondary" },
  { key: "runOnce", label: "Run Once", path: "/run", activeWhenRunning: true, kind: "secondary" },
  { key: "forceRun", label: "Force Run", path: "/run?force=true", activeWhenRunning: true, kind: "danger" },
];

export default function LiveDashboard() {
  const [state, setState] = useState<EngineState | null>(null);
  const [bundle, setBundle] = useState<Awaited<ReturnType<typeof fetchDashboardBundle>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("professional");
  const [selectedSymbol, setSelectedSymbol] = useState("ALL");
  const [resultFilter, setResultFilter] = useState<ResultFilter>("all");
  const [actionLoading, setActionLoading] = useState<ControlKey | null>(null);
  const [confirmForce, setConfirmForce] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const pushToast = useCallback((message: string, type: Toast["type"]) => {
    const id = Date.now() + Math.floor(Math.random() * 10000);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 3200);
  }, []);

  const loadOverview = useCallback(async () => {
    try {
      const payload = await fetchDashboardBundle();
      setBundle(payload);
      setError("");
    } catch (err) {
      setError((err as Error).message || "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshState = useCallback(async () => {
    const payload = await fetchEngineStateSafe();
    if (payload) setState(payload);
  }, []);

  const handleAction = useCallback(
    async (key: ControlKey, path: string) => {
      setActionLoading(key);
      try {
        const status = await triggerEngineAction(path);
        pushToast(`${key}: ${status}`, "success");
        await Promise.all([loadOverview(), refreshState()]);
      } catch (err) {
        pushToast((err as Error).message || "Action failed", "error");
      } finally {
        setActionLoading(null);
        setConfirmForce(false);
      }
    },
    [loadOverview, pushToast, refreshState],
  );

  useEffect(() => {
    loadOverview();
    refreshState();
    const timer = setInterval(loadOverview, HEAVY_REFRESH_MS);
    return () => clearInterval(timer);
  }, [loadOverview, refreshState]);

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
    const timer = setInterval(refreshState, UI_REFRESH_MS);
    return () => clearInterval(timer);
  }, [refreshState]);

  const account = bundle?.account ?? {};
  const risk = bundle?.risk ?? {};
  const activity = bundle?.activity ?? {};
  const symbols = bundle?.symbols ?? [];
  const trades = bundle?.trades ?? [];
  const running = Boolean(account.engine_running ?? state?.running ?? false);

  const symbolKeys = symbols.map((s) => s.symbol);
  const filteredTrades = trades
    .filter((row) => selectedSymbol === "ALL" || row.symbol === selectedSymbol)
    .filter((row) => resultFilter === "all" || row.result === resultFilter)
    .slice(0, 120);

  const ethCard = findSymbol(symbols, "ETHUSDT");
  const btcCard = findSymbol(symbols, "BTCUSDT");

  const pnlToday = num(account.realized_pnl_today ?? state?.realized_pnl_today_usd);
  const unrealizedPnl = num(account.unrealized_pnl ?? state?.unrealized_pnl_usd);
  const statusText = running ? "ACTIVE" : "STOPPED";
  const openPositions = Array.isArray(account.open_positions_detail) ? (account.open_positions_detail as Array<Record<string, unknown>>) : [];
  const openOrders = Array.isArray(account.open_orders) ? (account.open_orders as Array<Record<string, unknown>>) : [];
  const executions = Array.isArray(account.executions) ? (account.executions as Array<Record<string, unknown>>) : [];
  const eventTape = Array.isArray(account.event_tape) ? (account.event_tape as Array<Record<string, unknown>>) : [];

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#01040c] via-[#020712] to-[#090f1c] px-3 py-5 text-slate-100 md:px-6">
      <div className="mx-auto max-w-[1400px] space-y-4">
        <Card>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h1 className="text-3xl font-semibold tracking-tight">Signal Engine Dashboard</h1>
              <p className="mt-1 text-sm text-slate-400">API: {API_BASE}</p>
            </div>
            <div className="flex flex-col items-end gap-1">
              <div className="flex flex-wrap items-center gap-2">
                <StatusChip label="LIVE" tone="green" />
                <StatusChip label="PAPER" tone="blue" />
                <div className="inline-flex h-10 items-center rounded-full border border-slate-600/70 bg-slate-900/65 p-1 text-xs backdrop-blur-[2px]">
                  <button
                    type="button"
                    className={`btn rounded-full px-3 py-1 ${viewMode === "professional" ? "bg-cyan-500/20 text-cyan-100" : "text-slate-300"}`}
                    onClick={() => setViewMode("professional")}
                  >
                    Professional Mode
                  </button>
                  <button
                    type="button"
                    className={`btn rounded-full px-3 py-1 ${viewMode === "minimal" ? "bg-cyan-500/20 text-cyan-100" : "text-slate-300"}`}
                    onClick={() => setViewMode("minimal")}
                  >
                    Minimal Mode
                  </button>
                </div>
              </div>
              <p className="text-[11px] font-semibold tracking-[0.08em] text-slate-500">FOUNDER MODE</p>
            </div>
          </div>
        </Card>

        {error ? <p className="text-sm text-rose-300">{error}</p> : null}

        <section className="grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-7">
          <Kpi loading={loading} label="Equity" value={money(account.live_equity ?? state?.equity)} />
          <Kpi
            loading={loading}
            label="Unrealized PnL"
            value={money(unrealizedPnl)}
            valueClass={unrealizedPnl == null ? "" : unrealizedPnl < 0 ? "text-rose-300" : "text-emerald-300"}
          />
          <Kpi
            loading={loading}
            label="PnL Today"
            value={money(pnlToday)}
            valueClass={pnlToday == null ? "" : pnlToday < 0 ? "text-rose-300" : "text-emerald-300"}
          />
          <Kpi loading={loading} label="Daily DD %" value={pct(account.daily_drawdown_pct ?? state?.daily_loss_pct)} />
          <Kpi loading={loading} label="Global DD %" value={pct(account.global_drawdown_pct)} />
          <Kpi loading={loading} label="Trades Today" value={String(activity.trades_today ?? state?.trades_today ?? "—")} />
          <Kpi loading={loading} label="Status" value={statusText} />
        </section>

        <section>
          <Card className="p-3">
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              {CONTROLS.map((control) => {
                const loadingThis = actionLoading === control.key;
                const disabled =
                  loadingThis ||
                  (!!actionLoading && actionLoading !== control.key) ||
                  (running ? !control.activeWhenRunning : control.activeWhenRunning);
                return (
                  <button
                    key={control.key}
                    disabled={disabled}
                    className={`btn rounded-2xl border px-3 py-2.5 text-sm font-medium ${buttonTone(control.kind)} disabled:cursor-not-allowed disabled:opacity-50`}
                    onClick={() => {
                      if (control.key === "forceRun") {
                        setConfirmForce(true);
                        return;
                      }
                      handleAction(control.key, control.path);
                    }}
                  >
                    {loadingThis ? <span className="inline-flex items-center gap-2"><span className="h-3 w-3 animate-spin rounded-full border-2 border-current border-r-transparent" />Working...</span> : control.label}
                  </button>
                );
              })}
            </div>
          </Card>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Card title="Equity Curve">
            <Spark data={(bundle?.equitySeries ?? []).slice(-100)} color="#75e0cb" glow />
          </Card>
          <Card title="Drawdown Curve">
            <Spark data={toDrawdownCurve((bundle?.equitySeries ?? []).slice(-100))} color="#e3b75f" />
          </Card>
        </section>

        {viewMode === "professional" ? (
          <section className="grid gap-4 lg:grid-cols-4">
            <Card title="Risk Panel">
              <Metric label="Daily loss limit" value={pct(risk.daily_loss_limit_pct)} />
              <Metric label="Global DD limit" value={pct(risk.global_dd_limit_pct)} />
              <Metric label="Consecutive losses" value={fmt(risk.consecutive_losses)} />
              <Metric label="Cooldown remaining" value={`${fmt(risk.cooldown_remaining_seconds)}s`} />
              <Metric label="Max trades/day" value={`${fmt(risk.max_trades_per_day)} (used ${fmt(risk.trades_today)})`} />
            </Card>

            <Card title="Activity">
              <Metric label="Win rate" value={pct(activity.win_rate_today)} />
              <Metric label="Profit factor" value={fmt(activity.profit_factor)} />
              <Metric label="Expectancy" value={fmt(activity.expectancy)} />
              <Metric label="Avg win/loss" value={`${money(activity.avg_win)} / ${money(activity.avg_loss)}`} />
              <Metric label="Fees today/rolling" value={`${money(activity.fees_today)} / ${money(activity.fees_rolling)}`} />
              <Metric label="Equity reconcile Δ" value={money(account.equity_reconcile_delta)} />
            </Card>

            <SymbolCard symbol={ethCard} fallback="ETHUSDT" />
            <SymbolCard symbol={btcCard} fallback="BTCUSDT" />
          </section>
        ) : null}

        {viewMode === "professional" ? (
          <Card title="Config">
            <div className="grid gap-2 text-sm md:grid-cols-2">
              <Metric label="Engine mode" value={String(bundle?.debugConfig.effective?.engine_mode ?? "—")} />
              <Metric label="Strategy profile" value={String(bundle?.debugConfig.effective?.strategy_profile ?? "—")} />
              <Metric label="Account size" value={String(bundle?.debugConfig.effective?.account_size ?? "—")} />
              <Metric label="Min signal score" value={String(bundle?.debugConfig.effective?.min_signal_score ?? "—")} />
            </div>
          </Card>
        ) : null}


        <section className="grid gap-4 lg:grid-cols-2">
          <Card title="Open Positions">
            {openPositions.length ? (
              <div className="space-y-2 text-xs">
                {openPositions.map((row, idx) => (
                  <div key={`${String(row.id ?? idx)}`} className="rounded-lg border border-slate-700/60 p-2">
                    <p>{String(row.symbol ?? "—")} {String(row.side ?? "—")} qty {fmt(row.size)} @ {fmt(row.entry)}</p>
                    <p className="text-slate-300">mark {fmt(row.mark_price)} | uPnL {money(row.unrealized_pnl)} | R {fmt(row.unrealized_r)} | SL {fmt(row.stop)} | TP {fmt(row.take_profit)}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-400">No open positions</p>
            )}
          </Card>
          <Card title="Open Orders">
            {openOrders.length ? (
              <div className="space-y-2 text-xs">
                {openOrders.map((row, idx) => (
                  <div key={`${String(row.id ?? idx)}`} className="rounded-lg border border-slate-700/60 p-2">
                    <p>ID {String(row.id ?? "—")} | {String(row.symbol ?? "—")} {String(row.side ?? "—")}</p>
                    <p className="text-slate-300">price {fmt(row.price)} qty {fmt(row.qty)} status {String(row.status ?? "—")}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-400">No open orders</p>
            )}
          </Card>
        </section>

        <section className="grid gap-4 lg:grid-cols-2">
          <Card title="Executions">
            <div className="max-h-60 space-y-2 overflow-auto text-xs">
              {executions.length ? executions.slice(-50).reverse().map((row, idx) => (
                <div key={`${String(row.id ?? idx)}-${idx}`} className="rounded-lg border border-slate-700/60 p-2">
                  <p>{String(row.symbol ?? "—")} {String(row.side ?? "—")} @ {fmt(row.price)} x {fmt(row.qty)}</p>
                  <p className="text-slate-300">fee {money(row.fee)} | {String(row.status ?? "—")} | {formatTime(String(row.time ?? ""))}</p>
                </div>
              )) : <p className="text-sm text-slate-400">No executions</p>}
            </div>
          </Card>
          <Card title="Live Event Tape">
            <div className="max-h-60 space-y-2 overflow-auto text-xs">
              {eventTape.length ? eventTape.slice(-200).reverse().map((row, idx) => (
                <div key={`${String(row.correlation_id ?? idx)}-${idx}`} className="rounded-lg border border-slate-700/60 p-2">
                  <p>{formatTime(String(row.time ?? ""))} | {String(row.type ?? "event")}</p>
                  <p className="truncate text-slate-300">{JSON.stringify(row.payload ?? {})}</p>
                </div>
              )) : <p className="text-sm text-slate-400">No events</p>}
            </div>
          </Card>
        </section>

        <Card title="Trade Log" className="w-full p-6 lg:p-7">
          <div className="mb-4 grid gap-2 md:grid-cols-2">
            <select className="input" value={selectedSymbol} onChange={(e) => setSelectedSymbol(e.target.value)}>
              <option value="ALL">All symbols</option>
              {symbolKeys.map((symbol) => (
                <option key={symbol} value={symbol}>
                  {symbol}
                </option>
              ))}
            </select>
            <select className="input" value={resultFilter} onChange={(e) => setResultFilter(e.target.value as ResultFilter)}>
              <option value="all">All results</option>
              <option value="win">Win</option>
              <option value="loss">Loss</option>
              <option value="breakeven">BE</option>
            </select>
          </div>
          <TradeTable rows={filteredTrades} />
        </Card>
      </div>

      {confirmForce ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-4 backdrop-blur-sm">
          <Card className="w-full max-w-sm">
            <h3 className="text-lg font-semibold">Force Run</h3>
            <p className="mt-2 text-sm text-slate-300">Force a tick now?</p>
            <div className="mt-4 grid grid-cols-2 gap-2">
              <button className="btn rounded-2xl border border-slate-600/90 bg-slate-800/65 px-3 py-2" onClick={() => setConfirmForce(false)}>
                Cancel
              </button>
              <button
                className="btn rounded-2xl border border-rose-500/65 bg-rose-600/15 px-3 py-2 text-rose-100"
                onClick={() => handleAction("forceRun", "/run?force=true")}
              >
                Confirm
              </button>
            </div>
          </Card>
        </div>
      ) : null}

      <div className="fixed right-3 top-3 z-50 space-y-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`rounded-xl border px-3 py-2 text-sm shadow-[0_12px_28px_rgba(0,0,0,0.35)] backdrop-blur-sm ${
              toast.type === "success"
                ? "border-emerald-500/70 bg-emerald-950/95 text-emerald-200"
                : "border-rose-500/70 bg-rose-950/95 text-rose-200"
            }`}
          >
            {toast.message}
          </div>
        ))}
      </div>

      <style jsx>{`
        .btn {
          transition: transform 120ms ease, filter 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
        }
        .btn:hover {
          transform: translateY(-1px);
          filter: brightness(1.07);
          border-color: rgba(148, 163, 184, 0.75);
        }
        .btn:active {
          transform: translateY(0) scale(0.98);
          filter: brightness(0.99);
          box-shadow: inset 0 2px 6px rgba(2, 6, 23, 0.35);
        }
        .btn:disabled {
          filter: saturate(0.7) brightness(0.8);
          border-color: rgba(71, 85, 105, 0.55);
        }
        .btn:focus-visible,
        .input:focus-visible {
          outline: 2px solid rgba(56, 189, 248, 0.65);
          outline-offset: 2px;
        }
      `}</style>
    </div>
  );
}

function Card({ title, className = "", children }: { title?: string; className?: string; children: ReactNode }) {
  return (
    <div
      className={`rounded-2xl border border-slate-700/42 bg-[linear-gradient(160deg,rgba(255,255,255,0.032)_0%,rgba(15,23,42,0.93)_24%,rgba(15,23,42,0.9)_60%,rgba(30,41,59,0.8)_100%)] p-5 shadow-[0_24px_44px_rgba(2,6,23,0.44),0_1px_0_rgba(255,255,255,0.045)_inset] ring-1 ring-white/4 ${className}`}
    >
      {title ? <h3 className="mb-3 text-sm font-semibold text-slate-200">{title}</h3> : null}
      {children}
    </div>
  );
}

function Kpi({ loading, label, value, valueClass = "" }: { loading?: boolean; label: string; value: string; valueClass?: string }) {
  return (
    <Card className="p-4">
      {loading ? (
        <div className="h-11 animate-pulse rounded bg-slate-700/70" />
      ) : (
        <>
          <p className="text-[10px] text-slate-400/75">{label}</p>
          <p className={`mt-1 font-mono text-[2.7rem] font-semibold leading-none tabular-nums text-slate-50 text-right ${valueClass}`}>{value}</p>
        </>
      )}
    </Card>
  );
}

function StatusChip({ label, tone }: { label: string; tone: "green" | "blue" }) {
  const styles = {
    green: "border-emerald-300/55 bg-gradient-to-r from-emerald-500/20 to-emerald-500/8 text-emerald-100",
    blue: "border-cyan-300/55 bg-gradient-to-r from-cyan-500/20 to-cyan-500/8 text-cyan-100",
  };

  return (
    <span className={`inline-flex h-10 items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold backdrop-blur-[1px] shadow-[0_4px_10px_rgba(15,23,42,0.2)] ${styles[tone]}`}>
      <span className="text-[10px]">●</span>
      <span>{label}</span>
    </span>
  );
}

function SymbolCard({ symbol, fallback }: { symbol: NormalizedSymbol | null; fallback: string }) {
  const row = symbol ?? {
    symbol: fallback,
    regime: "—",
    regimeLabel: "—",
    lastDecision: "—",
    lastSkipReason: "—",
    atrPct: null,
    trendStrength: null,
    signalScore: null,
    unrealizedPnl: null,
    openPosition: "No open position",
  };

  return (
    <Card title={row.symbol}>
      <Metric label="Regime" value={row.regime} />
      <Metric label="Last decision" value={row.lastDecision} />
      <Metric label="Last skip reason" value={humanSkipReason(row.lastSkipReason)} />
      <Metric label="ATR %" value={pct(row.atrPct)} />
      <Metric label="Trend strength" value={fmt(row.trendStrength)} />
      <Metric label="Signal score" value={fmt(row.signalScore)} />
      <Metric label="Position" value={row.openPosition || "No open position"} />
    </Card>
  );
}

function TradeTable({ rows }: { rows: NormalizedTrade[] }) {
  const columns: Array<{ key: string; label: string; right?: boolean }> = [
    { key: "id", label: "ID", right: true },
    { key: "symbol", label: "Symbol" },
    { key: "side", label: "Side" },
    { key: "entry", label: "Entry", right: true },
    { key: "exit", label: "Exit", right: true },
    { key: "sl", label: "Stop (SL)", right: true },
    { key: "tp", label: "Take Profit (TP)", right: true },
    { key: "size", label: "Size", right: true },
    { key: "fees", label: "Fees", right: true },
    { key: "pnl", label: "PnL (USD)", right: true },
    { key: "result", label: "Result" },
    { key: "opened", label: "Opened", right: true },
    { key: "closed", label: "Closed", right: true },
    { key: "mode", label: "Mode" },
  ];

  return (
    <>
      <div className="hidden overflow-auto md:block">
        <table className="min-w-full text-[12px]">
          <thead>
            <tr className="text-slate-400">
              {columns.map((column) => (
                <th
                  key={column.key}
                  className={`px-2 py-1 ${column.right ? "text-right" : "text-left"}`}
                >
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((row) => (
                <tr key={row.id} className="border-t border-slate-700/50">
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{row.id || "—"}</td>
                  <td className="px-2 py-1">{row.symbol || "—"}</td>
                  <td className="px-2 py-1">
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${row.side.toLowerCase().includes("sell") ? "bg-orange-500/15 text-orange-200" : "bg-cyan-500/15 text-cyan-200"}`}>
                      {row.side || "—"}
                    </span>
                  </td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{fmt(row.entry)}</td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{fmt(row.exit)}</td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{fmt(row.sl)}</td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{fmt(row.tp)}</td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{fmt(row.size)}</td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{money(row.fee)}</td>
                  <td className={`px-2 py-1 text-right font-mono tabular-nums ${pnlClass(row.pnlUsd)}`}>{money(row.pnlUsd)}</td>
                  <td className="px-2 py-1">
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${resultClass(row.result)}`}>
                      {(row.resultLabel || row.result || "unknown").toString().toUpperCase()}
                    </span>
                  </td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{formatTime(row.opened)}</td>
                  <td className="px-2 py-1 text-right font-mono tabular-nums">{formatTime(row.closed)}</td>
                  <td className="px-2 py-1">{(row.mode || "paper").toUpperCase()}</td>
                </tr>
              ))
            ) : (
              <tr className="border-t border-slate-700/50">
                <td className="px-2 py-4 text-center text-sm text-slate-400" colSpan={columns.length}>
                  No trades yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="space-y-2 md:hidden">
        {rows.length ? (
          rows.map((row) => (
            <div key={row.id} className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-3.5 text-xs shadow-[0_8px_22px_rgba(0,0,0,0.3)]">
              <div className="flex items-center justify-between">
                <p className="font-semibold">{row.symbol || "—"}</p>
                <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${resultClass(row.result)}`}>
                  {(row.resultLabel || row.result || "unknown").toString().toUpperCase()}
                </span>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-1 font-mono tabular-nums">
                <p className="text-slate-400">ID</p><p className="text-right">{row.id || "—"}</p>
                <p className="text-slate-400">Side</p><p className="text-right">{row.side || "—"}</p>
                <p className="text-slate-400">Entry</p><p className="text-right">{fmt(row.entry)}</p>
                <p className="text-slate-400">Exit</p><p className="text-right">{fmt(row.exit)}</p>
                <p className="text-slate-400">Stop (SL)</p><p className="text-right">{fmt(row.sl)}</p>
                <p className="text-slate-400">Take Profit (TP)</p><p className="text-right">{fmt(row.tp)}</p>
                <p className="text-slate-400">Size</p><p className="text-right">{fmt(row.size)}</p>
                <p className="text-slate-400">Fees</p><p className="text-right">{money(row.fee)}</p>
                <p className="text-slate-400">PnL (USD)</p><p className={`text-right ${pnlClass(row.pnlUsd)}`}>{money(row.pnlUsd)}</p>
                <p className="text-slate-400">Opened</p><p className="text-right">{formatTime(row.opened)}</p>
                <p className="text-slate-400">Closed</p><p className="text-right">{formatTime(row.closed)}</p>
                <p className="text-slate-400">Mode</p><p className="text-right">{(row.mode || "paper").toUpperCase()}</p>
              </div>
            </div>
          ))
        ) : (
          <div className="rounded-xl border border-slate-700/80 bg-slate-900/70 p-3.5 text-xs text-slate-400">No trades yet</div>
        )}
      </div>
    </>
  );
}


function Metric({ label, value }: { label: string; value: string }) {
  return (
    <p className="flex items-center justify-between gap-2 py-0.5 text-sm">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono tabular-nums text-right">{value || "—"}</span>
    </p>
  );
}

function Spark({ data, color, glow = false }: { data: number[]; color: string; glow?: boolean }) {
  if (!data.length) {
    return (
      <div className="flex h-44 flex-col items-center justify-center rounded-xl border border-slate-700/80 bg-slate-950/72 text-xs text-slate-400">
        <span className="mb-1 text-sm">📉</span>
        <p>No data yet</p>
      </div>
    );
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((value, index) => `${(index / Math.max(1, data.length - 1)) * 100},${100 - ((value - min) / range) * 100}`).join(" ");

  return (
    <svg viewBox="0 0 100 100" className="h-44 w-full rounded-xl border border-slate-700/80 bg-slate-950/72">
      {[20, 40, 60, 80].map((line) => (
        <line key={line} x1="0" y1={line} x2="100" y2={line} stroke="rgba(148,163,184,0.16)" strokeWidth="0.5" />
      ))}
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.8" style={glow ? { filter: "drop-shadow(0 0 4px rgba(117,224,203,0.35))" } : undefined} />
    </svg>
  );
}

const num = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const fmt = (value: unknown): string => {
  const parsed = num(value);
  return parsed == null ? "—" : parsed.toFixed(2);
};

const money = (value: unknown): string => {
  const parsed = num(value);
  return parsed == null ? "—" : `${parsed < 0 ? "-$" : "$"}${Math.abs(parsed).toFixed(2)}`;
};

const pct = (value: unknown): string => {
  const parsed = num(value);
  return parsed == null ? "—" : `${(parsed * 100).toFixed(2)}%`;
};

const buttonTone = (kind: "primary" | "secondary" | "danger"): string => {
  if (kind === "primary") return "border-cyan-500/42 bg-cyan-500/8 text-cyan-100 shadow-[0_4px_10px_rgba(8,145,178,0.12)]";
  if (kind === "danger") return "border-rose-500/42 bg-rose-500/8 text-rose-100 shadow-[0_4px_10px_rgba(225,29,72,0.12)]";
  return "border-slate-600 bg-slate-800/70 text-slate-100";
};

const pnlClass = (value: unknown): string => {
  const parsed = num(value);
  if (parsed == null) return "text-slate-200";
  if (parsed > 0) return "text-emerald-200";
  if (parsed < 0) return "text-rose-200";
  return "text-amber-200";
};

const resultClass = (result: NormalizedTrade["result"]): string => {
  if (result === "win") return "bg-emerald-500/15 text-emerald-200";
  if (result === "loss") return "bg-rose-500/15 text-rose-200";
  if (result === "breakeven") return "bg-amber-500/15 text-amber-200";
  return "bg-slate-500/15 text-slate-200";
};

const formatTime = (value: string): string => {
  if (!value || value === "--") return "—";
  const stamp = Date.parse(value);
  if (Number.isNaN(stamp)) return value;
  return new Date(stamp).toLocaleString();
};

function findSymbol(symbols: NormalizedSymbol[], key: string): NormalizedSymbol | null {
  return symbols.find((row) => row.symbol.toUpperCase() === key) ?? null;
}

function toDrawdownCurve(equity: number[]): number[] {
  let peak = Number.NEGATIVE_INFINITY;
  return equity.map((value) => {
    peak = Math.max(peak, value);
    return peak > 0 ? ((peak - value) / peak) * 100 : 0;
  });
}

const SKIP_REASON_LABELS: Record<string, string> = {
  atr_too_low: "ATR too low",
  trend_too_weak: "Trend too weak",
  cooldown_active: "Cooldown active",
  funding_blackout: "Funding blackout",
  max_trades_reached: "Max trades reached",
  signal_too_weak: "Signal too weak",
  no_regime_match: "No regime match",
};

function humanSkipReason(raw: string): string {
  if (!raw || raw === "--") return "Unknown";
  return SKIP_REASON_LABELS[raw] ?? raw.replaceAll("_", " ");
}
