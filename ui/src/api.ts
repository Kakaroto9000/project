// Talks to the local Python server (src/newsmodel/live/server.py), which the window starts.
export const API = import.meta.env.VITE_API ?? "http://127.0.0.1:8765";

export type Call = "up" | "down" | "flat";

export interface Status {
  engine: string;
  news_checked: string | null;
  counts: Record<string, number>;
  next_ready: string | null;
  version: string;
  model: string;
  sources: string[];
  settings: {
    start_cash: number; trade_dollars: number; hold_minutes: number;
    slippage_pct: number; min_up_prob: number; max_late_minutes: number;
  };
}

export interface Incoming { uri: string; published: string; source: string; title: string; ready: string | null }

export interface Now {
  incoming: Incoming[];
  latest: string | null;
  previous: string | null;
  account: number; total_pnl: number; today_pnl: number; today_trades: number; today_hit: number | null;
  spark: [number, number][];
}

export interface Article {
  uri: string; title: string; body: string; source: string; url: string;
  published: string; received: string; entry: string | null; exit: string | null; ready: string | null;
  status: string; note: string | null;
}

export interface Prediction {
  ticker: string; p_down: number; p_flat: number; p_up: number; buy: number; call: Call;
  entry_price: number | null; exit_price: number | null; raw_move: number | null; move: number | null;
  threshold: number | null; actual: Call | null;
}

export interface ArticleData {
  article: Article;
  predictions: Prediction[];
  trades: { ticker: string; buy_price: number; sell_price: number; qty: number; pnl: number; pnl_pct: number }[];
  summary: { buys: number; pnl: number; buys_right: number; buys_known: number; calls_right: number; calls_known: number } | null;
}

export interface HistoryRow {
  uri: string; published: string; source: string; title: string; status: string; note: string | null;
  buys: number | null; right: number | null; avg: number | null; pnl: number | null;
}

export interface History {
  articles: HistoryRow[];
  card: {
    articles: number; pending: number; calls?: number; calls_right?: number | null; buys?: number;
    buys_right?: number | null; avg_buy_move?: number | null; avg_all_move?: number; pnl: number;
  };
}

export interface Trade {
  id: number; uri: string; ticker: string; title: string; source: string; p_up: number;
  buy_time: string; buy_price: number; sell_time: string; sell_price: number; qty: number;
  pnl: number; pnl_pct: number; actual: Call | null;
}

export interface Portfolio {
  stats: {
    value: number; total_pnl: number; today_pnl: number; today_trades: number; trades: number;
    winners: number; avg_pct: number; avg_pnl: number;
    best: { ticker: string; pnl: number; pnl_pct: number }; worst: { ticker: string; pnl: number; pnl_pct: number };
  } | null;
  equity: [number, number][];
  trades: Trade[];
  by_ticker?: { ticker: string; trades: number; pnl: number }[];
}

export interface LogLine { time: string; message: string }

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const post = (path: string) => fetch(API + path, { method: "POST" });

/** Opens a link in the normal browser (inside the Tauri window that needs the opener plugin). */
export async function openLink(url: string) {
  if ("__TAURI_INTERNALS__" in window) {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } else window.open(url, "_blank");
}

// --- Formatting (times in New York, where the market is) ---
const TZ = "America/New_York";
const fmt = (opts: Intl.DateTimeFormatOptions) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, ...opts });
const dayTime = fmt({ month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
const timeOnly = fmt({ hour: "numeric", minute: "2-digit" });
const timeSec = fmt({ hour: "numeric", minute: "2-digit", second: "2-digit" });

export const when = (iso?: string | null) => (iso ? dayTime.format(new Date(iso)) : "");
export const clock = (iso?: string | null) => (iso ? timeOnly.format(new Date(iso)) : "");
export const clockSec = (iso?: string | null) => (iso ? timeSec.format(new Date(iso)) : "");

export function ago(iso?: string | null, now = Date.now()): string {
  if (!iso) return "";
  const m = Math.round((now - new Date(iso).getTime()) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  return h < 24 ? `${h} h ago` : `${Math.floor(h / 24)} d ago`;
}

export function until(iso?: string | null, now = Date.now()): string {
  if (!iso) return "after the open";
  const s = Math.round((new Date(iso).getTime() - now) / 1000);
  if (s <= 0) return "any moment";
  const m = Math.floor(s / 60);
  if (m < 60) return `in ${m}:${String(s % 60).padStart(2, "0")}`;
  return `in ${Math.floor(m / 60)} h ${m % 60} min`;
}

export const money = (v?: number | null, sign = true) =>
  v == null ? "—" : `${v < 0 ? "−" : sign ? "+" : ""}$${Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
export const dollars = (v?: number | null) =>
  v == null ? "—" : `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
export const pct = (v?: number | null, digits = 2) => (v == null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}%`);
export const share = (v?: number | null, digits = 0) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
export const tone = (v?: number | null) => (v == null || v === 0 ? "text-zinc-300" : v > 0 ? "text-emerald-400" : "text-rose-400");

// Lightweight-charts draws times as UTC: shift each point to New York wall-clock time.
export function nyTime(ms: number): number {
  const d = new Date(ms);
  const ny = new Date(d.toLocaleString("en-US", { timeZone: TZ }));
  const utc = new Date(d.toLocaleString("en-US", { timeZone: "UTC" }));
  return Math.round((ms + (ny.getTime() - utc.getTime())) / 1000);
}
