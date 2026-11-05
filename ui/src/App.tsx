import { AnimatePresence, motion } from "framer-motion";
import { Activity, History as HistoryIcon, LayoutDashboard, RefreshCw, ScrollText, Wallet } from "lucide-react";
import { useEffect, useState } from "react";
import { clockSec, get, post, Status, until } from "./api";
import HistoryPage from "./pages/HistoryPage";
import LogPage from "./pages/LogPage";
import NowPage from "./pages/NowPage";
import PortfolioPage from "./pages/PortfolioPage";

type Page = "now" | "history" | "portfolio" | "log";
const NAV: { id: Page; label: string; icon: typeof Activity }[] = [
  { id: "now", label: "Now", icon: LayoutDashboard },
  { id: "history", label: "History", icon: HistoryIcon },
  { id: "portfolio", label: "Portfolio", icon: Wallet },
  { id: "log", label: "Log", icon: ScrollText },
];
const TITLES: Record<Page, [string, string]> = {
  now: ["Now", "Fresh news, the model’s latest calls, and how the one before played out"],
  history: ["History", "Every article the model has read, and how its pretend buys went"],
  portfolio: ["Portfolio", "The pretend account: $1,000 per “up” call, sold 30 minutes later"],
  log: ["Log", "What the engine has been doing"],
};

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export default function App() {
  const [page, setPage] = useState<Page>(() => (NAV.some((n) => "#" + n.id === location.hash) ? location.hash.slice(1) as Page : "now"));
  const [status, setStatus] = useState<Status | null>(null);
  const [offline, setOffline] = useState(false);
  const [tick, setTick] = useState(Date.now());
  const [spin, setSpin] = useState(false);

  // Poll the server; pages refetch when status.version changes
  useEffect(() => {
    let live = true;
    const load = () => get<Status>("/api/status")
      .then((s) => { if (live) { setStatus(s); setOffline(false); } })
      .catch(() => { if (live) setOffline(true); });
    load();
    const a = setInterval(load, 3000), b = setInterval(() => setTick(Date.now()), 1000);
    return () => { live = false; clearInterval(a); clearInterval(b); };
  }, []);

  const c = status?.counts ?? {};
  const running = !!status && /running|predicting|checking/.test(status.engine);
  const check = () => { setSpin(true); post("/api/check").finally(() => setTimeout(() => setSpin(false), 1200)); };

  return (
    <div className="relative flex h-full">
      <div className="aurora" /><div className="grain" />

      <aside className="relative z-10 flex w-[232px] shrink-0 flex-col border-r hairline bg-black/20 px-3 py-5 backdrop-blur-xl">
        <div className="flex items-center gap-2.5 px-2.5">
          <div className="grid size-9 place-items-center rounded-xl bg-gradient-to-br from-indigo-400 via-violet-400 to-teal-300 shadow-lg shadow-indigo-500/20">
            <Activity size={18} className="text-ink-950" strokeWidth={2.5} />
          </div>
          <div>
            <div className="text-[14.5px] font-semibold tracking-tight text-zinc-50">News Trader</div>
            <div className="text-[11px] text-zinc-500">pretend trading</div>
          </div>
        </div>

        <nav className="mt-8 flex flex-col gap-1">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setPage(id)}
              className={`relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[13.5px] font-medium transition ${page === id ? "text-zinc-50" : "text-zinc-400 hover:bg-white/[0.03] hover:text-zinc-200"}`}>
              {page === id && <motion.span layoutId="nav" className="absolute inset-0 rounded-xl bg-white/[0.07] ring-1 ring-white/10"
                transition={{ type: "spring", stiffness: 400, damping: 34 }} />}
              <Icon size={17} className="relative" />
              <span className="relative">{label}</span>
              {id === "now" && (c.waiting ?? 0) > 0 && (
                <span className="relative ml-auto rounded-full bg-indigo-400/15 px-2 text-[11px] font-semibold text-indigo-200 tabular-nums">{c.waiting}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="mt-auto flex flex-col gap-3 px-1">
          <div className="glass p-3.5">
            <div className="flex items-center gap-2 text-[12px] font-medium">
              <span className="relative flex size-2">
                {running && <span className="ping-soft absolute inline-flex size-full rounded-full bg-emerald-400" />}
                <span className={`relative inline-flex size-2 rounded-full ${offline ? "bg-rose-400" : running ? "bg-emerald-400" : "bg-amber-400"}`} />
              </span>
              <span className="truncate text-zinc-200">{offline ? "Engine offline" : status ? capital(status.engine) : "Connecting…"}</span>
            </div>
            <div className="mt-2 space-y-1 text-[11.5px] text-zinc-500">
              <div className="flex justify-between"><span>News checked</span><span className="text-zinc-400 tabular-nums">{clockSec(status?.news_checked) || "—"}</span></div>
              <div className="flex justify-between"><span>Next prediction</span><span className="text-zinc-400 tabular-nums">{status?.next_ready ? until(status.next_ready, tick) : "—"}</span></div>
              <div className="flex justify-between"><span>Predicted</span><span className="text-zinc-400 tabular-nums">{c.done ?? 0}</span></div>
              <div className="flex justify-between"><span>Failed</span><span className="text-zinc-400 tabular-nums">{c.error ?? 0}</span></div>
            </div>
          </div>
          <div className="px-1.5 text-[10.5px] leading-relaxed text-zinc-600">
            Model <span className="text-zinc-400">{status?.model ?? "—"}</span><br />
            No real orders are ever placed.
          </div>
        </div>
      </aside>

      <main className="relative z-10 flex min-w-0 flex-1 flex-col px-7 pt-6 pb-6">
        <header className="mb-5 flex items-end justify-between gap-4">
          <AnimatePresence mode="wait">
            <motion.div key={page} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
              <h1 className="text-[26px] font-semibold tracking-tight text-zinc-50">{TITLES[page][0]}</h1>
              <p className="mt-0.5 text-[13px] text-zinc-500">{TITLES[page][1]}</p>
            </motion.div>
          </AnimatePresence>
          <button onClick={check} title="The engine checks every minute anyway; this checks right now"
            className="group flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-2 text-[13px] font-semibold text-white shadow-lg shadow-indigo-500/25 transition hover:brightness-110 active:scale-[0.98]">
            <RefreshCw size={15} className={spin ? "animate-spin" : "transition group-hover:rotate-45"} />
            Check news now
          </button>
        </header>

        {offline && (
          <div className="mb-4 rounded-xl bg-rose-500/10 px-4 py-3 text-[13px] text-rose-200 ring-1 ring-rose-400/20">
            Can’t reach the engine at 127.0.0.1:8765. It starts with this window; if this stays, see data/live/server_output.log.
          </div>
        )}

        <AnimatePresence mode="wait">
          <motion.div key={page} className="min-h-0 flex-1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
            {page === "now" && <NowPage status={status} tick={tick} />}
            {page === "history" && <HistoryPage status={status} />}
            {page === "portfolio" && <PortfolioPage status={status} />}
            {page === "log" && <LogPage status={status} />}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
