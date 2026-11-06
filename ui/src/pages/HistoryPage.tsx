import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, Crosshair, Newspaper, Search, Target, Wallet, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { get, History, money, pct, share, Status, tone, when } from "../api";
import ArticlePanel from "../components/ArticlePanel";
import { Card, Empty, Pill, Skeleton, Source, Stat } from "../components/ui";

type Show = "all" | "traded" | "waiting";

export default function HistoryPage({ status }: { status: Status | null }) {
  const [h, setH] = useState<History | null>(null);
  const [q, setQ] = useState("");
  const [show, setShow] = useState<Show>("all");
  const [open, setOpen] = useState<string | null>(null);
  const version = status?.version ?? "";
  useEffect(() => { get<History>("/api/history").then(setH).catch(() => {}); }, [version]);

  const rows = useMemo(() => (h?.articles ?? []).filter((a) =>
    (show === "all" || (show === "traded" ? (a.buys ?? 0) > 0 : a.status !== "done")) &&
    (!q || (a.title + " " + a.source).toLowerCase().includes(q.toLowerCase()))), [h, q, show]);
  const c = h?.card;

  return (
    <div className="relative flex h-full min-h-0 flex-col gap-4">
      <div className="grid grid-cols-5 gap-4">
        <Stat label="Articles predicted" icon={<Newspaper size={14} />} value={c ? c.articles.toLocaleString() : "—"} note={c ? `${c.pending} waiting or failed` : ""} />
        <Stat label="Up/down calls right" icon={<Crosshair size={14} />} value={share(c?.calls_right, 1)} note={c?.calls ? `of ${c.calls.toLocaleString()} calls` : "no calls yet"} />
        <Stat label="Buys that beat SPY" icon={<Target size={14} />} value={share(c?.buys_right, 1)} note={c?.buys ? `of ${c.buys.toLocaleString()} buys` : "no buys yet"} />
        <Stat label="Avg buy vs SPY" icon={<BarChart3 size={14} />} tone={tone(c?.avg_buy_move)} value={pct(c?.avg_buy_move, 3)}
          note={c?.avg_all_move != null ? `all stocks ${pct(c.avg_all_move, 3)}` : ""} />
        <Stat label="Pretend P&L" icon={<Wallet size={14} />} tone={tone(c?.pnl)} value={money(c?.pnl)}
          note={`after ${status?.settings.slippage_pct ?? 0.05}% slippage each way`} />
      </div>

      <Card className="min-h-0 flex-1" pad={false}
        title="Every article"
        right={
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg bg-white/[0.04] p-0.5 ring-1 ring-white/10">
              {(["all", "traded", "waiting"] as Show[]).map((s) => (
                <button key={s} onClick={() => setShow(s)}
                  className={`relative rounded-md px-3 py-1 text-[12px] font-medium capitalize transition ${show === s ? "text-zinc-50" : "text-zinc-500 hover:text-zinc-300"}`}>
                  {show === s && <motion.span layoutId="hist-seg" className="absolute inset-0 rounded-md bg-white/10" />}
                  <span className="relative">{s}</span>
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 rounded-lg bg-white/[0.04] px-2.5 py-1.5 ring-1 ring-white/10 focus-within:ring-indigo-400/50">
              <Search size={13} className="text-zinc-500" />
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search headlines"
                className="w-52 bg-transparent text-[12.5px] text-zinc-100 outline-none placeholder:text-zinc-600" />
            </label>
          </div>
        }>
        <div className="grid grid-cols-[130px_150px_1fr_90px_70px_90px_90px_100px] gap-x-4 border-b hairline px-5 pb-2 text-[10.5px] font-semibold tracking-wider text-zinc-500 uppercase">
          <span>Published</span><span>Source</span><span>Headline</span><span>Status</span>
          <span className="text-right">Buys</span><span className="text-right">Beat SPY</span><span className="text-right">Avg buy</span><span className="text-right">P&amp;L</span>
        </div>
        <div className="scroll min-h-0 flex-1 px-2 py-1.5">
          {!h ? <div className="p-3"><Skeleton rows={10} /></div>
            : !rows.length ? <Empty icon={<Newspaper size={20} />} title="No articles yet" text="Predicted articles show up here with their pretend buys." />
            : rows.map((a) => (
              <button key={a.uri} onClick={() => setOpen(a.uri)}
                className={`grid w-full grid-cols-[130px_150px_1fr_90px_70px_90px_90px_100px] items-center gap-x-4 rounded-xl px-3 py-2.5 text-left text-[12.5px] transition hover:bg-white/[0.04] ${open === a.uri ? "bg-white/[0.06]" : ""}`}>
                <span className="text-zinc-400 tabular-nums">{when(a.published)}</span>
                <Source name={a.source} />
                <span className="truncate text-zinc-100">{a.title}</span>
                <span>{a.status === "done" ? <Pill tone="accent">predicted</Pill> : a.status === "error" ? <Pill tone="down">failed</Pill> : <Pill tone="amber">{a.status}</Pill>}</span>
                <span className="text-right text-zinc-300 tabular-nums">{a.status === "done" ? a.buys ?? 0 : ""}</span>
                <span className="text-right text-zinc-300 tabular-nums">{a.buys ? `${a.right}/${a.buys}` : ""}</span>
                <span className={`text-right tabular-nums ${tone(a.avg)}`}>{a.avg != null ? pct(a.avg) : ""}</span>
                <span className={`text-right font-medium tabular-nums ${tone(a.pnl)}`}>{a.pnl != null ? money(a.pnl) : ""}</span>
              </button>
            ))}
        </div>
      </Card>

      <AnimatePresence>
        {open && <>
          <motion.div key="shade" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={() => setOpen(null)} className="fixed inset-0 z-20 bg-black/50 backdrop-blur-[2px]" />
          <motion.div key="drawer" initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 320, damping: 34 }}
            className="fixed top-3 right-3 bottom-3 z-30 flex w-[min(760px,62vw)] flex-col">
            <button onClick={() => setOpen(null)}
              className="absolute top-3 -left-12 grid size-9 place-items-center rounded-full bg-zinc-900/90 text-zinc-300 ring-1 ring-white/10 transition hover:text-white">
              <X size={16} />
            </button>
            <ArticlePanel uri={open} version={version} kicker="Article" className="h-full bg-ink-900/95" />
          </motion.div>
        </>}
      </AnimatePresence>
    </div>
  );
}
