import { AnimatePresence, motion } from "framer-motion";
import { Check, ExternalLink, Hourglass, Newspaper, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ArticleData, clock, get, money, openLink, pct, Prediction, when } from "../api";
import { CallBadge, Card, Empty, MoveBar, Pill, ProbBar, Skeleton, Source, Ticker } from "./ui";

type Filter = "all" | "bought" | "up" | "down";

export default function ArticlePanel({ uri, version, kicker, className = "" }: {
  uri: string | null; version: string; kicker: string; className?: string;
}) {
  const [data, setData] = useState<ArticleData | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!uri) { setData(null); return; }
    let live = true;
    get<ArticleData>(`/api/article?uri=${encodeURIComponent(uri)}`).then((d) => live && setData(d)).catch(() => {});
    return () => { live = false; };
  }, [uri, version]);
  useEffect(() => { setOpen(false); }, [uri]);

  const bought = useMemo(() => new Set(data?.trades.map((t) => t.ticker)), [data]);
  const rows = useMemo(() => (data?.predictions ?? []).filter((p) =>
    filter === "all" || (filter === "bought" ? bought.has(p.ticker) : p.call === filter)), [data, filter, bought]);
  const scale = useMemo(() => Math.max(0.5, ...(data?.predictions ?? []).map((p) => Math.abs(p.move ?? 0))), [data]);

  if (!uri) return (
    <Card className={className}>
      <Empty icon={<Newspaper size={20} />} title="Nothing predicted yet"
        text="Each article is predicted once its 30-minute window is over and the free, 15-minute-delayed prices cover it." />
    </Card>
  );
  if (!data || data.article.uri !== uri) return <Card className={className}><div className="pt-5"><Skeleton rows={8} /></div></Card>;

  const a = data.article, s = data.summary;
  return (
    <Card className={className} pad={false}>
      <div className="px-5 pt-4">
        <div className="flex items-center justify-between gap-3">
          <span className="bg-gradient-to-r from-indigo-300 to-teal-200 bg-clip-text text-[11px] font-bold tracking-[0.14em] text-transparent uppercase">{kicker}</span>
          {a.url && (
            <button onClick={() => openLink(a.url)}
              className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] text-zinc-400 transition hover:bg-white/5 hover:text-zinc-100">
              Open article <ExternalLink size={13} />
            </button>
          )}
        </div>
        <h3 className="mt-2 text-[19px] leading-snug font-semibold tracking-tight text-zinc-50 select-text">{a.title}</h3>
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-zinc-500">
          <Source name={a.source} />
          <span>published {when(a.published)}</span>
          {a.entry && <span>· pretend trade {clock(a.entry)} → {clock(a.exit)} ET</span>}
        </div>
        {a.body && (
          <button onClick={() => setOpen(!open)} className="mt-2 w-full text-left">
            <p className={`text-[12.5px] leading-relaxed text-zinc-400 select-text ${open ? "" : "line-clamp-2"}`}>{a.body}</p>
          </button>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {s ? <>
            <Pill tone="accent">{s.buys} pretend buys</Pill>
            <Pill tone={s.pnl > 0 ? "up" : s.pnl < 0 ? "down" : "zinc"}>P&amp;L {money(s.pnl)}</Pill>
            <Pill>buys beat SPY {s.buys_right}/{s.buys_known}</Pill>
            <Pill>up/down calls right {s.calls_right}/{s.calls_known}</Pill>
          </> : <Pill tone="amber"><Hourglass size={12} /> {a.status === "waiting" ? `predicted at ${clock(a.ready)} ET` : a.status}</Pill>}
          {a.note && <Pill tone="amber">{a.note}</Pill>}
        </div>
      </div>

      {data.predictions.length > 0 && <>
        <div className="mt-3 flex items-center gap-1 border-b hairline px-5">
          {(["all", "bought", "up", "down"] as Filter[]).map((f) => (
            <button key={f} onClick={() => setFilter(f)}
              className={`relative px-3 pt-1 pb-2.5 text-[12.5px] font-medium capitalize transition ${filter === f ? "text-zinc-50" : "text-zinc-500 hover:text-zinc-300"}`}>
              {f === "up" ? "up calls" : f === "down" ? "down calls" : f}
              {filter === f && <motion.div layoutId={`tab-${kicker}`} className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-gradient-to-r from-indigo-400 to-teal-300" />}
            </button>
          ))}
          <span className="ml-auto text-[11.5px] text-zinc-600">{rows.length} stocks</span>
        </div>
        <div className="grid grid-cols-[minmax(92px,1fr)_minmax(84px,1.3fr)_62px_minmax(96px,1.5fr)_18px] items-center gap-x-3 px-5 py-2 text-[10.5px] font-semibold tracking-wider text-zinc-500 uppercase">
          <span>Stock</span><span>Model</span><span>Call</span><span>What happened vs SPY</span><span />
        </div>
        <div className="scroll min-h-0 flex-1 px-2 pb-3">
          <AnimatePresence initial={false}>
            {rows.map((p, i) => <Row key={p.ticker} p={p} bought={bought.has(p.ticker)} scale={scale} i={i} />)}
          </AnimatePresence>
        </div>
      </>}
    </Card>
  );
}

function Row({ p, bought, scale, i }: { p: Prediction; bought: boolean; scale: number; i: number }) {
  const right = p.actual == null || p.call === "flat" ? null : p.call === p.actual;
  const best = p.call === "up" ? p.p_up : p.call === "down" ? p.p_down : p.p_flat;
  return (
    <motion.div layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2, delay: Math.min(i, 20) * 0.01 }}
      className={`grid grid-cols-[minmax(92px,1fr)_minmax(84px,1.3fr)_62px_minmax(96px,1.5fr)_18px] items-center gap-x-3 rounded-xl px-3 py-2 text-[12.5px] transition hover:bg-white/[0.03] ${bought ? "bg-emerald-400/[0.035]" : ""}`}>
      <div className="flex items-center gap-2">
        <Ticker t={p.ticker} bought={bought} />
        {bought && <span className="text-[9.5px] font-bold tracking-wider text-emerald-300">BUY</span>}
      </div>
      <div className="flex flex-col gap-1">
        <ProbBar down={p.p_down} flat={p.p_flat} up={p.p_up} />
        <span className="text-[11px] text-zinc-500 tabular-nums">{p.call} {(best * 100).toFixed(0)}%</span>
      </div>
      <CallBadge call={p.call} />
      <div className="flex flex-col gap-1">
        <MoveBar v={p.move} scale={scale} />
        <span title={`the stock itself: ${pct(p.raw_move)}`}
          className={`truncate text-[11px] tabular-nums ${p.move == null ? "text-zinc-600" : p.move > 0 ? "text-emerald-400" : p.move < 0 ? "text-rose-400" : "text-zinc-400"}`}>
          {p.actual ? `${pct(p.move)} · ${p.actual}` : "no price"}
        </span>
      </div>
      <span className="grid place-items-center">
        {right === true && <Check size={15} className="text-emerald-400" />}
        {right === false && <X size={15} className="text-rose-400/80" />}
      </span>
    </motion.div>
  );
}
