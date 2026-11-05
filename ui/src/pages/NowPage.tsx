import { AnimatePresence, motion } from "framer-motion";
import { Clock3, Hourglass, Inbox, Target, TrendingUp, Wallet } from "lucide-react";
import { useEffect, useState } from "react";
import { ago, dollars, get, money, Now, share, Status, tone, until, when } from "../api";
import AreaChart from "../components/AreaChart";
import ArticlePanel from "../components/ArticlePanel";
import { Card, Counter, Empty, Source, Stat } from "../components/ui";

export default function NowPage({ status, tick }: { status: Status | null; tick: number }) {
  const [now, setNow] = useState<Now | null>(null);
  const version = status?.version ?? "";
  useEffect(() => { get<Now>("/api/now").then(setNow).catch(() => {}); }, [version]);
  const start = status?.settings.start_cash ?? 100000;

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="grid grid-cols-4 gap-4">
        <Stat label="Pretend account" icon={<Wallet size={14} />} tone={tone(now?.total_pnl)}
          value={now ? <Counter value={now.account} format={(v) => dollars(v)} /> : "—"}
          note={now ? `${money(now.total_pnl)} since start` : ""}>
          {now && now.spark.length > 1 && (
            <div className="pointer-events-none absolute right-0 bottom-0 h-14 w-40 opacity-80">
              <AreaChart data={now.spark} base={start} height="fill" mini />
            </div>
          )}
        </Stat>
        <Stat label="Today's P&L" icon={<TrendingUp size={14} />} tone={tone(now?.today_pnl)}
          value={now ? <Counter value={now.today_pnl} format={(v) => money(v)} /> : "—"}
          note={now ? `${now.today_trades} pretend trades today` : ""} />
        <Stat label="Today's buys that beat SPY" icon={<Target size={14} />}
          value={now?.today_hit != null ? share(now.today_hit) : "—"}
          note={now?.today_trades ? `of ${now.today_trades} buys` : "no buys yet today"} />
        <Stat label="Waiting for prices" icon={<Hourglass size={14} />}
          value={now ? now.incoming.length : "—"}
          note={status?.next_ready ? `next prediction ${until(status.next_ready, tick)}` : "nothing waiting"} />
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(320px,0.8fr)_1.6fr_1.6fr] gap-4">
        <Card title="Incoming news" className="min-h-0"
          right={<span className="text-[11.5px] whitespace-nowrap text-zinc-500">~45 min to prediction</span>} pad={false}>
          <div className="scroll min-h-0 flex-1 px-3 pb-3">
            {!now?.incoming.length ? (
              <Empty icon={<Inbox size={20} />} title="All caught up" text="New articles appear here and wait until the delayed prices cover their 30-minute window." />
            ) : (
              <AnimatePresence initial={false}>
                {now.incoming.map((a) => (
                  <motion.div key={a.uri} layout initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, height: 0 }}
                    className="group relative rounded-xl px-3 py-2.5 transition hover:bg-white/[0.03]">
                    <div className="flex items-center justify-between gap-2">
                      <Source name={a.source} />
                      <span className="text-[11px] text-zinc-500">{ago(a.published, tick)}</span>
                    </div>
                    <div className="mt-1 line-clamp-2 text-[13px] leading-snug text-zinc-200">{a.title}</div>
                    <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-zinc-500">
                      <Clock3 size={11} />
                      <span>{when(a.published)}</span>
                      <span className="ml-auto rounded-md bg-indigo-400/10 px-1.5 py-px font-medium text-indigo-200 tabular-nums">{until(a.ready, tick)}</span>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
          </div>
        </Card>
        <ArticlePanel uri={now?.latest ?? null} version={version} kicker="Latest prediction" className="min-h-0" />
        <ArticlePanel uri={now?.previous ?? null} version={version} kicker="News before it · predicted vs actual" className="min-h-0" />
      </div>
    </div>
  );
}
