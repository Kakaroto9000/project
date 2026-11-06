import { Activity, Award, CalendarDays, LineChart, Percent, Receipt, Wallet } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { dollars, get, money, pct, Portfolio, share, Status, tone, when } from "../api";
import AreaChart from "../components/AreaChart";
import { CallBadge, Card, Counter, Empty, Stat, Ticker } from "../components/ui";

const PAGE = 100;

export default function PortfolioPage({ status }: { status: Status | null }) {
  const [p, setP] = useState<Portfolio | null>(null);
  const [page, setPage] = useState(0);
  const version = status?.version ?? "";
  useEffect(() => { get<Portfolio>("/api/portfolio").then(setP).catch(() => {}); }, [version]);
  const s = p?.stats;
  const start = status?.settings.start_cash ?? 100000;
  const trades = useMemo(() => (p?.trades ?? []).slice(page * PAGE, (page + 1) * PAGE), [p, page]);
  const pages = Math.max(1, Math.ceil((p?.trades.length ?? 0) / PAGE));
  const maxAbs = Math.max(1, ...(p?.by_ticker ?? []).map((t) => Math.abs(t.pnl)));

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="grid grid-cols-6 gap-4">
        <Stat label="Value" icon={<Wallet size={14} />} tone={tone(s?.total_pnl)}
          value={s ? <Counter value={s.value} format={dollars} /> : dollars(start)} note={`started at ${dollars(start)}`} />
        <Stat label="Total P&L" icon={<LineChart size={14} />} tone={tone(s?.total_pnl)} value={money(s?.total_pnl)}
          note={s ? `${pct((s.total_pnl / start) * 100, 3)} of the account` : "no trades yet"} />
        <Stat label="Today" icon={<CalendarDays size={14} />} tone={tone(s?.today_pnl)} value={money(s?.today_pnl)} note={s ? `${s.today_trades} trades` : ""} />
        <Stat label="Trades" icon={<Receipt size={14} />} value={s ? s.trades.toLocaleString() : "—"}
          note={s ? `${dollars(s.trades * (status?.settings.trade_dollars ?? 1000))} traded` : ""} />
        <Stat label="Winners" icon={<Award size={14} />} value={share(s?.winners, 1)} note="made money after slippage" />
        <Stat label="Average trade" icon={<Percent size={14} />} tone={tone(s?.avg_pct)} value={pct(s?.avg_pct, 3)}
          note={s ? `${money(s.avg_pnl)} per trade` : ""} />
      </div>

      <div className="grid min-h-0 grid-cols-[1fr_340px] gap-4" style={{ flex: "0 0 42%" }}>
        <Card title="Pretend account value"
          right={<span className="text-[11.5px] text-zinc-500">
            {dollars(status?.settings.trade_dollars)} per “up” call · held {status?.settings.hold_minutes} min · {status?.settings.slippage_pct}% slippage each way · no real orders
          </span>}>
          {p && p.equity.length > 1 ? <div className="min-h-0 flex-1"><AreaChart data={p.equity} base={start} height="fill" /></div>
            : <Empty icon={<Activity size={20} />} title="No trades yet" text="The chart starts with the first pretend trade." />}
        </Card>
        <Card title="Best and worst stocks" pad={false}>
          <div className="scroll min-h-0 flex-1 px-5 pb-4">
            {(p?.by_ticker ?? []).slice().reverse().map((t) => (
              <div key={t.ticker} className="flex items-center gap-3 py-1.5">
                <div className="w-16"><Ticker t={t.ticker} /></div>
                <div className="relative h-2 flex-1 rounded-full bg-white/5">
                  <div className="absolute top-[-3px] left-1/2 h-3.5 w-px bg-white/15" />
                  <div className={`absolute top-0 h-full rounded-full ${t.pnl >= 0 ? "bg-emerald-400/90" : "bg-rose-400/90"}`}
                    style={t.pnl >= 0 ? { left: "50%", width: `${(t.pnl / maxAbs) * 50}%` } : { right: "50%", width: `${(-t.pnl / maxAbs) * 50}%` }} />
                </div>
                <span className={`w-20 text-right text-[12px] font-medium tabular-nums ${tone(t.pnl)}`}>{money(t.pnl)}</span>
              </div>
            ))}
            {!p?.by_ticker?.length && <Empty icon={<Award size={20} />} title="Nothing yet" />}
          </div>
        </Card>
      </div>

      <Card className="min-h-0 flex-1" pad={false} title="Every pretend trade"
        right={pages > 1 && (
          <div className="flex items-center gap-2 text-[12px] text-zinc-400">
            <button disabled={page === 0} onClick={() => setPage(page - 1)} className="rounded-md px-2 py-1 hover:bg-white/5 disabled:opacity-30">←</button>
            <span className="tabular-nums">{page + 1} / {pages}</span>
            <button disabled={page >= pages - 1} onClick={() => setPage(page + 1)} className="rounded-md px-2 py-1 hover:bg-white/5 disabled:opacity-30">→</button>
          </div>
        )}>
        <div className="grid grid-cols-[120px_80px_1fr_64px_86px_86px_76px_80px_96px_76px] gap-x-3 border-b hairline px-5 pb-2 text-[10.5px] font-semibold tracking-wider text-zinc-500 uppercase">
          <span>Bought</span><span>Ticker</span><span>Headline</span><span className="text-right">P(up)</span><span className="text-right">Buy</span>
          <span className="text-right">Sell</span><span className="text-right">Shares</span><span>Actual</span><span className="text-right">P&amp;L</span><span className="text-right">%</span>
        </div>
        <div className="scroll min-h-0 flex-1 px-2 py-1.5">
          {trades.map((t) => (
            <div key={t.id} className="grid grid-cols-[120px_80px_1fr_64px_86px_86px_76px_80px_96px_76px] items-center gap-x-3 rounded-xl px-3 py-2 text-[12.5px] transition hover:bg-white/[0.03]">
              <span className="text-zinc-400 tabular-nums">{when(t.buy_time)}</span>
              <Ticker t={t.ticker} bought />
              <span className="truncate text-zinc-300">{t.title}</span>
              <span className="text-right text-zinc-300 tabular-nums">{(t.p_up * 100).toFixed(0)}%</span>
              <span className="text-right text-zinc-400 tabular-nums">{dollars(t.buy_price)}</span>
              <span className="text-right text-zinc-400 tabular-nums">{dollars(t.sell_price)}</span>
              <span className="text-right text-zinc-500 tabular-nums">{t.qty.toFixed(3)}</span>
              <CallBadge call={t.actual} />
              <span className={`text-right font-medium tabular-nums ${tone(t.pnl)}`}>{money(t.pnl)}</span>
              <span className={`text-right tabular-nums ${tone(t.pnl_pct)}`}>{pct(t.pnl_pct)}</span>
            </div>
          ))}
          {p && !p.trades.length && <Empty icon={<Receipt size={20} />} title="No pretend trades yet" text="Every “up” call becomes a $1,000 pretend buy, sold 30 minutes later." />}
        </div>
      </Card>
    </div>
  );
}
