import { animate, motion, useMotionValue, useTransform } from "framer-motion";
import { ReactNode, useEffect } from "react";
import { Call } from "../api";

export function Card({ children, className = "", title, right, pad = true }: {
  children: ReactNode; className?: string; title?: ReactNode; right?: ReactNode; pad?: boolean;
}) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, ease: "easeOut" }}
      className={`glass relative flex min-h-0 flex-col ${className}`}>
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 px-5 pt-4 pb-3">
          <h2 className="text-[13px] font-semibold tracking-wide text-zinc-300">{title}</h2>
          {right}
        </header>
      )}
      <div className={`flex min-h-0 flex-1 flex-col ${pad ? "px-5 pb-5" : ""}`}>{children}</div>
    </motion.section>
  );
}

/** A number that glides to its new value. */
export function Counter({ value, format }: { value: number; format: (v: number) => string }) {
  const mv = useMotionValue(value);
  const text = useTransform(mv, (v) => format(v));
  useEffect(() => {
    const c = animate(mv, value, { duration: 0.8, ease: [0.16, 1, 0.3, 1] });
    return c.stop;
  }, [value]);
  return <motion.span>{text}</motion.span>;
}

export function Stat({ label, value, note, tone = "", icon, children }: {
  label: string; value: ReactNode; note?: ReactNode; tone?: string; icon?: ReactNode; children?: ReactNode;
}) {
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
      className="glass relative overflow-hidden px-5 py-4">
      <div className="flex items-center gap-2 text-[12px] font-medium text-zinc-400">
        {icon}<span>{label}</span>
      </div>
      <div className={`mt-1.5 text-[28px] leading-tight font-semibold tracking-tight tabular-nums ${tone || "text-zinc-50"}`}>{value}</div>
      {note && <div className="mt-0.5 text-[12px] text-zinc-500">{note}</div>}
      {children}
    </motion.div>
  );
}

export function Pill({ children, tone = "zinc" }: { children: ReactNode; tone?: "up" | "down" | "accent" | "zinc" | "amber" }) {
  const t = {
    up: "bg-emerald-400/10 text-emerald-300 ring-emerald-400/20",
    down: "bg-rose-400/10 text-rose-300 ring-rose-400/20",
    accent: "bg-indigo-400/10 text-indigo-200 ring-indigo-400/25",
    amber: "bg-amber-400/10 text-amber-200 ring-amber-400/20",
    zinc: "bg-white/[0.04] text-zinc-300 ring-white/10",
  }[tone];
  return <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11.5px] font-medium ring-1 ring-inset ${t}`}>{children}</span>;
}

const SOURCE_COLORS: Record<string, string> = {
  Benzinga: "from-sky-400 to-blue-500", "fiscal.ai": "from-slate-400 to-slate-500", CNBC: "from-amber-300 to-orange-500",
  Fortune: "from-rose-400 to-red-500", "Business Insider": "from-indigo-400 to-violet-500", TechCrunch: "from-emerald-300 to-green-500",
  "The Verge": "from-fuchsia-400 to-pink-500", OilPrice: "from-yellow-300 to-amber-500",
};

export function Source({ name }: { name: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px] font-medium text-zinc-300">
      <span className={`size-2 rounded-full bg-gradient-to-br ${SOURCE_COLORS[name] ?? "from-zinc-400 to-zinc-500"}`} />
      {name}
    </span>
  );
}

export function CallBadge({ call }: { call: Call | null | undefined }) {
  if (!call) return <span className="text-[12px] text-zinc-600">no price</span>;
  const s = { up: ["▲", "text-emerald-300 bg-emerald-400/10"], down: ["▼", "text-rose-300 bg-rose-400/10"], flat: ["–", "text-zinc-400 bg-white/[0.04]"] }[call];
  return <span className={`inline-flex w-[58px] items-center justify-center gap-1 rounded-md py-0.5 text-[11.5px] font-semibold ${s[1]}`}>{s[0]} {call}</span>;
}

/** Down / flat / up as one stacked bar. */
export function ProbBar({ down, flat, up }: { down: number; flat: number; up: number }) {
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-white/5" title={`down ${(down * 100).toFixed(0)}% · flat ${(flat * 100).toFixed(0)}% · up ${(up * 100).toFixed(0)}%`}>
      <div className="bg-rose-400/80" style={{ width: `${down * 100}%` }} />
      <div className="bg-zinc-500/40" style={{ width: `${flat * 100}%` }} />
      <div className="bg-emerald-400/90" style={{ width: `${up * 100}%` }} />
    </div>
  );
}

/** A move in % as a bar growing left (down) or right (up) from the middle. */
export function MoveBar({ v, scale = 1 }: { v: number | null; scale?: number }) {
  if (v == null) return <div className="h-1.5 w-full rounded-full bg-white/5" />;
  const w = Math.min(Math.abs(v) / scale, 1) * 50;
  return (
    <div className="relative h-1.5 w-full rounded-full bg-white/5">
      <div className="absolute top-[-3px] left-1/2 h-3 w-px bg-white/20" />
      <div className={`absolute top-0 h-full rounded-full ${v >= 0 ? "bg-emerald-400/90" : "bg-rose-400/90"}`}
        style={v >= 0 ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
    </div>
  );
}

export function Ticker({ t, bought }: { t: string; bought?: boolean }) {
  return (
    <span className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-[12px] font-bold tracking-tight ring-1 ring-inset ${bought
      ? "bg-emerald-400/15 text-emerald-200 ring-emerald-400/30" : "bg-white/[0.04] text-zinc-200 ring-white/10"}`}>{t}</span>
  );
}

export function Empty({ icon, title, text }: { icon: ReactNode; title: string; text?: string }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 py-10 text-center">
      <div className="grid size-11 place-items-center rounded-2xl bg-white/[0.04] text-zinc-400 ring-1 ring-white/10">{icon}</div>
      <div className="text-[14px] font-semibold text-zinc-200">{title}</div>
      {text && <div className="max-w-sm text-[12.5px] leading-relaxed text-zinc-500">{text}</div>}
    </div>
  );
}

export function Skeleton({ rows = 6 }: { rows?: number }) {
  return <div className="flex flex-col gap-2.5">{Array.from({ length: rows }, (_, i) => <div key={i} className="shimmer h-9" />)}</div>;
}
