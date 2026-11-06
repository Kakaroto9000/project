import { ScrollText } from "lucide-react";
import { useEffect, useState } from "react";
import { clockSec, get, LogLine, Status, when } from "../api";
import { Card, Empty } from "../components/ui";

export default function LogPage({ status }: { status: Status | null }) {
  const [lines, setLines] = useState<LogLine[]>([]);
  useEffect(() => { get<LogLine[]>("/api/log").then(setLines).catch(() => {}); }, [status?.version]);
  return (
    <Card title="What the engine did" className="h-full" pad={false}>
      <div className="scroll min-h-0 flex-1 px-5 pb-4 font-mono text-[12px] leading-[1.7] select-text">
        {!lines.length && <Empty icon={<ScrollText size={20} />} title="Quiet so far" />}
        {lines.map((l, i) => {
          const bad = /error|fail/i.test(l.message);
          return (
            <div key={i} className="flex gap-4 border-b border-white/[0.03] py-1">
              <span className="shrink-0 text-zinc-600" title={when(l.time)}>{clockSec(l.time)}</span>
              <span className={`whitespace-pre-wrap ${bad ? "text-rose-300" : l.message.startsWith("Predicted") ? "text-zinc-200" : "text-zinc-400"}`}>{l.message}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
