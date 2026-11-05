import { ColorType, createChart, CrosshairMode, IChartApi, ISeriesApi, LineStyle, UTCTimestamp } from "lightweight-charts";
import { useEffect, useRef } from "react";
import { nyTime } from "../api";

/** Account value over time: a gradient area that turns red when below the start. */
export default function AreaChart({ data, base, height = 300, mini = false }: {
  data: [number, number][]; base: number; height?: number | "fill"; mini?: boolean;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi>();
  const series = useRef<ISeriesApi<"Baseline">>();

  useEffect(() => {
    if (!el.current) return;
    const c = createChart(el.current, {
      autoSize: true,
      // The TradingView logo is the library's required attribution: kept on the full chart, hidden on the tiny sparkline
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#71717a", fontFamily: "Inter Variable", fontSize: 11,
        attributionLogo: !mini },
      grid: { vertLines: { visible: false }, horzLines: { color: mini ? "transparent" : "rgba(255,255,255,0.04)" } },
      rightPriceScale: { visible: !mini, borderVisible: false, scaleMargins: { top: 0.15, bottom: 0.1 } },
      timeScale: { visible: !mini, borderVisible: false, timeVisible: true, secondsVisible: false },
      crosshair: {
        mode: CrosshairMode.Magnet,
        vertLine: { color: "rgba(124,140,255,0.5)", style: LineStyle.Dashed, labelBackgroundColor: "#272b3a", visible: !mini },
        horzLine: { color: "rgba(124,140,255,0.5)", style: LineStyle.Dashed, labelBackgroundColor: "#272b3a", visible: !mini },
      },
      handleScroll: !mini, handleScale: !mini,
      localization: { priceFormatter: (p: number) => "$" + p.toLocaleString("en-US", { maximumFractionDigits: 0 }) },
    });
    series.current = c.addBaselineSeries({
      baseValue: { type: "price", price: base },
      topLineColor: "#34d399", topFillColor1: "rgba(52,211,153,0.28)", topFillColor2: "rgba(52,211,153,0.02)",
      bottomLineColor: "#fb7185", bottomFillColor1: "rgba(251,113,133,0.02)", bottomFillColor2: "rgba(251,113,133,0.28)",
      lineWidth: 2, priceLineVisible: false, lastValueVisible: !mini,
      crosshairMarkerRadius: 4, crosshairMarkerBorderColor: "#0c0e14",
    });
    chart.current = c;
    return () => c.remove();
  }, [mini, base]);

  useEffect(() => {
    // One point per second at most (several trades can close together): keep the last value
    const pts = new Map<number, number>();
    for (const [ms, v] of data) pts.set(nyTime(ms), v);
    series.current?.setData([...pts.entries()].sort((a, b) => a[0] - b[0]).map(([t, v]) => ({ time: t as UTCTimestamp, value: v })));
    chart.current?.timeScale().fitContent();
  }, [data]);

  return <div ref={el} style={{ height: height === "fill" ? "100%" : height }} className="w-full" />;
}
