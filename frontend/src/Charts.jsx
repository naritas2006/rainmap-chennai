import React, { useState } from "react";

// Small hand-rolled SVG charts: one series each, thin marks, hover tooltips.

function useTip() {
  const [tip, setTip] = useState(null);
  const show = (e, html) => {
    const box = e.currentTarget.ownerSVGElement.getBoundingClientRect();
    setTip({ x: e.clientX - box.left, y: e.clientY - box.top, html });
  };
  return [tip, show, () => setTip(null)];
}

const Tip = ({ tip }) =>
  tip ? <div className="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.html}</div> : null;

const niceMax = (v) => {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return Math.ceil(v / p) * p;
};

export function HBar({ rows, valueKey, labelKey, format, title }) {
  const [tip, show, hide] = useTip();
  const W = 640, rowH = 30, left = 230, right = 56;
  const H = rows.length * rowH + 24;
  const max = niceMax(Math.max(...rows.map((r) => r[valueKey] || 0)));
  const x = (v) => left + ((W - left - right) * Math.max(v, 0)) / max;
  return (
    <div className="chart" aria-label={title}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        <g className="grid">
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={x(max * f)} x2={x(max * f)} y1={0} y2={H - 20} />
              <text x={x(max * f)} y={H - 6} textAnchor="middle">{format(max * f)}</text>
            </g>
          ))}
        </g>
        {rows.map((r, i) => {
          const y = i * rowH + 6, v = r[valueKey] || 0;
          return (
            <g key={r[labelKey]} onMouseMove={(e) => show(e, `${r[labelKey]}: ${format(v)}`)} onMouseLeave={hide}>
              <rect x={0} y={y - 4} width={W} height={rowH} fill="transparent" />
              <text x={left - 8} y={y + 12} textAnchor="end">{r[labelKey]}</text>
              <rect x={left} y={y + 2} width={Math.max(x(v) - left, 1)} height={16} rx={4}
                    fill={r.highlight ? "var(--shared)" : "var(--none)"} />
              <text x={x(v) + 6} y={y + 14}>{format(v)}</text>
            </g>
          );
        })}
      </svg>
      <Tip tip={tip} />
    </div>
  );
}

export function LineChart({ points, xLabel, yLabel, xFormat = (v) => v, yFormat = (v) => v }) {
  const [tip, show, hide] = useTip();
  const W = 520, H = 220, l = 48, r = 16, t = 12, b = 40;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = niceMax(Math.max(...ys, 1));
  const X = (v) => l + ((W - l - r) * (v - x0)) / (x1 - x0 || 1);
  const Y = (v) => H - b - ((H - t - b) * v) / y1;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} by ${xLabel}`}>
        <g className="grid">
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={l} x2={W - r} y1={Y(y1 * f)} y2={Y(y1 * f)} />
              <text x={l - 6} y={Y(y1 * f) + 4} textAnchor="end">{yFormat(y1 * f)}</text>
            </g>
          ))}
        </g>
        <polyline fill="none" stroke="var(--shared)" strokeWidth={2}
                  points={points.map((p) => `${X(p.x)},${Y(p.y)}`).join(" ")} />
        {points.map((p) => (
          <g key={p.x} onMouseMove={(e) => show(e, p.tip || `${xFormat(p.x)}: ${yFormat(p.y)}`)} onMouseLeave={hide}>
            <circle cx={X(p.x)} cy={Y(p.y)} r={12} fill="transparent" />
            <circle cx={X(p.x)} cy={Y(p.y)} r={4.5} fill="var(--shared)" stroke="var(--surface-1)" strokeWidth={2} />
            <text x={X(p.x)} y={H - b + 16} textAnchor="middle">{xFormat(p.x)}</text>
          </g>
        ))}
        <text x={(l + W - r) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
        <text x={12} y={H / 2} textAnchor="middle" transform={`rotate(-90 12 ${H / 2})`}>{yLabel}</text>
      </svg>
      <Tip tip={tip} />
    </div>
  );
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function MonthlyBand({ band }) {
  const [tip, show, hide] = useTip();
  const W = 640, H = 240, l = 44, r = 12, t = 12, b = 28;
  const max = niceMax(Math.max(...band.p90, ...band.test_mean));
  const X = (i) => l + ((W - l - r) * (i + 0.5)) / 12;
  const Y = (v) => H - b - ((H - t - b) * v) / max;
  const area = band.p90.map((v, i) => `${X(i)},${Y(v)}`).join(" ") + " " +
    band.p10.map((v, i) => `${X(i)},${Y(v)}`).reverse().join(" ");
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Monthly rainfall: model band and held-out years">
        <g className="grid">
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={l} x2={W - r} y1={Y(max * f)} y2={Y(max * f)} />
              <text x={l - 6} y={Y(max * f) + 4} textAnchor="end">{Math.round(max * f)}</text>
            </g>
          ))}
        </g>
        <polygon points={area} fill="var(--band)" />
        <polyline fill="none" stroke="var(--shared)" strokeWidth={2}
                  points={band.p50.map((v, i) => `${X(i)},${Y(v)}`).join(" ")} />
        {MONTHS.map((m, i) => (
          <g key={m} onMouseLeave={hide}
             onMouseMove={(e) => show(e, `${m}: model median ${band.p50[i]} mm (10–90%: ${band.p10[i]}–${band.p90[i]}), held-out mean ${band.test_mean[i]} mm`)}>
            <rect x={X(i) - (W - l - r) / 24} y={t} width={(W - l - r) / 12} height={H - t - b} fill="transparent" />
            <circle cx={X(i)} cy={Y(band.test_mean[i])} r={4.5} fill="var(--private)" stroke="var(--surface-1)" strokeWidth={2} />
            <text x={X(i)} y={H - 8} textAnchor="middle">{m}</text>
          </g>
        ))}
      </svg>
      <Tip tip={tip} />
      <div className="legend">
        <span><span className="swatch" style={{ background: "var(--band)" }} />Model 10–90% band</span>
        <span><span className="swatch" style={{ background: "var(--shared)" }} />Model median</span>
        <span><span className="swatch" style={{ background: "var(--private)", borderRadius: 6 }} />Held-out 2016–2023 mean</span>
      </div>
    </div>
  );
}
