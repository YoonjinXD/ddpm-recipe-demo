// Minimal SVG line chart: thin lines, recessive grid, direct labels, hover crosshair + tooltip.
const NS = 'http://www.w3.org/2000/svg';
const C = { grid: '#e6e5df', axis: '#c3c2b7', muted: '#6b6a65', text2: '#3f3e3b', text: '#0b0b0b' };

function niceLinTicks(min, max, n = 5) {
  const step0 = (max - min) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function fmtNum(v, d = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e4) return v.toExponential(1).replace('e+', 'e');
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(d);
  if (a === 0) return '0';
  return v.toExponential(1);
}

/**
 * opt = { width, height, x: {min, max, label, fmt}, y: {min, max, log, label, fmt},
 *         series: [{id, label, color, width, dash, opacity, pts: [[x, y|null], ...]}],
 *         refs: [{y, label, color}], marker: x|null }
 */
export function lineChart(el, opt) {
  const W = opt.width || 560, Hh = opt.height || 260;
  const m = { l: 52, r: 96, t: 10, b: 38 };
  const iw = W - m.l - m.r, ih = Hh - m.t - m.b;
  const ylog = !!opt.y.log;
  const ty = v => (ylog ? Math.log10(v) : v);
  const y0 = ty(opt.y.min), y1 = ty(opt.y.max);
  const sx = x => m.l + ((x - opt.x.min) / (opt.x.max - opt.x.min)) * iw;
  const sy = v => m.t + ih - ((ty(v) - y0) / (y1 - y0)) * ih;
  const clipY = v => Math.max(m.t, Math.min(m.t + ih, sy(v)));
  let s = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="${opt.aria || ''}">`;
  // grid + y ticks
  const yt = ylog
    ? Array.from({ length: Math.floor(y1) - Math.ceil(y0) + 1 }, (_, i) => Math.ceil(y0) + i)
        .filter((e, _, arr) => arr.length <= 6 || (e - arr[0]) % 2 === 0).map(e => Math.pow(10, e))
    : niceLinTicks(opt.y.min, opt.y.max, 4);
  for (const v of yt) {
    const y = sy(v);
    s += `<line x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`;
    const lab = ylog ? (v >= 1000 || v < 0.01 ? `1e${Math.round(Math.log10(v))}` : `${v}`) : (opt.y.fmt ? opt.y.fmt(v) : v);
    s += `<text x="${m.l - 8}" y="${y + 4}" fill="${C.muted}" font-size="11.5" text-anchor="end">${lab}</text>`;
  }
  const xt = opt.x.ticks || niceLinTicks(opt.x.min, opt.x.max, 5);
  for (const v of xt) {
    const x = sx(v);
    s += `<text x="${x}" y="${m.t + ih + 17}" fill="${C.muted}" font-size="11.5" text-anchor="middle">${opt.x.fmt ? opt.x.fmt(v) : v}</text>`;
  }
  s += `<line x1="${m.l}" x2="${m.l + iw}" y1="${m.t + ih}" y2="${m.t + ih}" stroke="${C.axis}" stroke-width="1"/>`;
  if (opt.x.label) s += `<text x="${m.l + iw / 2}" y="${Hh - 4}" fill="${C.muted}" font-size="12" text-anchor="middle">${opt.x.label}</text>`;
  if (opt.y.label) s += `<text x="${12}" y="${m.t + ih / 2}" fill="${C.muted}" font-size="12" text-anchor="middle" transform="rotate(-90 12 ${m.t + ih / 2})">${opt.y.label}</text>`;
  // reference lines
  for (const r of opt.refs || []) {
    const y = sy(r.y);
    s += `<line x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}" stroke="${r.color || C.muted}" stroke-width="1.5" stroke-dasharray="5 4"/>`;
  }
  // marker
  if (opt.marker !== null && opt.marker !== undefined) {
    const x = sx(opt.marker);
    s += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ih}" stroke="${C.text2}" stroke-width="1" stroke-dasharray="2 3"/>`;
  }
  // series
  const ends = [];
  for (const se of opt.series) {
    let d = '', pen = false;
    for (const [x, y] of se.pts) {
      const ok = y !== null && y !== undefined && Number.isFinite(y) && (!ylog || y > 0);
      if (!ok) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${clipY(y).toFixed(1)}`;
      pen = true;
    }
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" stroke-linejoin="round" stroke-linecap="round"
      ${se.dash ? `stroke-dasharray="${se.dash}"` : ''} opacity="${se.opacity ?? 1}"/>`;
    const last = [...se.pts].reverse().find(p => p[1] !== null && Number.isFinite(p[1]) && (!ylog || p[1] > 0));
    if (last && se.label) ends.push({ y: clipY(last[1]), x: sx(last[0]), label: se.label, color: se.color, op: se.opacity ?? 1 });
  }
  for (const r of opt.refs || []) if (r.label) ends.push({ y: sy(r.y), x: m.l + iw, label: r.label, color: C.muted, op: 1, ref: true });
  // direct labels with simple de-overlap
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
  const yMax = m.t + ih + 4;
  if (ends.length && ends[ends.length - 1].y > yMax) {
    ends[ends.length - 1].y = yMax;
    for (let i = ends.length - 2; i >= 0; i--) ends[i].y = Math.min(ends[i].y, ends[i + 1].y - 14);
  }
  for (const e of ends) {
    s += `<text x="${m.l + iw + 8}" y="${e.y + 4}" fill="${C.text2}" font-size="12" opacity="${Math.max(e.op, 0.7)}">${e.label}</text>`;
  }
  // hover layer
  s += `<line class="xh" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" stroke="${C.text2}" stroke-width="1" opacity="0"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/>`;
  s += '</svg>';
  el.innerHTML = `<div class="chart">${s}<div class="tip" hidden></div></div>`;
  const svg = el.querySelector('svg'), tip = el.querySelector('.tip'), xh = el.querySelector('.xh'), hit = el.querySelector('.hit');
  const xs = opt.series[0]?.pts.map(p => p[0]) || [];
  const move = ev => {
    const rect = svg.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * (W / rect.width);
    const xv = opt.x.min + ((px - m.l) / iw) * (opt.x.max - opt.x.min);
    let best = 0;
    xs.forEach((x, i) => { if (Math.abs(x - xv) < Math.abs(xs[best] - xv)) best = i; });
    const X = sx(xs[best]);
    xh.setAttribute('x1', X); xh.setAttribute('x2', X); xh.setAttribute('opacity', 0.6);
    const rows = opt.series.map(se => {
      const y = se.pts[best]?.[1];
      return `<div class="row"><span><i class="sw" style="background:${se.color};width:9px;height:9px;display:inline-block;border-radius:2px;margin-right:6px"></i>${se.label || se.id}</span><span>${fmtNum(y)}</span></div>`;
    }).join('');
    tip.innerHTML = `<div class="k">${opt.x.tipLabel || 'x'} = ${opt.x.fmt ? opt.x.fmt(xs[best]) : xs[best]}</div>${rows}`;
    tip.hidden = false;
    const cx = (X / W) * rect.width;
    tip.style.left = `${cx > rect.width * 0.6 ? cx - tip.offsetWidth - 24 : cx}px`;
    tip.style.top = `${(ev.clientY - rect.top)}px`;
  };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerleave', () => { tip.hidden = true; xh.setAttribute('opacity', 0); });
}

export function legendHTML(items) {
  return `<div class="legend">${items.map(it => `<span><i class="ln${it.dash ? ' dash' : ''}" style="border-color:${it.color};opacity:${it.opacity ?? 1}"></i>${it.label}</span>`).join('')}</div>`;
}
