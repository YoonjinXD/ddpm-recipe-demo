import { BarScene, drawThumb, P } from './bars3d.js';
import { lineChart, legendHTML, fmtNum } from './charts.js';
import * as TX from './content.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const COL = { eps: '#2a78d6', x0: '#eb6834', v: '#138a60', flow: '#c98500', signal: '#6b6a65', noise: '#e34948',
  muted: '#898781', ones: '#6250d6' };
const RECIPE = ['Target', 'Schedule', 'Sampler'];
const rt = (n, text) => `<span class="rn c${n}">${n}</span><span class="rk c${n}">${RECIPE[n - 1]}</span> ${text}`;
const TL = { eps: 'ε', x0: 'x₀', v: 'v' };
const SCHED_NAME = { linear: 'linear', cosine: 'cosine', sigmoid: 'sigmoid' };
const SAMPLER_NAME = { ddpm1000: 'DDPM · 1000', ddim50: 'DDIM · 50', ddim10: 'DDIM · 10', ddim5: 'DDIM · 5' };

// ------------------------------------------------------------------ data
const LUT = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const sg = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  LUT[h] = e === 0 ? sg * f * 2 ** -24 : e === 31 ? (f ? NaN : sg * Infinity) : sg * (1 + f / 1024) * 2 ** (e - 15);
}
const cache = new Map();
function f16(rel) {
  if (!cache.has(rel)) {
    cache.set(rel, fetch(`data/${rel}`).then(r => { if (!r.ok) throw new Error(rel); return r.arrayBuffer(); })
      .then(b => { const u = new Uint16Array(b), o = new Float32Array(u.length); for (let i = 0; i < u.length; i++) o[i] = LUT[u[i]]; return o; }));
  }
  return cache.get(rel);
}
let M, BASE;
const base = (kind, d) => BASE.subarray((kind * 4 + d) * P, (kind * 4 + d + 1) * P);
const coef1 = (sched, t) => ({ a: M.common.schedules[sched].a[t], s: M.common.schedules[sched].s[t] });
const coef2 = (path, tau) => path === 'curved' ? coef1('cosine', Math.round(tau * 1000)) : { a: 1 - tau, s: tau };

// ------------------------------------------------------------------ formatting
const K = s => (window.katex ? window.katex.renderToString(s, { throwOnError: false }) : s);
function cf(v, d = 3) {
  if (!Number.isFinite(v)) return '\\infty';
  if (v === 0) return '0';
  if (Math.abs(v) < 0.001) { const [m, e] = v.toExponential(1).split('e'); return `${m}\\!\\times\\!10^{${+e}}`; }
  if (Math.abs(v) >= 100) return v.toFixed(0);
  return v.toFixed(d);
}
const pct = v => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`);
const fidTxt = v => (v === null || v === undefined ? '✗ NaN' : v >= 1000 ? v.toExponential(1).replace('e+', 'e') : v >= 100 ? v.toFixed(0) : v.toFixed(1));
function seqColor(k) { // sequential blue on a light surface: dark = small distance (better), light = large
  const st = ['#104281', '#256abf', '#3987e5', '#86b6ef', '#cde2fb'];
  const x = Math.max(0, Math.min(1, k)) * (st.length - 1), i = Math.min(st.length - 2, Math.floor(x)), f = x - i;
  const p = h => [1, 3, 5].map(j => parseInt(h.slice(j, j + 2), 16));
  const a = p(st[i]), b = p(st[i + 1]);
  const c = a.map((v, j) => Math.round(v + (b[j] - v) * f));
  return { bg: `rgb(${c})`, fg: x > 2.4 ? '#0b0b0b' : '#ffffff' };
}

// ------------------------------------------------------------------ state
const S1 = { target: 'eps', sched: 'linear', sampler: 'ddpm1000', clip: false, digit: 0, tk: 1, st: 5, seed: 0, frame: 0, err: 'x0' };
const S2 = { path: 'straight', sampler: 'euler', steps: 10, tk: 3, st: 4, seed: 0, frame: 0 };
const V = {}; // scenes
const VER = { t1: 0, s1: 0, t2: 0, s2: 0 }; // drop stale async updates
const clipKey = () => (S1.clip ? 'clip' : 'raw');
const runKey1 = () => `${S1.sched}-${S1.target}|${S1.sampler}|${clipKey()}`;

// ------------------------------------------------------------------ recipe bars
// part 1: each recipe has its own picker inside its own section
function renderPickers1() {
  const name = (r, v) => r.opts.find(o => o.v === v).html;
  const [target, sched, sampler] = TX.P1_RECIPES;
  for (const r of TX.P1_RECIPES) {
    let h = r.opts.map(o => `<button data-k="${r.key}" data-v="${o.v}" class="${S1[r.key] === o.v ? 'on' : ''}">${o.html}</button>`).join('');
    if (r.clip) h += `<label class="chk" style="margin-left:10px"><input type="checkbox" data-clip ${S1.clip ? 'checked' : ''}> clip <i>x̂</i><sub>0</sub> to [−1, 1]</label>`;
    const el = $(`#pick-${r.key}`);
    el.innerHTML = h;
    $$('button[data-k]', el).forEach(b => b.addEventListener('click', () => setState(1, { [b.dataset.k]: b.dataset.v })));
    const cb = $('input[data-clip]', el);
    if (cb) cb.addEventListener('change', () => setState(1, { clip: cb.checked }));
    const others = [target, sched, sampler].filter(o => o !== r).map(o => `${o.label}: ${name(o, S1[o.key])}`);
    $(`#ctx-${r.key}`).innerHTML = `using ${others.join(' · ')}`;
  }
}

function setState(part, patch) {
  Object.assign(part === 1 ? S1 : S2, patch);
  if (part === 1) { renderPickers1(); updateTrain1(); updateSched1(); updateSample1(); renderCards1(); }
  else if (V.t2) { renderPickers2(); updateTrain2(); updateSched2(); updateSample2(); renderCards2(); }
}

// small segmented pickers (digits, seeds)
function seg(el, items, cur, onPick) {
  el.innerHTML = items.map((it, i) => `<button data-i="${i}" class="${i === cur ? 'on' : ''}">${it}</button>`).join('');
  $$('button', el).forEach(b => b.addEventListener('click', () => onPick(+b.dataset.i)));
}

// ------------------------------------------------------------------ PART 1: hero
function updateHero() {
  const v = base(0, 0);
  V.hero.set(0, { v });
  V.hero.showValues(0, v);
  drawThumb($('#hero-thumb'), v);
}

// ------------------------------------------------------------------ PART 1: the three schedules side by side
function updateSched1() {
  const t = M.common.train_t[S1.st];
  const x0 = base(0, 0), eps = base(1, 0);
  const scheds = ['linear', 'cosine', 'sigmoid'];
  scheds.forEach((sc, k) => {
    const { a, s } = coef1(sc, t);
    const xt = new Float32Array(P), g = new Float32Array(P);
    for (let i = 0; i < P; i++) { xt[i] = a * x0[i] + s * eps[i]; g[i] = s * eps[i]; }
    V.sc.set(k, { v: xt, ghost: g });
  });
  V.sc.setLabels(scheds.map(sc => {
    const { a } = coef1(sc, t);
    const on = sc === S1.sched ? `<span class="tag" style="color:#f4c7c7">selected</span>` : '';
    return `<b>${SCHED_NAME[sc]}</b>${Math.round(a * a * 100)}% of the image left${on}`;
  }));
  $('#sched-t').value = S1.st;
  $('#sched-t-out').innerHTML = `<i>t</i> = ${t} of 1000`;
  renderSchedChart1();
}

// ------------------------------------------------------------------ PART 1: training view
async function updateTrain1() {
  const my = ++VER.t1;
  const name = `${S1.sched}-${S1.target}`, md = M.p1.models[name];
  const arr = await f16(md.train_file);
  if (my !== VER.t1) return;
  const tt = M.common.train_t, k = S1.tk, d = S1.digit, t = tt[k];
  const { a, s } = coef1(S1.sched, t);
  const x0 = base(0, d), eps = base(1, d);
  const off = (d * tt.length + k) * 2 * P;
  const epsh = arr.subarray(off, off + P), x0h = arr.subarray(off + P, off + 2 * P);
  const xt = new Float32Array(P), gT = new Float32Array(P), gH = new Float32Array(P);
  let e0 = 0, ee = 0, big = 0;
  for (let i = 0; i < P; i++) {
    xt[i] = a * x0[i] + s * eps[i]; gT[i] = s * eps[i]; gH[i] = s * epsh[i];
    e0 += (x0h[i] - x0[i]) ** 2; ee += (epsh[i] - eps[i]) ** 2;
    if (!(Math.abs(x0h[i]) < 7)) big++;
  }
  e0 /= P; ee /= P;
  V.t1.set(0, { v: xt, ghost: gT }); V.t1.set(1, { v: xt, ghost: gH }); V.t1.set(2, { v: x0h });
  const tag = `<span class="tag" style="color:${COL[S1.target]}">network output</span>`;
  V.t1.setLabels([
    `<b>What the network sees: <i>x<sub>t</sub></i></b>red = the noise we mixed in`,
    `<b>${S1.target === 'eps' ? 'Its guess of the noise' : 'The noise its answer implies'}</b>red = <i>s<sub>t</sub> ε̂</i>${S1.target === 'eps' ? tag : ''}`,
    `<b>Its guess of the clean image: <i>x̂</i><sub>0</sub></b>${S1.target === 'x0' ? tag : `computed from ${S1.target === 'v' ? '<i>v̂</i>' : '<i>ε̂</i>'}`}`,
  ]);
  $('#train-t-ddpm').value = k;
  $('#train-t-out-ddpm').innerHTML = `<i>t</i> = ${t} of 1000`;
  $('#train-formula-ddpm').innerHTML = `<div class="row">${K(`x_t = a_t\\,x_0 + s_t\\,\\varepsilon = ${cf(a)}\\,x_0 + ${cf(s)}\\,\\varepsilon`)}
    <span class="cap">${SCHED_NAME[S1.sched]} schedule · ${Math.round(a * a * 100)}% of the image’s energy is left (a<sub>t</sub>² = ᾱ<sub>t</sub>)</span></div>`;
  const conv = {
    eps: `\\hat x_0 = \\frac{x_t - s_t\\,\\hat\\varepsilon}{a_t} = \\frac{x_t - ${cf(s)}\\,\\hat\\varepsilon}{${cf(a)}}`,
    x0: `\\hat\\varepsilon = \\frac{x_t - a_t\\,\\hat x_0}{s_t} = \\frac{x_t - ${cf(a)}\\,\\hat x_0}{${cf(s)}}`,
    v: `\\hat x_0 = a_t x_t - s_t \\hat v = ${cf(a)}\\,x_t - ${cf(s)}\\,\\hat v,\\quad \\hat\\varepsilon = s_t x_t + a_t \\hat v`,
  }[S1.target];
  $('#train-under-ddpm').innerHTML = `<span>The other half, by inverting the mix: ${K(conv)}</span>
    <span>Training loss for this example (vs. ${TL[S1.target]}): <b>${fmtNum(md.train_loss[d][k], 4)}</b></span>
    <span>Error of <i>x̂</i><sub>0</sub>: <b>${fmtNum(e0, 4)}</b> · error of <i>ε̂</i>: <b>${fmtNum(ee, 4)}</b></span>
    ${big ? `<span style="color:${COL.flow}">${big} bars too tall to draw (shown in amber): the clean-image guess exploded.</span>` : ''}`;
  $('#train-code-ddpm').innerHTML = TX.p1Code(S1.target);
  renderSchedChart1();
}

// ------------------------------------------------------------------ PART 1: sampling view
let film1Key = '';
async function updateSample1() {
  const my = ++VER.s1;
  const key = runKey1(), run = M.p1.runs[key];
  const arr = await f16(run.file);
  if (my !== VER.s1) return;
  const F = run.frames_t.length;
  if (S1.frame > F - 1) S1.frame = F - 1;
  const sl = $('#sample-f-ddpm'); sl.max = F - 1; sl.value = S1.frame;
  const g = (f, c) => arr.subarray(((S1.seed * F + f) * 3 + c) * P, ((S1.seed * F + f) * 3 + c + 1) * P);
  const f = S1.frame, t = run.frames_t[f], last = f === F - 1;
  const { a, s } = coef1(S1.sched, t);
  const x = g(f, 0), epsh = g(f, 1), x0h = g(f, 2);
  const gh = new Float32Array(P);
  if (!last) for (let i = 0; i < P; i++) gh[i] = s * epsh[i];
  V.s1.set(0, { v: x, ghost: last ? null : gh });
  V.s1.set(1, { v: last ? x : x0h });
  V.s1.set(2, { v: last ? x : g(f + 1, 0) });
  const tn = last ? 0 : run.frames_t[f + 1];
  V.s1.setLabels(last ? ['<b>Finished: <i>x</i><sub>0</sub></b>the generated image', '<b>(same)</b>', '<b>(same)</b>'] : [
    `<b>Now: <i>x<sub>t</sub></i>, <i>t</i> = ${t}</b>red = predicted noise <i>s<sub>t</sub> ε̂</i>`,
    `<b>Clean guess <i>x̂</i><sub>0</sub></b>remove the noise, undo the scaling`,
    `<b>Next frame: <i>t</i> = ${tn}</b>${S1.sampler === 'ddpm1000' ? '100 small steps later' : run.n > 10 ? `${run.n / 10} steps later` : 'one step later'}`,
  ]);
  $('#sample-f-out-ddpm').innerHTML = `step ${run.frame_steps[f]} of ${run.n} · <i>t</i> = ${t}`;
  // formula
  let fh;
  if (last) fh = `<span class="cap">Done. Drag the slider back to watch the noise being removed.</span>`;
  else if (S1.sampler === 'ddpm1000') {
    const A = M.common.schedules[S1.sched].a, ap = A[t - 1], alpha = (A[t] / ap) ** 2, beta = 1 - alpha, abar = A[t] ** 2;
    const sig = Math.sqrt(Math.max(0, beta * (1 - ap * ap) / (1 - abar)));
    fh = `<div class="row">${K(`x_{t-1} = \\tfrac{1}{\\sqrt{\\alpha_t}}\\Big(x_t - \\tfrac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\hat\\varepsilon\\Big) + \\sigma_t z = ${cf(1 / Math.sqrt(alpha))}\\,\\big(x_t - ${cf(beta / Math.sqrt(1 - abar))}\\,\\hat\\varepsilon\\big) + ${cf(sig)}\\,z`)}
      <span class="cap">Algorithm 2 at <i>t</i> = ${t} (<i>z</i> = fresh noise). Repeated 100 times between frames.${clipKey() === 'clip' ? ' Computed through a clipped <i>x̂</i><sub>0</sub>, as in the official code.' : ''}</span></div>`;
  } else {
    const tp = t - 1000 / run.n, an = M.common.schedules[S1.sched].a[tp], sn = M.common.schedules[S1.sched].s[tp];
    fh = `<div class="row">${K(`\\hat x_0 = \\frac{x_t - s_t\\,\\hat\\varepsilon}{a_t} = \\frac{x_t - ${cf(s)}\\,\\hat\\varepsilon}{${cf(a)}}`)}
      ${K(`x_{t'} = a_{t'}\\,\\hat x_0 + s_{t'}\\,\\hat\\varepsilon = ${cf(an)}\\,\\hat x_0 + ${cf(sn)}\\,\\hat\\varepsilon`)}
      <span class="cap">DDIM step <i>t</i> = ${t} → <i>t′</i> = ${tp}: no fresh noise.${clipKey() === 'clip' ? ' <i>x̂</i><sub>0</sub> is clipped to [−1, 1] first.' : ''}</span></div>`;
  }
  $('#sample-formula-ddpm').innerHTML = fh;
  // filmstrip
  if (film1Key !== key + S1.seed) {
    film1Key = key + S1.seed;
    const fs = $('#film-ddpm');
    fs.innerHTML = run.frames_t.map((tt, i) => `<button class="film" data-f="${i}"><canvas width="28" height="28"></canvas>t=${tt}</button>`).join('');
    $$('.film', fs).forEach((b, i) => { drawThumb($('canvas', b), g(i, 0)); b.addEventListener('click', () => { S1.frame = i; updateSample1(); }); });
  }
  $$('#film-ddpm .film').forEach((b, i) => b.classList.toggle('on', i === f));
  seg($('#seeds-ddpm'), ['noise A', 'noise B', 'noise C', 'noise D'], S1.seed, i => { S1.seed = i; updateSample1(); });
}

// ------------------------------------------------------------------ PART 1: cards
// a 2D swiss roll, noised by each schedule: easier to read than a single digit
const SCHED_COL = { linear: '#2a78d6', cosine: '#eb6834', sigmoid: '#138a60' };
const ROLL = (() => {
  let seed = 7;
  const rand = () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const n = 600, pts = [];
  for (let i = 0; i < n; i++) {
    const u = rand(), th = 1.5 * Math.PI * (1 + 2 * u);
    pts.push({ u, x: th * Math.cos(th) / 6 + 0.04 * gauss(), y: th * Math.sin(th) / 6 + 0.04 * gauss(), ex: gauss(), ey: gauss() });
  }
  return pts;
})();
const ROLL_T = [0, 100, 300, 500, 700, 1000];

function drawRoll(canvas, a, s) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || 120;
  canvas.width = canvas.height = w * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, w);
  const sc = w / 6.4;
  for (const p of ROLL) {
    const x = a * p.x + s * p.ex, y = a * p.y + s * p.ey;
    const k = Math.round(60 + 150 * p.u);
    ctx.fillStyle = `rgb(${Math.round(k * 0.35)}, ${Math.round(k * 0.55)}, ${Math.min(255, k + 40)})`;
    ctx.fillRect(w / 2 + x * sc - 1.2, w / 2 - y * sc - 1.2, 2.4, 2.4);
  }
}

function renderSchedChart1() {
  const el = $('#card-sched-ddpm');
  const sch = M.common.schedules;
  const tNow = M.common.train_t[S1.st];
  const near = ROLL_T.reduce((b, t) => (Math.abs(t - tNow) < Math.abs(b - tNow) ? t : b), ROLL_T[0]);

  let grid = '<div class="roll-grid"><div></div>' + ROLL_T.map(t => `<div class="roll-t${t === near ? ' on' : ''}"><i>t</i> = ${t}</div>`).join('');
  for (const nm of ['linear', 'cosine', 'sigmoid']) {
    grid += `<button class="roll-name${nm === S1.sched ? ' on' : ''}" data-s="${nm}"><i class="sw" style="background:${SCHED_COL[nm]}"></i>${SCHED_NAME[nm]}</button>`;
    grid += ROLL_T.map(t => `<canvas class="roll${t === near ? ' on' : ''}${nm === S1.sched ? ' sel' : ''}" data-s="${nm}" data-t="${t}"></canvas>`).join('');
  }
  grid += '</div>';

  const ts = [];
  for (let t = 0; t <= 1000; t += 10) ts.push(t);
  const series = ['linear', 'cosine', 'sigmoid'].map(nm => ({
    id: nm, label: SCHED_NAME[nm], color: SCHED_COL[nm], width: nm === S1.sched ? 3 : 1.8, opacity: nm === S1.sched ? 1 : 0.7,
    pts: ts.map(t => [t, sch[nm].s[t]]),
  }));

  el.innerHTML = `<h3>${rt(2, 'the same swiss roll, noised by each schedule')}</h3>
    <p class="sub">Each dot is one data point; color follows its position along the roll. Read across a row to see how fast the shape dissolves. Click a schedule to select it.</p>
    ${grid}
    <h4 class="mini">How fast the noise grows: the noise weight <i>s<sub>t</sub></i></h4>
    ${legendHTML(series.map(s => ({ label: s.label, color: s.color, opacity: s.opacity })))}
    <div class="chart-slot"></div>
    <p class="take">Linear piles on noise early: by <i>t</i> ≈ 300 the roll is already gone. Cosine and sigmoid keep the shape longer and spread the destruction more evenly over the steps (sigmoid: Jabri et al., 2023).</p>`;
  $$('canvas.roll', el).forEach(c => { const { a, s } = coef1(c.dataset.s, +c.dataset.t); drawRoll(c, a, s); });
  $$('[data-s]', el).forEach(b => b.addEventListener('click', () => setState(1, { sched: b.dataset.s })));
  lineChart($('.chart-slot', el), { width: 560, height: 200, aria: 'noise weight by schedule',
    x: { min: 0, max: 1000, label: 'noise level t', tipLabel: 't' }, y: { min: 0, max: 1, label: 'noise weight' },
    series, marker: tNow });
}

function renderErrChart1() {
  const el = $('#card-err-ddpm');
  const mode = S1.err, col = mode === 'x0' ? 2 : 3;
  const series = ['eps', 'x0', 'v'].map(tg => {
    const rows = M.p1.models[`${S1.sched}-${tg}`].loss_by_t;
    const on = tg === S1.target;
    return { id: tg, label: `predict ${TL[tg]}`, color: COL[tg], width: on ? 3 : 1.6, opacity: on ? 1 : 0.55, pts: rows.map(r => [r[0], r[col]]) };
  });
  const vals = series.flatMap(s => s.pts.map(p => p[1])).filter(v => v !== null && Number.isFinite(v) && v > 0);
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...vals)))), hi = Math.min(1e4, Math.pow(10, Math.ceil(Math.log10(Math.max(...vals)))));
  const refs = mode === 'x0' ? [{ y: M.common.baseline.mean_digit_mse, label: 'guess “average digit”' }] : [{ y: 1, label: 'guess “no noise”' }];
  const take = {
    eps: `ε-prediction is accurate when there is little noise, but turning its guess into a clean image divides by <i>a<sub>t</sub></i>. At <i>t</i> = <i>T</i> (${SCHED_NAME[S1.sched]}, <i>a<sub>T</sub></i> = ${fmtNum(M.common.schedules[S1.sched].a[1000])}) its clean-image error is <b>${fmtNum(M.p1.models[`${S1.sched}-eps`].loss_by_t.at(-1)[2])}</b>, against ${fmtNum(M.common.baseline.mean_digit_mse)} for just guessing the average digit. Switch the schedule to cosine or sigmoid to see it explode.`,
    x0: `x₀-prediction is the mirror image: its noise guess divides by <i>s<sub>t</sub></i>, which is ≈ 0 at <i>t</i> = 1. There its noise error is <b>${fmtNum(M.p1.models[`${S1.sched}-x0`].loss_by_t[0][3])}</b>, against ${fmtNum(M.p1.models[`${S1.sched}-eps`].loss_by_t[0][3])} for ε-prediction (see the noise view). DDPM §3.2 found its samples worse.`,
    v: '<i>v</i> never divides by something near zero, so both of its derived guesses stay reasonable from one end to the other.',
  }[S1.target];
  el.innerHTML = `<h3>${rt(1, 'what the network predicts')}</h3><p class="sub">Average error at every noise level, measured on 1000 test digits (log scale). All three networks were trained with the ${SCHED_NAME[S1.sched]} schedule.</p>
    <div class="seg small-seg" style="margin-bottom:6px"><button data-m="x0" class="${mode === 'x0' ? 'on' : ''}">error of the clean-image guess <i>x̂</i><sub>0</sub></button><button data-m="eps" class="${mode === 'eps' ? 'on' : ''}">error of the noise guess <i>ε̂</i></button></div>
    ${legendHTML(series.map(s => ({ label: s.label, color: s.color, opacity: s.opacity })).concat([{ label: refs[0].label, color: COL.muted, dash: true }]))}
    <div class="chart-slot"></div><p class="take">${take}</p>`;
  $$('button[data-m]', el).forEach(b => b.addEventListener('click', () => { S1.err = b.dataset.m; renderErrChart1(); }));
  lineChart($('.chart-slot', el), { width: 560, height: 250, aria: 'error by noise level',
    x: { min: 0, max: 1000, label: 'noise level t', tipLabel: 't' }, y: { min: lo, max: hi, log: true, label: 'mean squared error' },
    series, refs });
}

function overviewTable1() {
  const cl = clipKey();
  const cells = [];
  for (const tg of ['eps', 'x0', 'v']) for (const sc of ['linear', 'cosine', 'sigmoid']) cells.push(M.p1.runs[`${sc}-${tg}|${S1.sampler}|${cl}`].fid);
  const fin = cells.filter(v => v !== null && Number.isFinite(v)).map(v => Math.log(v));
  const lo = Math.min(...fin), hi = Math.max(...fin);
  let h = `<table class="overview"><tr><th></th>${['linear', 'cosine', 'sigmoid'].map(s => `<th>${SCHED_NAME[s]}</th>`).join('')}</tr>`;
  for (const tg of ['eps', 'x0', 'v']) {
    h += `<tr><th>predict ${TL[tg]}</th>`;
    for (const sc of ['linear', 'cosine', 'sigmoid']) {
      const r = M.p1.runs[`${sc}-${tg}|${S1.sampler}|${cl}`], on = tg === S1.target && sc === S1.sched;
      if (r.fid === null) h += `<td class="fail${on ? ' on' : ''}" data-t="${tg}" data-s="${sc}" title="sampling produced NaN">✗ NaN</td>`;
      else { const c = seqColor(hi > lo ? (Math.log(r.fid) - lo) / (hi - lo) : 0); h += `<td class="${on ? 'on' : ''}" data-t="${tg}" data-s="${sc}" style="background:${c.bg};color:${c.fg}" title="FID-style ${r.fid.toFixed(2)}, ${pct(r.digit_rate)} look like digits">${fidTxt(r.fid)}</td>`; }
    }
    h += '</tr>';
  }
  return h + '</table>';
}

function renderSamples1() {
  const el = $('#card-samples-ddpm');
  const key = runKey1(), r = M.p1.runs[key];
  const calls = { ddpm1000: 1000, ddim50: 50, ddim10: 10, ddim5: 5 }[S1.sampler];
  el.innerHTML = `<h3>${rt(3, 'the samples')}</h3>
    <p class="sub">${Math.min(64, M.common.n_eval)} of ${M.common.n_eval} samples with predict ${TL[S1.target]} · ${SCHED_NAME[S1.sched]} · ${SAMPLER_NAME[S1.sampler]}${clipKey() === 'clip' ? ' · clipped' : ''}. Every recipe starts from the same noise; the top-left four are “noise A–D” from the slider above.</p>
    <div class="samples"><img alt="generated digits" src="data/${r.grid}">
    <div><div class="stats">
      <div class="stat"><div class="k">look like a digit</div><div class="v ${r.digit_rate < 0.5 ? 'bad' : ''}">${pct(r.digit_rate)}</div></div>
      <div class="stat"><div class="k">FID-style distance ↓</div><div class="v ${r.fid === null || r.fid > 50 ? 'bad' : ''}">${fidTxt(r.fid)}</div></div>
      <div class="stat"><div class="k">network calls</div><div class="v">${calls}</div></div></div>
      <p class="sub" style="margin:0 0 6px">FID-style distance for all nine (target × schedule) networks with the same sampler. Darker is better; click a cell to switch.</p>
      ${overviewTable1()}
    </div></div>`;
  $$('td[data-t]', el).forEach(td => td.addEventListener('click', () => setState(1, { target: td.dataset.t, sched: td.dataset.s })));
}

function renderFamous() {
  const el = $('#card-refs-ddpm');
  const th = (n, text) => `<th class="c${n}"><span class="rk">${n} · ${text}</span></th>`;
  const rows = TX.FAMOUS.map((f, i) => `<tr data-i="${i}" class="${f.go ? 'go' : ''}"><td><b>${f.name}</b>${f.audio ? ' <span class="pill">audio</span>' : ''}<br><span class="small">${f.ref ? TX.cite(f.ref) : 'Stability AI, 2022'}</span></td>
    <td>${f.predicts}</td><td>${f.sched}</td><td>${f.sampler}</td></tr>`).join('');
  el.innerHTML = `<h3>Who used which recipe</h3><p class="sub">Well-known image and audio models, written as the same three recipes.</p>
    <div class="famous"><table><tr><th>Model</th>${th(1, 'Target')}${th(2, 'Schedule')}${th(3, 'Sampler')}</tr>${rows}</table></div>
    <p class="small">Click a row to load the closest recipe this demo can run (models trained on MNIST, not the real systems).</p>`;
  $$('tr[data-i]', el).forEach(tr => tr.addEventListener('click', ev => {
    if (ev.target.closest('a')) return;
    const f = TX.FAMOUS[+tr.dataset.i];
    if (!f.go) return;
    const { part, ...st } = f.go;
    if (part === 1) { setState(1, st); $('#sampler').scrollIntoView({ behavior: 'smooth' }); }
    else { Object.assign(S2, st); location.hash = '#f-sampler'; route(); setState(2, {}); $('#f-sampler').scrollIntoView({ behavior: 'smooth' }); }
  }));
}

function renderCards1() { renderSchedChart1(); renderErrChart1(); renderSamples1(); }

// ------------------------------------------------------------------ PART 2
const PATH_NAME = { straight: 'straight line', curved: 'curved (cosine)' };
const PATH_COL = { straight: COL.flow, curved: COL.v };
const P2_SAMPLER = { euler: 'Euler', stoch: 'stochastic' };
const P2_STEPS = [1, 2, 5, 10, 50];
const runKey2 = () => `gauss|${S2.path}|${S2.sampler}|${S2.steps}`;
const stepsTxt = n => `${n} step${n > 1 ? 's' : ''}`;
const tf = v => (v < 0.01 && v > 0 ? v.toFixed(3) : (+v.toFixed(2)).toString());

function renderPickers2() {
  const pick = (el, items, cur, key) => {
    el.innerHTML = items.map(([v, h]) => `<button data-v="${v}" class="${cur === v ? 'on' : ''}">${h}</button>`).join('');
    $$('button', el).forEach(b => b.addEventListener('click', () => setState(2, { [key]: key === 'steps' ? +b.dataset.v : b.dataset.v })));
  };
  pick($('#pick2-path'), [['straight', 'straight line'], ['curved', 'curved = diffusion (cosine)']], S2.path, 'path');
  pick($('#pick2-sampler'), [['euler', 'Euler'], ['stoch', 'stochastic']], S2.sampler, 'sampler');
  pick($('#pick2-steps'), P2_STEPS.map(n => [n, stepsTxt(n)]), S2.steps, 'steps');
  const tgt = `Target: ${S2.path === 'curved' ? '<i>v</i> (the velocity along the curve)' : 'velocity <i>u</i>'}`;
  const path = `Schedule: ${PATH_NAME[S2.path]}`, smp = `Sampler: ${P2_SAMPLER[S2.sampler]} · ${stepsTxt(S2.steps)}`;
  $('#ctx2-target').innerHTML = `using ${path} · ${smp}`;
  $('#ctx2-path').innerHTML = `using ${tgt} · ${smp}`;
  $('#ctx2-sampler').innerHTML = `using ${tgt} · ${path}`;
}

function p2Model() {
  if (S2.path === 'curved') return { md: M.p1.models['cosine-v'], tt: M.common.train_t.map(t => t / 1000) };
  return { md: M.p2.models['fm-gauss'], tt: M.common.fm_train_t };
}

// ------------------------------------------------------------------ PART 2: target (training view)
async function updateTrain2() {
  const my = ++VER.t2;
  const pm = p2Model(), arr = await f16(pm.md.train_file);
  if (my !== VER.t2) return;
  const k = S2.tk, tau = pm.tt[k], curved = S2.path === 'curved';
  const x0 = base(0, 0), eps = base(1, 0);
  const { a, s } = coef2(S2.path, tau);
  const off = k * 2 * P, x0hk = arr.subarray(off + P, off + 2 * P);
  const xt = new Float32Array(P), gT = new Float32Array(P), mv = new Float32Array(P);
  let e0 = 0;
  for (let i = 0; i < P; i++) { xt[i] = a * x0[i] + s * eps[i]; gT[i] = s * eps[i]; mv[i] = x0hk[i] - xt[i]; e0 += (x0hk[i] - x0[i]) ** 2; }
  e0 /= P;
  V.t2.set(0, { v: xt, ghost: gT }); V.t2.set(1, { v: xt, arrows: mv }); V.t2.set(2, { v: x0hk });
  V.t2.setLabels([
    `<b>What the network sees: <i>x<sub>t</sub></i></b>red = the noise mixed in`,
    `<b>Its velocity guess</b>amber arrows = where each pixel is headed<span class="tag" style="color:${COL.flow}">network output</span>`,
    `<b>Where the arrows land: <i>x̂</i><sub>0</sub></b>${curved ? 'computed from <i>v̂</i>' : '<i>x<sub>t</sub></i> − <i>t û</i>'}`,
  ]);
  $('#train-t-flow').value = k;
  $('#train-t-out-flow').innerHTML = `<i>t</i> = ${tf(tau)}`;
  const fx = curved
    ? K(`x_t = a_t\\,x_0 + s_t\\,\\varepsilon = ${cf(a)}\\,x_0 + ${cf(s)}\\,\\varepsilon`)
    : K(`x_t = (1-t)\\,x_0 + t\\,\\varepsilon = ${cf(a)}\\,x_0 + ${cf(s)}\\,\\varepsilon`);
  $('#train-formula-flow').innerHTML = `<div class="row">${fx}<span class="cap">${curved ? 'curved path = Part 1’s cosine schedule' : 'straight line: no schedule to tune'}</span></div>`;
  const tgt = curved ? K('v = a_t\\,\\varepsilon - s_t\\,x_0') : K('u = \\varepsilon - x_0 = \\tfrac{d x_t}{dt}');
  const back = curved ? K(`\\hat x_0 = a_t x_t - s_t \\hat v = ${cf(a)}\\,x_t - ${cf(s)}\\,\\hat v`) : K(`\\hat x_0 = x_t - t\\,\\hat u = x_t - ${cf(tau)}\\,\\hat u`);
  $('#train-under-flow').innerHTML = `<span>Target: ${tgt}</span><span>The clean image, by walking back: ${back}</span>
    <span>Training loss for this example: <b>${fmtNum(pm.md.train_loss[0][k], 4)}</b> · error of <i>x̂</i><sub>0</sub>: <b>${fmtNum(e0, 4)}</b></span>`;
  $('#train-code-flow').innerHTML = TX.p2Code(S2);
}

function renderErrChart2() {
  const el = $('#card-err-flow');
  const L = M.common.loss_t;
  const fm = M.p2.models['fm-gauss'].loss_by_t, cv = M.p1.models['cosine-v'].loss_by_t, ce = M.p1.models['cosine-eps'].loss_by_t;
  const series = [
    { id: 'fm', label: 'flow matching: predict u (straight)', color: COL.flow, rows: fm, on: S2.path === 'straight' },
    { id: 'cv', label: 'Part 1: predict v (cosine)', color: COL.v, rows: cv, on: S2.path === 'curved' },
    { id: 'ce', label: 'Part 1: predict ε (cosine)', color: COL.eps, rows: ce, on: false },
  ].map(s => ({ id: s.id, label: s.label, color: s.color, width: s.on ? 3 : 1.6, opacity: s.on ? 1 : 0.55, pts: s.rows.map((r, i) => [L[i] / 1000, r[2]]) }));
  const vals = series.flatMap(s => s.pts.map(p => p[1])).filter(v => v !== null && Number.isFinite(v) && v > 0);
  const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...vals)))), hi = Math.min(1e4, Math.pow(10, Math.ceil(Math.log10(Math.max(...vals)))));
  el.innerHTML = `<h3>${rt(1, 'how good is the velocity?')}</h3><p class="sub">Error of the clean-image guess <i>x̂</i><sub>0</sub> at every <i>t</i>, measured on 1000 test digits (log scale).</p>
    ${legendHTML(series.map(s => ({ label: s.label, color: s.color, opacity: s.opacity })).concat([{ label: 'guess “average digit”', color: COL.muted, dash: true }]))}
    <div class="chart-slot"></div>
    <p class="take">A velocity is a mix of noise and image, so turning it into a clean image never divides by a number near zero.
      At pure noise (<i>t</i> = 1) the velocity model’s error is <b>${fmtNum(fm.at(-1)[2])}</b>, about the same as guessing the average digit (${fmtNum(M.common.baseline.mean_digit_mse)}):
      the best anyone can do from pure noise. ε-prediction on the same cosine path reaches <b>${fmtNum(ce.at(-1)[2])}</b> there.</p>`;
  lineChart($('.chart-slot', el), { width: 560, height: 250, aria: 'error by time',
    x: { min: 0, max: 1, label: 'time t (0 = data, 1 = noise)', tipLabel: 't', fmt: v => v.toFixed(2) }, y: { min: lo, max: hi, log: true, label: 'mean squared error' },
    series, refs: [{ y: M.common.baseline.mean_digit_mse, label: 'guess “average digit”' }], marker: p2Model().tt[S2.tk] });
}

// ------------------------------------------------------------------ PART 2: schedule (path)
const PATH_T = [0, 0.2, 0.4, 0.6, 0.8, 1];

function updateSched2() {
  const tau = S2.st / 10;
  const x0 = base(0, 0), eps = base(1, 0);
  ['straight', 'curved'].forEach((p, k) => {
    const { a, s } = coef2(p, tau);
    const xt = new Float32Array(P), g = new Float32Array(P);
    for (let i = 0; i < P; i++) { xt[i] = a * x0[i] + s * eps[i]; g[i] = s * eps[i]; }
    V.sf.set(k, { v: xt, ghost: g });
  });
  V.sf.setLabels(['straight', 'curved'].map(p => {
    const { a, s } = coef2(p, tau);
    const on = p === S2.path ? `<span class="tag" style="color:#f4c7c7">selected</span>` : '';
    return `<b>${PATH_NAME[p]}</b>image × ${a.toFixed(2)} + noise × ${s.toFixed(2)}${on}`;
  }));
  $('#sched-t-flow').value = S2.st;
  $('#sched-t-out-flow').innerHTML = `<i>t</i> = ${tf(tau)}`;
  renderPathChart2();
}

function renderPathChart2() {
  const el = $('#card-sched-flow');
  const tau = S2.st / 10;
  const near = PATH_T.reduce((b, t) => (Math.abs(t - tau) < Math.abs(b - tau) ? t : b), PATH_T[0]);
  let grid = '<div class="roll-grid"><div></div>' + PATH_T.map(t => `<div class="roll-t${t === near ? ' on' : ''}"><i>t</i> = ${t}</div>`).join('');
  for (const p of ['straight', 'curved']) {
    grid += `<button class="roll-name${p === S2.path ? ' on' : ''}" data-p="${p}"><i class="sw" style="background:${PATH_COL[p]}"></i>${PATH_NAME[p]}</button>`;
    grid += PATH_T.map(t => `<canvas class="roll${t === near ? ' on' : ''}${p === S2.path ? ' sel' : ''}" data-p="${p}" data-t="${t}"></canvas>`).join('');
  }
  grid += '</div>';
  const xs = Array.from({ length: 101 }, (_, i) => i / 100);
  const series = ['straight', 'curved'].map(p => ({
    id: p, label: PATH_NAME[p], color: PATH_COL[p], width: p === S2.path ? 3 : 1.6, opacity: p === S2.path ? 1 : 0.55,
    pts: xs.map(a => [a, p === 'straight' ? 1 - a : Math.sqrt(1 - a * a)]),
  }));
  const mid = coef2('curved', 0.5);
  el.innerHTML = `<h3>${rt(2, 'the same swiss roll, on both paths')}</h3>
    <p class="sub">Read across a row to see the shape dissolve. Click a path to select it.</p>
    ${grid}
    <h4 class="mini">The path itself: how much image (<i>a<sub>t</sub></i>) and how much noise (<i>s<sub>t</sub></i>) at each moment</h4>
    ${legendHTML([{ label: 'straight: a + s = 1 (flow matching)', color: PATH_COL.straight }, { label: 'curved: a² + s² = 1 (diffusion, cosine)', color: PATH_COL.curved }])}
    <div class="chart-slot"></div>
    <p class="take">Both paths start at the image and end at pure noise. On the straight line the two weights add up to 1, so halfway (<i>t</i> = 0.5) it is half image + half noise
      and the whole roll shrinks toward the middle; the curved path keeps the total size fixed (${mid.a.toFixed(2)} image + ${mid.s.toFixed(2)} noise).
      The straight path is the shortest, but in this small experiment that alone does not buy fewer steps (see the table in the Sampler section):
      the learned flow is not perfectly straight, because each noise sample is paired with a random image. Rectified Flow straightens it with an extra “reflow” stage (Liu et al., 2023).</p>`;
  $$('canvas.roll', el).forEach(c => { const { a, s } = coef2(c.dataset.p, +c.dataset.t); drawRoll(c, a, s); });
  $$('[data-p]', el).forEach(b => b.addEventListener('click', () => setState(2, { path: b.dataset.p })));
  lineChart($('.chart-slot', el), { width: 560, height: 250, aria: 'paths',
    x: { min: 0, max: 1, label: 'image weight a_t', tipLabel: 'a', fmt: v => v.toFixed(2) }, y: { min: 0, max: 1, label: 'noise weight s_t' }, series });
}

// ------------------------------------------------------------------ PART 2: sampler
let film2Key = '';
async function updateSample2() {
  const my = ++VER.s2;
  const key = runKey2(), run = M.p2.runs[key];
  const arr = await f16(run.file);
  if (my !== VER.s2) return;
  const F = run.frames_t.length, curved = S2.path === 'curved';
  if (S2.frame > F - 1) S2.frame = F - 1;
  const sl = $('#sample-f-flow'); sl.max = F - 1; sl.value = S2.frame;
  const g = (f, c) => arr.subarray(((S2.seed * F + f) * 3 + c) * P, ((S2.seed * F + f) * 3 + c + 1) * P);
  const tAt = i => (curved ? run.frames_t[i] / 1000 : run.frames_t[i]);
  const f = S2.frame, last = f === F - 1, tau = tAt(f);
  const x = g(f, 0), x0h = g(f, 2);
  const mv = new Float32Array(P);
  if (!last) { const nx = g(f + 1, 0); for (let i = 0; i < P; i++) mv[i] = nx[i] - x[i]; }
  V.s2.set(0, { v: x, arrows: last ? null : mv });
  V.s2.set(1, { v: last ? x : x0h });
  V.s2.set(2, { v: last ? x : g(f + 1, 0) });
  V.s2.setLabels(last ? ['<b>Finished</b>the generated image', '<b>(same)</b>', '<b>(same)</b>'] : [
    `<b>Now: <i>t</i> = ${tf(tau)}</b>arrows = the move to the next frame`,
    `<b>Clean guess <i>x̂</i><sub>0</sub></b>where the current velocity points`,
    `<b>Next frame: <i>t</i> = ${tf(tAt(f + 1))}</b>${run.n > 10 ? `${run.n / 10} steps later` : 'one step later'}`,
  ]);
  $('#sample-f-out-flow').innerHTML = `step ${run.frame_steps[f]} of ${run.n} · <i>t</i> = ${tf(tau)}`;
  let fh;
  const dt = 1 / run.n;
  if (last) fh = `<span class="cap">Done after ${run.n} network call${run.n > 1 ? 's' : ''}. Drag the slider back to watch the noise turn into a digit.</span>`;
  else if (S2.sampler === 'euler') {
    fh = curved
      ? `<div class="row">${K(`x_{t-\\Delta t} = x_t - \\Delta t\\,\\frac{dx}{dt} = x_t - ${cf(dt)}\\,\\frac{dx}{dt}`)}<span class="cap">Euler along the curve (in angle, with <i>v̂</i> as the velocity). This is DDIM.</span></div>`
      : `<div class="row">${K(`x_{t-\\Delta t} = x_t - \\Delta t\\,\\hat u = x_t - ${cf(dt)}\\,\\hat u`)}<span class="cap">Euler on a straight line. The whole sampler is this one line.</span></div>`;
  } else {
    fh = `<div class="row">${K(`x_{t'} = a_{t'}\\,\\hat x_0 + \\sqrt{s_{t'}^2 - c^2}\;\\hat\\varepsilon + c\\,z`)}<span class="cap">Stochastic step: part of the predicted noise is replaced by fresh noise <i>z</i> (DDPM-style, η = 1).</span></div>`;
  }
  $('#sample-formula-flow').innerHTML = fh;
  if (film2Key !== key + S2.seed) {
    film2Key = key + S2.seed;
    const fs = $('#film-flow');
    fs.innerHTML = run.frames_t.map((_, i) => `<button class="film" data-f="${i}"><canvas width="28" height="28"></canvas>t=${tf(tAt(i))}</button>`).join('');
    $$('.film', fs).forEach((b, i) => { drawThumb($('canvas', b), g(i, 0)); b.addEventListener('click', () => { S2.frame = i; updateSample2(); }); });
  }
  $$('#film-flow .film').forEach((b, i) => b.classList.toggle('on', i === f));
  seg($('#seeds-flow'), ['noise A', 'noise B', 'noise C', 'noise D'], S2.seed, i => { S2.seed = i; updateSample2(); });
}

function overviewTable2() {
  const rows = [['straight', 'euler'], ['straight', 'stoch'], ['curved', 'euler'], ['curved', 'stoch']];
  const all = rows.flatMap(([pa, sa]) => P2_STEPS.map(n => M.p2.runs[`gauss|${pa}|${sa}|${n}`].fid)).filter(v => v !== null).map(Math.log);
  const lo = Math.min(...all), hi = Math.max(...all);
  let h = `<table class="overview"><tr><th></th>${P2_STEPS.map(n => `<th>${stepsTxt(n)}</th>`).join('')}</tr>`;
  for (const [pa, sa] of rows) {
    h += `<tr><th>${pa} · ${P2_SAMPLER[sa]}</th>`;
    for (const n of P2_STEPS) {
      const r = M.p2.runs[`gauss|${pa}|${sa}|${n}`], on = pa === S2.path && sa === S2.sampler && n === S2.steps, d = `data-r="${pa}|${sa}|${n}"`;
      if (r.fid === null) { h += `<td class="fail${on ? ' on' : ''}" ${d}>✗</td>`; continue; }
      const c = seqColor((Math.log(r.fid) - lo) / (hi - lo || 1));
      h += `<td class="${on ? 'on' : ''}" ${d} style="background:${c.bg};color:${c.fg}" title="${pct(r.digit_rate)} look like digits">${fidTxt(r.fid)}</td>`;
    }
    h += '</tr>';
  }
  return h + '</table>';
}

function renderSamples2() {
  const el = $('#card-samples-flow');
  const r = M.p2.runs[runKey2()];
  const at = (pa, sa, n) => fidTxt(M.p2.runs[`gauss|${pa}|${sa}|${n}`].fid);
  el.innerHTML = `<h3>${rt(3, 'the samples')}</h3>
    <p class="sub">${Math.min(64, M.common.n_eval)} of ${M.common.n_eval} samples, ${PATH_NAME[S2.path]} · ${P2_SAMPLER[S2.sampler]} · ${stepsTxt(S2.steps)}. Same starting noise as Part 1; the top-left four are “noise A–D”.</p>
    <div class="samples"><img alt="generated digits" src="data/${r.grid}">
    <div><div class="stats">
      <div class="stat"><div class="k">look like a digit</div><div class="v ${r.digit_rate < 0.5 ? 'bad' : ''}">${pct(r.digit_rate)}</div></div>
      <div class="stat"><div class="k">FID-style distance ↓</div><div class="v ${r.fid === null || r.fid > 50 ? 'bad' : ''}">${fidTxt(r.fid)}</div></div>
      <div class="stat"><div class="k">network calls</div><div class="v">${S2.steps}</div></div></div>
      <p class="sub" style="margin:0 0 6px">FID-style distance for every path × sampler × step count. Darker is better; click a cell to switch.</p>
      ${overviewTable2()}
    </div></div>
    <p class="take">Fewer steps cost quality on every path: Euler on the straight line goes from ${at('straight', 'euler', 50)} at 50 steps to ${at('straight', 'euler', 5)} at 5.
      Straight and curved stay close at every step count, and adding noise (stochastic) helps only when there are many steps.
      For reference, DDPM in Part 1 needs 1000 steps for its best samples.</p>`;
  $$('td[data-r]', el).forEach(td => td.addEventListener('click', () => {
    const [path, sampler, n] = td.dataset.r.split('|');
    setState(2, { path, sampler, steps: +n });
  }));
}

function renderCards2() { renderErrChart2(); renderPathChart2(); renderSamples2(); }

function renderStatic2() {
  const row = (r, i) => `<tr><td class="c${i + 1}"><span class="rk">${i + 1} · ${r[0]}</span></td><td>${r[1]}</td><td>${r[2]}</td></tr>`;
  $('#compare-table').innerHTML = `<table><tr><th>Recipe</th><th>DDPM (Part 1)</th><th>Flow matching</th></tr>${TX.COMPARE.map(row).join('')}</table>`;
  $('#meet').innerHTML = TX.meetCards().map(c => `<div class="card"><h3>${c[0]}</h3><p>${c[1]}</p></div>`).join('');
  $('#tree').innerHTML = TX.TREE.map(e => `<div class="ev${e[3] ? ' key' : ''}"><span class="yr">${e[0]}</span>${e[2]} <span class="small">(${TX.cite(e[1])})</span></div>`).join('');
  $('#biblio').innerHTML = `<h3 class="sub" style="margin-top:0">References</h3>${TX.bibliography()}`;
}

// ------------------------------------------------------------------ playback + keys
function player(btn, getMax, getF, setF) {
  let timer = null;
  const stop = () => { clearInterval(timer); timer = null; btn.textContent = '▶'; };
  btn.addEventListener('click', () => {
    if (timer) return stop();
    if (getF() >= getMax()) setF(0);
    btn.textContent = '❚❚';
    timer = setInterval(() => { if (getF() >= getMax()) return stop(); setF(getF() + 1); }, 850);
  });
  return { toggle: () => btn.click(), stop };
}

function wireSliders() {
  $('#train-t-ddpm').addEventListener('input', e => { S1.tk = +e.target.value; updateTrain1(); });
  $('#sched-t').addEventListener('input', e => { S1.st = +e.target.value; updateSched1(); });
  $('#sample-f-ddpm').addEventListener('input', e => { S1.frame = +e.target.value; updateSample1(); });
  $('#train-t-flow').addEventListener('input', e => { S2.tk = +e.target.value; updateTrain2(); renderErrChart2(); });
  $('#sched-t-flow').addEventListener('input', e => { S2.st = +e.target.value; updateSched2(); });
  $('#sample-f-flow').addEventListener('input', e => { S2.frame = +e.target.value; updateSample2(); });
  const p1 = player($('#play-ddpm'), () => +$('#sample-f-ddpm').max, () => S1.frame, f => { S1.frame = f; updateSample1(); });
  const p2 = player($('#play-flow'), () => +$('#sample-f-flow').max, () => S2.frame, f => { S2.frame = f; updateSample2(); });
  // keyboard: arrows move the slider of whichever 3D view is most visible; space plays
  const views = [['#train-canvas-ddpm', '#train-t-ddpm', null], ['#sched-canvas', '#sched-t', null], ['#sample-canvas-ddpm', '#sample-f-ddpm', p1],
    ['#train-canvas-flow', '#train-t-flow', null], ['#sched-canvas-flow', '#sched-t-flow', null], ['#sample-canvas-flow', '#sample-f-flow', p2]];
  window.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea, select') && e.target.type !== 'range') return;
    let best = null, bestVis = 0;
    for (const v of views) {
      const r = $(v[0]).getBoundingClientRect();
      const vis = Math.max(0, Math.min(innerHeight, r.bottom) - Math.max(0, r.top));
      if (vis > bestVis) { bestVis = vis; best = v; }
    }
    if (!best || bestVis < 80) return;
    const sl = $(best[1]);
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const nv = Math.max(+sl.min, Math.min(+sl.max, +sl.value + (e.key === 'ArrowRight' ? 1 : -1)));
      sl.value = nv; sl.dispatchEvent(new Event('input'));
    } else if (e.key === ' ' && best[2]) { e.preventDefault(); best[2].toggle(); }
  });
}

// ------------------------------------------------------------------ routing
function route() {
  const part = /^#(flow|f-)/.test(location.hash) ? 'flow' : 'ddpm';
  $$('.part').forEach(s => { s.hidden = s.dataset.part !== part; });
  $$('.tabs a').forEach(a => a.classList.toggle('on', a.dataset.part === part));
  if (part === 'flow' && !V.t2) initPart2();
}

function initPart2() {
  V.t2 = new BarScene($('#train-canvas-flow'), { grids: 3 });
  V.sf = new BarScene($('#sched-canvas-flow'), { grids: 2 });
  V.s2 = new BarScene($('#sample-canvas-flow'), { grids: 3 });
  V.t2.setLegend([['#c3c2b7', 'pixel value'], [COL.noise, 'noise part'], ['#eda100', 'velocity (arrows)']]);
  V.sf.setLegend([['#c3c2b7', 'pixel value'], [COL.noise, 'noise part']]);
  V.s2.setLegend([['#c3c2b7', 'pixel value'], ['#eda100', 'move to the next frame (arrows)']]);
  setState(2, {});
}

// ------------------------------------------------------------------ boot
async function boot() {
  try {
    M = await (await fetch('data/manifest.json')).json();
    BASE = await f16('base.f16');
  } catch (err) {
    $('#loading').textContent = 'Could not load data/manifest.json. Serve this folder over http (not file://).';
    return;
  }
  renderStatic2();
  V.hero = new BarScene($('#hero-canvas'), { grids: 1, elev: 0.95, zoom: true });
  V.hero.setLegend([['#e9e8e4', 'white pixel = tall bar'], ['#3a3a38', 'black pixel = flat']]);
  V.t1 = new BarScene($('#train-canvas-ddpm'), { grids: 3 });
  V.s1 = new BarScene($('#sample-canvas-ddpm'), { grids: 3 });
  V.sc = new BarScene($('#sched-canvas'), { grids: 3 });
  V.sc.setLegend([['#c3c2b7', 'pixel value'], [COL.noise, 'noise part']]);
  const lg1 = [['#c3c2b7', 'pixel value'], [COL.noise, 'noise part'], [COL.flow, 'too tall to draw']];
  V.t1.setLegend(lg1); V.s1.setLegend(lg1);
  renderPickers1();
  renderFamous();
  updateHero();
  updateSched1();
  await Promise.all([updateTrain1(), updateSample1()]);
  renderCards1();
  wireSliders();
  window.addEventListener('hashchange', route);
  route();
  $('#loading').classList.add('done');
}
window.__demo = { V, S1, S2 }; // handy for poking at the page from the browser console
boot();
