/* Little Language Model: the page. It coordinates the training workers (sends weights, averages their
   gradients, takes the AdamW step), asks the probe worker for samples and views, and draws everything.
   Text from the model or the visitor only ever goes in through textContent. */
(() => {
'use strict';
const $ = s => document.querySelector(s);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const SHAPE = { C: 48, D: 48, H: 4, L: 2, F: 192 };
const B = 8;                       // sequences per worker per step
const LR = 6e-3, WARM = 30, DECAY = 4000, MAX_STEPS = 5000;
const MILESTONES = [0, 20, 50, 100, 200, 400, 800, 1600, 3200, 5000];
const CORPORA = {
  shakespeare: { label: 'Shakespeare', prompts: ['ROMEO:\n', 'JULIET:\n', 'KING RICHARD III:\n', 'First Citizen:\n', 'MENENIUS:\n', 'LADY CAPULET:\n'], talk: 'ROMEO:', lapse: 'JULIET:\n' },
  code: { label: 'its own source code', prompts: ['function ', '  const ', '    if (', '// '], talk: 'function ', lapse: 'function ' },
  own: { label: 'your text', prompts: null, talk: '', lapse: null },
};

const cores = navigator.hardwareConcurrency || 4;
const nW = Math.max(1, Math.min(8, cores - 2));
let S = null, epoch = 0, running = false, pending = 0, acc = null, lossSum = 0, valSum = 0, valN = 0, stepT0 = 0;
let workers = [], probe = null, probeBusy = false, lastProbe = 0, genId = 0;

/* ───────── workers ───────── */

function makeWorkers() {
  try {
    for (let i = 0; i < nW; i++) { const w = new Worker('worker.js?v=__V__'); w.onmessage = onTrainer; w.onerror = workerFailed; workers.push(w); }
    probe = new Worker('worker.js?v=__V__'); probe.onmessage = onProbe; probe.onerror = workerFailed;
    $('#sCores').textContent = nW;
    return true;
  } catch (e) { workerFailed(); return false; }
}
function workerFailed() {
  running = false;
  status('This browser blocked background workers, so training can’t run here. Try a current Chrome, Edge, Firefox or Safari.');
  $('#go').disabled = true;
}

/* ───────── set up a run ───────── */

function setup(kind, text) {
  stop();
  epoch++;
  genId++;
  const voc = GPT.vocab(text);
  const tokens = Int32Array.from(voc.encode(text));
  const cfg = Object.assign({}, SHAPE, { V: voc.chars.length });
  const model = new GPT.Model(cfg).init(1000 + epoch);
  S = { kind, voc, tokens, split: Math.floor(tokens.length * 0.9), cfg, cfgKey: kind + ':' + epoch, params: model.params, adam: new GPT.Adam(model, { lr: LR }),
    step: 0, seen: 0, ema: null, losses: [], vals: [], lapse: [], rate: 0, sample: '', pca: null, startedAt: 0 };
  acc = new Float32Array(model.lay.size);
  for (let i = 0; i < workers.length; i++) workers[i].postMessage({ type: 'init', id: epoch, cfg, tokens: S.tokens, split: S.split, seed: epoch * 97 + i });
  $('#pcount').textContent = model.lay.size.toLocaleString('en-US');
  document.querySelectorAll('.cfgC').forEach(e => { e.textContent = cfg.C; });
  document.querySelectorAll('.cfgD').forEach(e => { e.textContent = cfg.D; });
  document.querySelectorAll('.cfgL').forEach(e => { e.textContent = cfg.L; });
  document.querySelectorAll('.cfgH').forEach(e => { e.textContent = cfg.H; });
  document.querySelectorAll('.cfgF').forEach(e => { e.textContent = cfg.F; });
  const c = CORPORA[kind];
  if (kind === 'own') { c.prompts = [text.slice(0, 12)]; c.talk = text.slice(0, 20).split('\n')[0]; c.lapse = text.slice(0, 1); }
  $('#prompt').value = c.talk;
  $('#gen').textContent = '';
  $('#lapse').replaceChildren();
  writeText('');
  vitals();
  drawLoss();
  mapTarget = null;
  status('Ready to learn from ' + c.label + ': ' + S.tokens.length.toLocaleString('en-US') + ' characters, ' + cfg.V + ' different ones.');
  setLive('untrained');
  probeNow(true);
}

/* ───────── training loop ───────── */

function start() {
  if (!S || running || S.step >= MAX_STEPS) return;
  running = true;
  if (!S.startedAt) S.startedAt = performance.now();
  $('#go').textContent = 'Pause';
  setLive('learning');
  status(nW > 1 ? 'Training on ' + nW + ' CPU cores in parallel. It pauses by itself when you switch tabs.' : 'Training on 1 CPU core (this device has few to spare). It pauses by itself when you switch tabs.');
  tick();
}
function stop() {
  running = false;
  $('#go').textContent = S && S.step ? 'Keep training' : 'Start training';
  if (S) setLive(S.step ? 'paused' : 'untrained');
}

function tick() {
  if (!running || pending) return;
  pending = workers.length;
  acc.fill(0); lossSum = 0; valSum = 0; valN = 0;
  stepT0 = performance.now();
  const val = S.step % 25 === 0;
  for (const w of workers) w.postMessage({ type: 'step', id: epoch, step: S.step, params: S.params, B, T: S.cfg.C, val });
}

function onTrainer(e) {
  const m = e.data;
  if (m.type === 'error') { status('Training stopped: ' + m.message); stop(); return; }
  if (m.type !== 'grads' || m.id !== epoch || m.step !== S.step) return;
  const g = m.grads;
  for (let i = 0; i < g.length; i++) acc[i] += g[i];
  lossSum += m.loss;
  if (m.val != null) { valSum += m.val; valN++; }
  if (--pending) return;
  for (let i = 0; i < acc.length; i++) acc[i] /= workers.length;
  S.adam.step(S.params, acc, GPT.lrAt(S.step, LR, WARM, DECAY));
  const loss = lossSum / workers.length;
  S.ema = S.ema == null ? loss : S.ema * 0.9 + loss * 0.1;
  S.losses.push([S.step, S.ema]);
  if (valN) S.vals.push([S.step, valSum / valN]);
  S.step++;
  const chars = workers.length * B * S.cfg.C;
  S.seen += chars;
  const dt = (performance.now() - stepT0) / 1000;
  S.rate = S.rate ? S.rate * 0.85 + (chars / dt) * 0.15 : chars / dt;
  if (MILESTONES.includes(S.step) || performance.now() - lastProbe > 1500) probeNow(MILESTONES.includes(S.step));
  if (S.step >= MAX_STEPS) { stop(); status('Done: ' + MAX_STEPS.toLocaleString('en-US') + ' steps. Talk to it below, or start over with different text.'); setLive('trained'); }
  scheduleDraw();
  if (running) setTimeout(tick, 0);
}

/* ───────── the probe: samples, odds, attention, map ───────── */

let probeQueue = null;
function probeNow(milestone) {
  if (!S) return;
  const c = CORPORA[S.kind];
  const req = { type: 'probe', tag: epoch, step: S.step, cfg: S.cfg, cfgKey: S.cfgKey, params: S.params,
    sample: milestone ? { prompt: S.voc.encode(c.lapse), n: 110, temp: 0.7, seed: 42, lapse: true } : { prompt: S.voc.encode(c.prompts[S.step % c.prompts.length]), n: 260, temp: 0.8, seed: S.step + 7 },
    inspect: { prompt: S.voc.encode($('#prompt').value || c.talk || '\n'), temp: +$('#temp').value } };
  if (probeBusy) { if (milestone || !probeQueue) probeQueue = req; return; }
  send(req);
}
function send(req) {
  probeBusy = true; lastProbe = performance.now();
  req.params = req.params.slice();
  probe.postMessage(req);
  probe._last = req;
}

function onProbe(e) {
  const m = e.data;
  if (m.type === 'gen') { onGen(m); return; }
  if (m.type === 'error') { probeBusy = false; return; }
  if (m.type !== 'probe') return;
  probeBusy = false;
  const req = probe._last;
  if (m.tag === epoch && S) {
    const c = CORPORA[S.kind];
    if (m.sample) {
      const text = S.voc.decode(m.sample);
      if (req.sample.lapse) {
        addLapse(m.step, c.lapse, text);
        if (m.step === 0 || !S.sample) writeText(c.lapse + text);
      } else writeText(S.voc.decode(req.sample.prompt) + text);
    }
    if (m.probs) drawOdds(m.probs);
    if (m.att) drawHeads(m.att, m.ctx);
    if (m.pca) setMap(m.pca);
  }
  if (probeQueue) { const q = probeQueue; probeQueue = null; q.params = S.params; q.step = S.step; send(q); }
}

/* ───────── writing panel ───────── */

let typing = null;
function writeText(text) {
  if (S) S.sample = text;
  const el = $('#writing');
  if (typing) cancelAnimationFrame(typing);
  if (reduced || !text) { el.textContent = text || '…'; return; }
  let i = 0;
  const step = () => { i = Math.min(text.length, i + 6); el.textContent = text.slice(0, i); if (i < text.length) typing = requestAnimationFrame(step); };
  step();
}

function addLapse(step, prompt, text) {
  if (S.lapse.some(x => x.step === step)) return;
  S.lapse.push({ step });
  const loss = S.ema == null ? Math.log(S.cfg.V) : S.ema;
  const li = document.createElement('li');
  const h = document.createElement('div'); h.className = 'lp-h';
  const b = document.createElement('b'); b.textContent = step === 0 ? 'Untrained' : 'Step ' + step.toLocaleString('en-US');
  const s = document.createElement('span'); s.textContent = 'loss ' + loss.toFixed(2) + ' · ' + S.seen.toLocaleString('en-US') + ' characters read';
  h.append(b, s);
  const pre = document.createElement('pre');
  const pr = document.createElement('span'); pr.className = 'pr'; pr.textContent = prompt;
  pre.append(pr, text);
  li.append(h, pre);
  $('#lapse').prepend(li);
}

/* ───────── vitals ───────── */

function vitals() {
  if (!S) return;
  $('#sStep').textContent = S.step.toLocaleString('en-US');
  $('#sLoss').textContent = S.ema == null ? '—' : S.ema.toFixed(3);
  $('#sPerp').textContent = S.ema == null ? S.cfg.V : Math.exp(S.ema).toFixed(1);
  $('#sTok').textContent = S.seen >= 1e6 ? (S.seen / 1e6).toFixed(2) + 'M' : S.seen.toLocaleString('en-US');
  $('#sRate').textContent = S.rate ? Math.round(S.rate).toLocaleString('en-US') : '—';
}

/* ───────── drawing ───────── */

let drawQueued = false;
function scheduleDraw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(() => { drawQueued = false; vitals(); drawLoss(); }); } }

function fit(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1), w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function drawLoss() {
  const cv = $('#lossChart');
  if (!cv.clientWidth || !S) return;
  const { ctx, w, h } = fit(cv), L = 40, R = 12, T = 12, Bm = 26;
  ctx.clearRect(0, 0, w, h);
  const rand = Math.log(S.cfg.V), maxX = Math.max(200, S.step * 1.1), lo = 0.8, hi = rand + 0.35;
  const X = s => L + (s / maxX) * (w - L - R), Y = v => T + (1 - (v - lo) / (hi - lo)) * (h - T - Bm);
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillStyle = css('--dim'); ctx.strokeStyle = 'rgba(255,255,255,.07)'; ctx.lineWidth = 1;
  for (let v = 1; v <= hi; v += 1) { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w - R, Y(v)); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(v.toFixed(1), L - 6, Y(v) + 4); }
  ctx.textAlign = 'center';
  const stepTick = maxX > 2500 ? 1000 : maxX > 1200 ? 500 : maxX > 500 ? 200 : 50;
  for (let s = 0; s <= maxX; s += stepTick) ctx.fillText(s.toLocaleString('en-US'), X(s), h - 8);
  ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(255,255,255,.35)';
  ctx.beginPath(); ctx.moveTo(L, Y(rand)); ctx.lineTo(w - R, Y(rand)); ctx.stroke(); ctx.setLineDash([]);
  ctx.textAlign = 'right'; ctx.fillStyle = 'rgba(255,255,255,.5)'; ctx.fillText('pure guessing', w - R, Y(rand) - 6);
  const line = (pts, color, width) => {
    if (pts.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round'; ctx.beginPath();
    pts.forEach(([s, v], i) => { const x = X(s), y = Y(Math.max(lo, Math.min(hi, v))); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.stroke();
  };
  const thin = S.losses.length > 1500 ? S.losses.filter((_, i) => i % Math.ceil(S.losses.length / 1500) === 0) : S.losses;
  line(thin, css('--lime'), 2.2);
  line(S.vals, css('--pink'), 1.8);
  for (const [s, v] of S.vals) { ctx.fillStyle = css('--pink'); ctx.beginPath(); ctx.arc(X(s), Y(Math.max(lo, Math.min(hi, v))), 2.4, 0, 7); ctx.fill(); }
}

/* embedding map: eases toward each new PCA snapshot, with axes flipped to match the last one so it doesn't jump */
let mapTarget = null, mapCur = null, mapRaf = 0;
const VOWELS = new Set('aeiouAEIOU');
function charClass(ch) {
  if (ch === ' ' || ch === '\n' || ch === '\t') return 'sp';
  if (/[aeiou]/.test(ch)) return 'vow';
  if (/[a-z]/.test(ch)) return 'con';
  if (/[A-Z]/.test(ch)) return 'up';
  if (/[0-9]/.test(ch)) return 'num';
  return 'pun';
}
const CLASS_COLOR = { vow: '--lime', con: '--ink', up: '--pink', pun: '--amber', sp: '--cyan', num: '--violet' };
function showChar(ch) { return ch === ' ' ? '␣' : ch === '\n' ? '↵' : ch === '\t' ? '⇥' : ch === '�' ? '?' : ch; }

function setMap(pca) {
  const V = S.cfg.V, pts = new Float64Array(pca);
  if (mapTarget && mapTarget.length === pts.length) {
    for (let a = 0; a < 2; a++) { let d = 0; for (let v = 0; v < V; v++) d += pts[v * 2 + a] * mapTarget[v * 2 + a]; if (d < 0) for (let v = 0; v < V; v++) pts[v * 2 + a] *= -1; }
  }
  // Scale each axis by its 90th-percentile spread, so one stray character can't squash the rest into a dot.
  for (let a = 0; a < 2; a++) {
    const abs = []; for (let v = 0; v < V; v++) abs.push(Math.abs(pts[v * 2 + a]));
    abs.sort((x, y) => x - y);
    const q = abs[Math.floor(abs.length * 0.9)] || 1e-9;
    for (let v = 0; v < V; v++) pts[v * 2 + a] = Math.max(-1.08, Math.min(1.08, pts[v * 2 + a] / q * 0.85));
  }
  mapTarget = pts;
  if (!mapCur || mapCur.length !== pts.length || reduced) mapCur = Float64Array.from(pts);
  if (!mapRaf) mapRaf = requestAnimationFrame(drawMap);
}
function drawMap() {
  mapRaf = 0;
  const cv = $('#map');
  if (!mapTarget || !cv.clientWidth) return;
  let moving = false;
  for (let i = 0; i < mapCur.length; i++) { const d = mapTarget[i] - mapCur[i]; if (Math.abs(d) > 1e-3) { mapCur[i] += d * 0.12; moving = true; } }
  const { ctx, w, h } = fit(cv), pad = 22;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,.06)'; ctx.beginPath(); ctx.moveTo(w / 2, 8); ctx.lineTo(w / 2, h - 8); ctx.moveTo(8, h / 2); ctx.lineTo(w - 8, h / 2); ctx.stroke();
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const chars = S.voc.chars;
  for (let v = 0; v < chars.length; v++) {
    const x = w / 2 + mapCur[v * 2] * (w / 2 - pad), y = h / 2 - mapCur[v * 2 + 1] * (h / 2 - pad), cl = charClass(chars[v]);
    ctx.font = (cl === 'up' || cl === 'pun' ? '500 12px' : '600 14px') + ' "JetBrains Mono", monospace';
    ctx.fillStyle = css(CLASS_COLOR[cl]);
    ctx.globalAlpha = cl === 'con' ? 0.85 : 1;
    ctx.fillText(showChar(chars[v]), x, y);
  }
  ctx.globalAlpha = 1;
  if (moving) mapRaf = requestAnimationFrame(drawMap);
}

function drawOdds(probs) {
  const top = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
  $('#odds').replaceChildren(...top.map(([p, i]) => {
    const li = document.createElement('li');
    const ch = document.createElement('code'); ch.textContent = showChar(S.voc.chars[i]);
    const tr = document.createElement('span'); tr.className = 'otrack';
    const f = document.createElement('span'); f.className = 'ofill'; f.style.width = (p * 100).toFixed(1) + '%'; tr.append(f);
    const n = document.createElement('span'); n.className = 'on'; n.textContent = (p * 100 < 1 ? (p * 100).toFixed(1) : Math.round(p * 100)) + '%';
    li.append(ch, tr, n);
    return li;
  }));
}

function drawHeads(att, ctxIds) {
  const show = Math.min(ctxIds.length, 28), off = ctxIds.length - show;
  const rows = [];
  att.forEach((heads, l) => heads.forEach((row, hh) => {
    const div = document.createElement('div'); div.className = 'head';
    const lab = document.createElement('span'); lab.className = 'hl'; lab.textContent = 'L' + (l + 1) + '·H' + (hh + 1);
    const txt = document.createElement('span'); txt.className = 'ht';
    let mx = 1e-9;
    for (let u = off; u < row.length; u++) mx = Math.max(mx, row[u]);
    for (let u = off; u < ctxIds.length; u++) {
      const s = document.createElement('span');
      s.textContent = showChar(S.voc.chars[ctxIds[u]]);
      const a = row[u] / mx;
      s.style.backgroundColor = 'rgba(198,255,61,' + (0.04 + 0.86 * a).toFixed(3) + ')';
      if (a > 0.55) s.className = 'hot';
      s.title = (row[u] * 100).toFixed(1) + '%';
      txt.append(s);
    }
    div.append(lab, txt);
    rows.push(div);
  }));
  $('#heads').replaceChildren(...rows);
}

/* ───────── talk ───────── */

let genText = '';
$('#talkForm').addEventListener('submit', e => {
  e.preventDefault();
  if (!S) return;
  const p = $('#prompt').value;
  genId++;
  genText = '';
  const out = $('#gen');
  const pr = document.createElement('span'); pr.className = 'pr'; pr.textContent = p;
  out.replaceChildren(pr);
  probe.postMessage({ type: 'generate', id: genId, cfg: S.cfg, cfgKey: S.cfgKey, params: S.params.slice(), prompt: S.voc.encode(p || '\n'), n: 300, temp: +$('#temp').value, seed: (Math.random() * 1e9) | 0 });
});
function onGen(m) {
  if (m.id !== genId) return;
  genText += S.voc.decode(m.chunk);
  const out = $('#gen');
  const last = out.lastChild;
  if (last && last.nodeType === 3) last.textContent = genText; else out.append(genText);
}
$('#temp').addEventListener('input', () => { $('#tempOut').textContent = (+$('#temp').value).toFixed(1); });
let promptTimer = 0;
$('#prompt').addEventListener('input', () => { clearTimeout(promptTimer); promptTimer = setTimeout(() => probeNow(false), 200); });

/* ───────── controls ───────── */

function status(t) { $('#status').textContent = t; }
function setLive(state) {
  const el = $('#liveDot');
  el.dataset.state = state;
  $('#liveTxt').textContent = { untrained: 'untrained', learning: 'learning', paused: 'paused', trained: 'trained' }[state];
}

$('#go').addEventListener('click', () => { if (running) { stop(); status('Paused at step ' + S.step.toLocaleString('en-US') + '.'); } else start(); });
$('#reset').addEventListener('click', () => { const k = S.kind; setup(k, textFor(k)); });
document.querySelectorAll('[data-corpus]').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('[data-corpus]').forEach(x => x.setAttribute('aria-checked', String(x === b)));
  const k = b.dataset.corpus;
  $('#owntext').hidden = k !== 'own';
  if (k === 'own') { $('#ownbox').focus(); return; }
  setup(k, textFor(k));
  start();
}));
const MIN_OWN = 2000;
$('#ownbox').addEventListener('input', () => {
  const n = $('#ownbox').value.length;
  $('#owncount').textContent = n.toLocaleString('en-US') + ' characters' + (n < MIN_OWN ? ' · ' + (MIN_OWN - n).toLocaleString('en-US') + ' more to go' : ' · ready');
  $('#owngo').disabled = n < MIN_OWN;
});
$('#owngo').addEventListener('click', () => { const t = $('#ownbox').value; if (t.length >= MIN_OWN) { setup('own', t); start(); } });
function textFor(k) { return k === 'own' ? $('#ownbox').value : window.LLM_CORPUS[k]; }

let pausedByTab = false;
document.addEventListener('visibilitychange', () => {
  if (document.hidden && running) { pausedByTab = true; stop(); status('Paused while the tab is hidden.'); }
  else if (!document.hidden && pausedByTab) { pausedByTab = false; start(); }
});
window.addEventListener('resize', () => { drawLoss(); if (mapTarget) { mapCur = Float64Array.from(mapTarget); drawMap(); } });

/* ───────── go ───────── */

if (!window.GPT || !window.LLM_CORPUS) { status('Couldn’t load the model code.'); return; }
if (makeWorkers()) {
  setup('shakespeare', window.LLM_CORPUS.shakespeare);
  // Desktops start on their own; phones and small machines wait for a tap, to spare the battery.
  if (cores >= 4 && window.matchMedia('(pointer: fine)').matches) setTimeout(start, 600);
  else status('Tap “Start training” to begin. It works hard while it runs: about ' + nW + ' core' + (nW > 1 ? 's' : '') + ' on this device.');
}
})();
