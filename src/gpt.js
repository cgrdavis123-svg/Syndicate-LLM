/* A small GPT-style language model, written out by hand: token and position embeddings, pre-norm transformer
   blocks (multi-head causal self-attention + GELU MLP), a final layer norm and a tied output head, with every
   backward pass derived and coded manually. No libraries, no autograd. The same file runs in the training
   workers, on the page (for sampling and the visualisations) and in the Node gradient-check tests. */
(function (root) {
'use strict';

const DEFAULTS = { V: 65, C: 64, D: 64, H: 4, L: 2, F: 256 };

/* ───────── parameters: one flat array, named slices ───────── */

function layout(cfg) {
  const { V, C, D, L, F } = cfg, spec = [];
  const add = (name, size, init) => spec.push({ name, size, init });
  add('wte', V * D, 'normal');
  add('wpe', C * D, 'normal');
  for (let l = 0; l < L; l++) {
    add(`l${l}.ln1g`, D, 'one'); add(`l${l}.ln1b`, D, 'zero');
    add(`l${l}.wqkv`, D * 3 * D, 'normal'); add(`l${l}.bqkv`, 3 * D, 'zero');
    add(`l${l}.wo`, D * D, 'resid'); add(`l${l}.bo`, D, 'zero');
    add(`l${l}.ln2g`, D, 'one'); add(`l${l}.ln2b`, D, 'zero');
    add(`l${l}.w1`, D * F, 'normal'); add(`l${l}.b1`, F, 'zero');
    add(`l${l}.w2`, F * D, 'resid'); add(`l${l}.b2`, D, 'zero');
  }
  add('lnfg', D, 'one'); add('lnfb', D, 'zero');
  let off = 0;
  const map = {};
  for (const s of spec) { s.off = off; map[s.name] = s; off += s.size; }
  return { spec, map, size: off };
}

function views(arr, lay, L) {
  const v = {};
  for (const s of lay.spec) v[s.name] = arr.subarray(s.off, s.off + s.size);
  v.layers = [];
  for (let l = 0; l < L; l++) {
    const o = {};
    for (const k of ['ln1g', 'ln1b', 'wqkv', 'bqkv', 'wo', 'bo', 'ln2g', 'ln2b', 'w1', 'b1', 'w2', 'b2']) o[k] = v[`l${l}.${k}`];
    v.layers.push(o);
  }
  return v;
}

function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let x = a; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
}
function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }

/* ───────── the building blocks, forward and backward ───────── */

function layernormF(x, out, g, b, mean, rstd, T, D) {
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let m = 0;
    for (let i = 0; i < D; i++) m += x[o + i];
    m /= D;
    let v = 0;
    for (let i = 0; i < D; i++) { const d = x[o + i] - m; v += d * d; }
    const r = 1 / Math.sqrt(v / D + 1e-5);
    mean[t] = m; rstd[t] = r;
    for (let i = 0; i < D; i++) out[o + i] = (x[o + i] - m) * r * g[i] + b[i];
  }
}
// Adds into dx (the residual stream's gradient arrives there already).
function layernormB(dout, x, g, mean, rstd, dx, dg, db, T, D) {
  for (let t = 0; t < T; t++) {
    const o = t * D, m = mean[t], r = rstd[t];
    let s1 = 0, s2 = 0;
    for (let i = 0; i < D; i++) {
      const xh = (x[o + i] - m) * r, d = dout[o + i], dxh = d * g[i];
      dg[i] += d * xh; db[i] += d;
      s1 += dxh; s2 += dxh * xh;
    }
    s1 /= D; s2 /= D;
    for (let i = 0; i < D; i++) {
      const xh = (x[o + i] - m) * r;
      dx[o + i] += r * (dout[o + i] * g[i] - s1 - xh * s2);
    }
  }
}

// out[T×N] = in[T×K] · W[K×N] + b, read through WT (W transposed, [N×K]) so every output is one dot product
// with a running sum in a register instead of a read-modify-write per multiply.
function linearF(inp, WT, bias, out, T, K, N) {
  for (let t = 0; t < T; t++) {
    const io = t * K, oo = t * N;
    for (let n = 0; n < N; n++) {
      const wo = n * K;
      let s = bias[n];
      for (let k = 0; k < K; k++) s += inp[io + k] * WT[wo + k];
      out[oo + n] = s;
    }
  }
}
// dIn += dOut · Wᵀ (rows of W are contiguous); dW += Inᵀ · dOut and db += Σ dOut, done on transposed copies
// (inT [K×T], doutT [N×T]) so both are contiguous dot products too.
function transpose(src, dst, R, Cc) { for (let r = 0; r < R; r++) { const o = r * Cc; for (let c = 0; c < Cc; c++) dst[c * R + r] = src[o + c]; } }
function linearB(dout, inp, W, dinp, dW, db, T, K, N, inT, doutT) {
  for (let t = 0; t < T; t++) {
    const oo = t * N, io = t * K;
    for (let k = 0; k < K; k++) {
      const ko = k * N;
      let s = 0;
      for (let n = 0; n < N; n++) s += dout[oo + n] * W[ko + n];
      dinp[io + k] += s;
    }
  }
  transpose(inp, inT, T, K);
  transpose(dout, doutT, T, N);
  for (let n = 0; n < N; n++) { const o = n * T; let s = 0; for (let t = 0; t < T; t++) s += doutT[o + t]; db[n] += s; }
  for (let k = 0; k < K; k++) {
    const ko = k * T, wo = k * N;
    for (let n = 0; n < N; n++) {
      const no = n * T;
      let s = 0;
      for (let t = 0; t < T; t++) s += inT[ko + t] * doutT[no + t];
      dW[wo + n] += s;
    }
  }
}

// Causal multi-head attention over a packed [T × 3D] q|k|v buffer. att keeps the softmax weights [H × T × T].
function attnF(qkv, att, out, T, D, H) {
  const hs = D / H, sc = 1 / Math.sqrt(hs), D3 = 3 * D;
  out.fill(0);
  for (let h = 0; h < H; h++) {
    for (let t = 0; t < T; t++) {
      const qo = t * D3 + h * hs, ao = (h * T + t) * T;
      let mx = -Infinity;
      for (let u = 0; u <= t; u++) {
        const ko = u * D3 + D + h * hs;
        let s = 0;
        for (let i = 0; i < hs; i++) s += qkv[qo + i] * qkv[ko + i];
        s *= sc;
        att[ao + u] = s;
        if (s > mx) mx = s;
      }
      let sum = 0;
      for (let u = 0; u <= t; u++) { const e = Math.exp(att[ao + u] - mx); att[ao + u] = e; sum += e; }
      for (let u = 0; u <= t; u++) att[ao + u] /= sum;
      for (let u = t + 1; u < T; u++) att[ao + u] = 0;
      const oo = t * D + h * hs;
      for (let u = 0; u <= t; u++) {
        const a = att[ao + u], vo = u * D3 + 2 * D + h * hs;
        for (let i = 0; i < hs; i++) out[oo + i] += a * qkv[vo + i];
      }
    }
  }
}
function attnB(dout, qkv, att, dqkv, tmp, T, D, H) {
  const hs = D / H, sc = 1 / Math.sqrt(hs), D3 = 3 * D;
  for (let h = 0; h < H; h++) {
    for (let t = 0; t < T; t++) {
      const ao = (h * T + t) * T, oo = t * D + h * hs, qo = t * D3 + h * hs;
      let dot = 0;
      for (let u = 0; u <= t; u++) {
        const vo = u * D3 + 2 * D + h * hs, a = att[ao + u];
        let s = 0;
        for (let i = 0; i < hs; i++) { const d = dout[oo + i]; s += d * qkv[vo + i]; dqkv[vo + i] += a * d; }
        tmp[u] = s; dot += a * s;
      }
      for (let u = 0; u <= t; u++) {
        const ds = att[ao + u] * (tmp[u] - dot) * sc, ko = u * D3 + D + h * hs;
        for (let i = 0; i < hs; i++) { dqkv[qo + i] += ds * qkv[ko + i]; dqkv[ko + i] += ds * qkv[qo + i]; }
      }
    }
  }
}

// GELU in its sigmoid form, x·σ(1.702x): the same curve to within about 0.02, and one exp instead of a tanh.
function geluF(x, y, n) { for (let i = 0; i < n; i++) { const v = x[i]; y[i] = v / (1 + Math.exp(-1.702 * v)); } }
function geluB(x, dy, dx, n) {
  for (let i = 0; i < n; i++) { const v = x[i], sg = 1 / (1 + Math.exp(-1.702 * v)); dx[i] = dy[i] * (sg + 1.702 * v * sg * (1 - sg)); }
}

/* ───────── the model ───────── */

function Model(cfg, ArrayT) {
  this.cfg = Object.assign({}, DEFAULTS, cfg);
  if (this.cfg.D % this.cfg.H) throw new Error('D must divide by H');
  this.A = ArrayT || Float32Array;
  this.lay = layout(this.cfg);
  this.params = new this.A(this.lay.size);
  this.grads = new this.A(this.lay.size);
  this.P = views(this.params, this.lay, this.cfg.L);
  this.G = views(this.grads, this.lay, this.cfg.L);
  this.ws = null;
  const { D, L, F } = this.cfg, A = this.A;
  this.WT = [];
  for (let l = 0; l < L; l++) this.WT.push({ wqkv: new A(3 * D * D), wo: new A(D * D), w1: new A(F * D), w2: new A(D * F) });
}

// Transposed copies of the weight matrices for the forward pass. Rebuilt on every forward: it costs about 1%
// of a training step and means nothing can ever read stale weights.
Model.prototype.transposeWeights = function () {
  const { D, L, F } = this.cfg;
  for (let l = 0; l < L; l++) {
    const p = this.P.layers[l], wt = this.WT[l];
    transpose(p.wqkv, wt.wqkv, D, 3 * D); transpose(p.wo, wt.wo, D, D); transpose(p.w1, wt.w1, D, F); transpose(p.w2, wt.w2, F, D);
  }
};

Model.prototype.init = function (seed) {
  const r = rng(seed || 1), { L } = this.cfg;
  for (const s of this.lay.spec) {
    const a = this.params.subarray(s.off, s.off + s.size);
    if (s.init === 'one') a.fill(1);
    else if (s.init === 'zero') a.fill(0);
    else { const std = s.init === 'resid' ? 0.02 / Math.sqrt(2 * L) : 0.02; for (let i = 0; i < a.length; i++) a[i] = gauss(r) * std; }
  }
  return this;
};

Model.prototype.workspace = function (T) {
  if (this.ws && this.ws.T === T) return this.ws;
  const { V, D, H, L, F } = this.cfg, A = this.A;
  const ws = { T, res: [], layers: [], lnf: new A(T * D), mf: new A(T), rf: new A(T), logits: new A(T * V),
    dA: new A(T * D), dB: new A(T * D), d1: new A(T * D), d2: new A(T * 3 * D), dh1: new A(T * F), dh2: new A(T * F), dlnf: new A(T * D), tmp: new A(T), tA: new A(Math.max(F, 3 * D) * T), tB: new A(Math.max(F, 3 * D) * T) };
  for (let l = 0; l <= L; l++) ws.res.push(new A(T * D));
  for (let l = 0; l < L; l++) ws.layers.push({ ln1: new A(T * D), m1: new A(T), r1: new A(T), qkv: new A(T * 3 * D), att: new A(H * T * T),
    ao: new A(T * D), x1: new A(T * D), ln2: new A(T * D), m2: new A(T), r2: new A(T), hpre: new A(T * F), hact: new A(T * F) });
  this.ws = ws;
  return ws;
};

// Runs tokens[o .. o+T) through the network; logits for every position land in ws.logits.
Model.prototype.forward = function (tok, o, T) {
  const { V, D, H, L, F, C } = this.cfg, P = this.P;
  if (T > C) throw new Error('Sequence longer than the context');
  const ws = this.workspace(T), e = ws.res[0];
  if (!this.noTranspose) this.transposeWeights();
  for (let t = 0; t < T; t++) {
    const id = tok[o + t], eo = id * D, po = t * D, xo = t * D;
    for (let i = 0; i < D; i++) e[xo + i] = P.wte[eo + i] + P.wpe[po + i];
  }
  for (let l = 0; l < L; l++) {
    const w = ws.layers[l], p = P.layers[l], wt = this.WT[l], x = ws.res[l], y = ws.res[l + 1];
    layernormF(x, w.ln1, p.ln1g, p.ln1b, w.m1, w.r1, T, D);
    linearF(w.ln1, wt.wqkv, p.bqkv, w.qkv, T, D, 3 * D);
    attnF(w.qkv, w.att, w.ao, T, D, H);
    linearF(w.ao, wt.wo, p.bo, w.x1, T, D, D);
    for (let i = 0; i < T * D; i++) w.x1[i] += x[i];
    layernormF(w.x1, w.ln2, p.ln2g, p.ln2b, w.m2, w.r2, T, D);
    linearF(w.ln2, wt.w1, p.b1, w.hpre, T, D, F);
    geluF(w.hpre, w.hact, T * F);
    linearF(w.hact, wt.w2, p.b2, y, T, F, D);
    for (let i = 0; i < T * D; i++) y[i] += w.x1[i];
  }
  layernormF(ws.res[L], ws.lnf, P.lnfg, P.lnfb, ws.mf, ws.rf, T, D);
  const wte = P.wte;
  for (let t = 0; t < T; t++) {
    const xo = t * D, lo = t * V;
    for (let v = 0; v < V; v++) {
      const eo = v * D;
      let s = 0;
      for (let i = 0; i < D; i++) s += ws.lnf[xo + i] * wte[eo + i];
      ws.logits[lo + v] = s;
    }
  }
  return ws;
};

// Forward + backward for one sequence; targets are tok[o+1 .. o+T]. Adds scale·∂loss/∂θ into this.grads.
Model.prototype.backward = function (tok, o, T, scale) {
  const { V, D, H, L, F } = this.cfg, P = this.P, G = this.G;
  const ws = this.forward(tok, o, T);
  let loss = 0;
  const lg = ws.logits;
  for (let t = 0; t < T; t++) {
    const lo = t * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (lg[lo + v] > mx) mx = lg[lo + v];
    let sum = 0;
    for (let v = 0; v < V; v++) { const e = Math.exp(lg[lo + v] - mx); lg[lo + v] = e; sum += e; }
    const target = tok[o + t + 1];
    for (let v = 0; v < V; v++) lg[lo + v] /= sum;
    loss -= Math.log(Math.max(lg[lo + target], 1e-30));
    for (let v = 0; v < V; v++) lg[lo + v] = (lg[lo + v] - (v === target ? 1 : 0)) * scale;   // now ∂loss/∂logit
  }
  // tied head: logits = lnf · wteᵀ
  const dlnf = ws.dlnf;
  dlnf.fill(0);
  for (let t = 0; t < T; t++) {
    const xo = t * D, lo = t * V;
    for (let v = 0; v < V; v++) {
      const d = lg[lo + v], eo = v * D;
      for (let i = 0; i < D; i++) { dlnf[xo + i] += d * P.wte[eo + i]; G.wte[eo + i] += d * ws.lnf[xo + i]; }
    }
  }
  let dres = ws.dA, dx1 = ws.dB;
  dres.fill(0);
  layernormB(dlnf, ws.res[L], P.lnfg, ws.mf, ws.rf, dres, G.lnfg, G.lnfb, T, D);
  for (let l = L - 1; l >= 0; l--) {
    const w = ws.layers[l], p = P.layers[l], g = G.layers[l], x = ws.res[l];
    ws.dh1.fill(0);
    linearB(dres, w.hact, p.w2, ws.dh1, g.w2, g.b2, T, F, D, ws.tA, ws.tB);
    geluB(w.hpre, ws.dh1, ws.dh2, T * F);
    ws.d1.fill(0);
    linearB(ws.dh2, w.ln2, p.w1, ws.d1, g.w1, g.b1, T, D, F, ws.tA, ws.tB);
    dx1.set(dres);
    layernormB(ws.d1, w.x1, p.ln2g, w.m2, w.r2, dx1, g.ln2g, g.ln2b, T, D);
    ws.d1.fill(0);
    linearB(dx1, w.ao, p.wo, ws.d1, g.wo, g.bo, T, D, D, ws.tA, ws.tB);
    ws.d2.fill(0);
    attnB(ws.d1, w.qkv, w.att, ws.d2, ws.tmp, T, D, H);
    const dln1 = ws.d1;
    dln1.fill(0);
    linearB(ws.d2, w.ln1, p.wqkv, dln1, g.wqkv, g.bqkv, T, D, 3 * D, ws.tA, ws.tB);
    dres.set(dx1);
    layernormB(dln1, x, p.ln1g, w.m1, w.r1, dres, g.ln1g, g.ln1b, T, D);
  }
  for (let t = 0; t < T; t++) {
    const id = tok[o + t], eo = id * D, xo = t * D;
    for (let i = 0; i < D; i++) { G.wte[eo + i] += dres[xo + i]; G.wpe[xo + i] += dres[xo + i]; }
  }
  return loss;
};

// Mean cross-entropy over B sequences packed back to back, each T+1 tokens long. Gradients are zeroed first.
Model.prototype.lossAndGrad = function (batch, B, T) {
  this.grads.fill(0);
  this.transposeWeights();
  this.noTranspose = true;
  let loss = 0;
  const scale = 1 / (B * T);
  try { for (let b = 0; b < B; b++) loss += this.backward(batch, b * (T + 1), T, scale); } finally { this.noTranspose = false; }
  return loss / (B * T);
};

// Loss without gradients (validation).
Model.prototype.loss = function (batch, B, T) {
  const { V } = this.cfg;
  this.transposeWeights();
  this.noTranspose = true;
  try { return this.lossNoT(batch, B, T); } finally { this.noTranspose = false; }
};
Model.prototype.lossNoT = function (batch, B, T) {
  const { V } = this.cfg;
  let loss = 0;
  for (let b = 0; b < B; b++) {
    const o = b * (T + 1), ws = this.forward(batch, o, T), lg = ws.logits;
    for (let t = 0; t < T; t++) {
      const lo = t * V;
      let mx = -Infinity;
      for (let v = 0; v < V; v++) if (lg[lo + v] > mx) mx = lg[lo + v];
      let sum = 0;
      for (let v = 0; v < V; v++) sum += Math.exp(lg[lo + v] - mx);
      loss -= lg[lo + batch[o + t + 1]] - mx - Math.log(sum);
    }
  }
  return loss / (B * T);
};

// Next-token probabilities after a context (last C tokens), at a temperature.
Model.prototype.next = function (ctx, temp) {
  const { V, C } = this.cfg;
  const s = ctx.length > C ? ctx.slice(ctx.length - C) : ctx;
  const T = Math.max(1, s.length);
  const ws = this.forward(s.length ? s : [0], 0, T);
  const lo = (T - 1) * V, p = new Float64Array(V);
  let mx = -Infinity;
  for (let v = 0; v < V; v++) if (ws.logits[lo + v] > mx) mx = ws.logits[lo + v];
  let sum = 0;
  const tt = Math.max(temp || 1, 1e-3);
  for (let v = 0; v < V; v++) { p[v] = Math.exp((ws.logits[lo + v] - mx) / tt); sum += p[v]; }
  for (let v = 0; v < V; v++) p[v] /= sum;
  return p;
};

Model.prototype.generate = function (ctx, n, temp, r) {
  const out = ctx.slice();
  r = r || Math.random;
  for (let k = 0; k < n; k++) {
    const p = this.next(out, temp);
    let x = r(), v = 0;
    for (; v < p.length - 1; v++) { x -= p[v]; if (x <= 0) break; }
    out.push(v);
  }
  return out.slice(ctx.length);
};

/* ───────── optimiser ───────── */

// AdamW with global-norm clipping. Decay only on the weight matrices, as GPT-2 does.
function Adam(model, opts) {
  this.o = Object.assign({ lr: 3e-3, b1: 0.9, b2: 0.99, eps: 1e-8, wd: 0.01, clip: 1 }, opts);
  this.m = new Float32Array(model.lay.size);
  this.v = new Float32Array(model.lay.size);
  this.t = 0;
  this.decay = new Uint8Array(model.lay.size);
  for (const s of model.lay.spec) if (s.init === 'normal' || s.init === 'resid') this.decay.fill(1, s.off, s.off + s.size);
}
Adam.prototype.step = function (params, grads, lr) {
  const o = this.o, n = params.length;
  let norm = 0;
  for (let i = 0; i < n; i++) norm += grads[i] * grads[i];
  norm = Math.sqrt(norm);
  const c = norm > o.clip ? o.clip / norm : 1;
  this.t++;
  const bc1 = 1 - Math.pow(o.b1, this.t), bc2 = 1 - Math.pow(o.b2, this.t), m = this.m, v = this.v;
  for (let i = 0; i < n; i++) {
    const g = grads[i] * c;
    m[i] = o.b1 * m[i] + (1 - o.b1) * g;
    v[i] = o.b2 * v[i] + (1 - o.b2) * g * g;
    const upd = (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + o.eps);
    params[i] -= lr * (upd + (this.decay[i] ? o.wd * params[i] : 0));
  }
  return norm;
};

// Warm up for the first steps, then a slow cosine down to a tenth.
function lrAt(step, base, warm, total) {
  if (step < warm) return base * (step + 1) / warm;
  const p = Math.min(1, (step - warm) / Math.max(1, total - warm));
  return base * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * p)));
}

/* ───────── text ↔ tokens ───────── */

// Character-level vocabulary: every character in the text, capped at the most frequent 110 (+1 for "other").
function vocab(text) {
  const counts = new Map();
  for (const ch of text) counts.set(ch, (counts.get(ch) || 0) + 1);
  let chars = [...counts.keys()];
  if (chars.length > 110) chars = chars.sort((a, b) => counts.get(b) - counts.get(a)).slice(0, 110).concat(['�']);
  chars.sort();
  const index = new Map(chars.map((c, i) => [c, i]));
  const unk = index.has('�') ? index.get('�') : 0;
  return { chars, encode: s => { const out = []; for (const ch of s) out.push(index.has(ch) ? index.get(ch) : unk); return out; }, decode: ids => ids.map(i => chars[i]).join('') };
}

function batchFrom(tokens, lo, hi, B, T, r, out) {
  out = out || new Int32Array(B * (T + 1));
  for (let b = 0; b < B; b++) {
    const s = lo + Math.floor(r() * (hi - lo - T - 1));
    for (let t = 0; t <= T; t++) out[b * (T + 1) + t] = tokens[s + t];
  }
  return out;
}

// Top two principal directions of the embedding table, by power iteration on the D×D covariance.
function pca2(wte, V, D) {
  const mean = new Float64Array(D);
  for (let v = 0; v < V; v++) for (let i = 0; i < D; i++) mean[i] += wte[v * D + i] / V;
  const cov = new Float64Array(D * D);
  for (let v = 0; v < V; v++) for (let i = 0; i < D; i++) {
    const a = wte[v * D + i] - mean[i];
    for (let j = 0; j < D; j++) cov[i * D + j] += a * (wte[v * D + j] - mean[j]);
  }
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let x = new Float64Array(D).map((_, i) => Math.sin(i * 7.13 + c * 3.1) + 0.01);
    for (let it = 0; it < 60; it++) {
      const y = new Float64Array(D);
      for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += cov[i * D + j] * x[j]; y[i] = s; }
      for (const p of comps) { let d = 0; for (let i = 0; i < D; i++) d += y[i] * p[i]; for (let i = 0; i < D; i++) y[i] -= d * p[i]; }
      let n = 0; for (let i = 0; i < D; i++) n += y[i] * y[i]; n = Math.sqrt(n) || 1;
      for (let i = 0; i < D; i++) x[i] = y[i] / n;
    }
    comps.push(x);
  }
  const pts = new Float64Array(V * 2);
  for (let v = 0; v < V; v++) for (let c = 0; c < 2; c++) { let s = 0; for (let i = 0; i < D; i++) s += (wte[v * D + i] - mean[i]) * comps[c][i]; pts[v * 2 + c] = s; }
  return pts;
}

const api = { Model, Adam, lrAt, vocab, batchFrom, pca2, rng, layout, DEFAULTS };
if (typeof module !== 'undefined' && module.exports) module.exports = api;
root.GPT = api;
})(typeof self !== 'undefined' ? self : globalThis);
