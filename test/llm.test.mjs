// Tests for the little language model (Syndicate-LLM). Run: npm test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const ctx = { console, Math, Float32Array, Float64Array, Int32Array, Uint8Array, Map };
ctx.globalThis = ctx;
vm.runInNewContext(readFileSync(new URL('../src/gpt.js', import.meta.url), 'utf8'), ctx);
const G = ctx.GPT;

// Finite differences in float64: every tensor's hand-written gradient against (f(θ+h) − f(θ−h)) / 2h.
const tiny = { V: 7, C: 6, D: 8, H: 2, L: 2, F: 12 };
function checkModel(cfg, seed) {
  const m = new G.Model(cfg, Float64Array).init(seed);
  for (let i = 0; i < m.params.length; i++) m.params[i] += (Math.sin(i * 12.9898 + seed) * 0.3);   // move off the symmetric init
  const r = G.rng(seed), B = 3, T = cfg.C;
  const batch = Int32Array.from({ length: B * (T + 1) }, () => Math.floor(r() * cfg.V));
  m.lossAndGrad(batch, B, T);
  const grad = Float64Array.from(m.grads);
  const worst = {};
  for (const s of m.lay.spec) {
    let w = 0;
    for (let k = 0; k < Math.min(s.size, 12); k++) {
      const i = s.off + Math.floor(r() * s.size), h = 1e-5, old = m.params[i];
      m.params[i] = old + h; const a = m.loss(batch, B, T);
      m.params[i] = old - h; const b = m.loss(batch, B, T);
      m.params[i] = old;
      const num = (a - b) / (2 * h), an = grad[i];
      const rel = Math.abs(num - an) / Math.max(1e-7, Math.abs(num) + Math.abs(an));
      if (Math.abs(num - an) > 1e-8) w = Math.max(w, rel);
    }
    worst[s.name] = w;
  }
  return worst;
}

for (const [label, seed] of [['seed 1', 1], ['seed 2', 2]]) {
  test('gradient check, every tensor, ' + label, () => {
    const worst = checkModel(tiny, seed);
    for (const [name, rel] of Object.entries(worst)) assert.ok(rel < 1e-5, `${name}: relative error ${rel.toExponential(2)}`);
    assert.equal(Object.keys(worst).length, 4 + 12 * tiny.L);
  });
}
test('gradient check, one layer, one head, longer context', () => {
  const worst = checkModel({ V: 5, C: 9, D: 6, H: 1, L: 1, F: 10 }, 3);
  for (const [name, rel] of Object.entries(worst)) assert.ok(rel < 1e-5, `${name}: ${rel}`);
});

test('fresh model predicts close to uniform: loss ≈ ln V', () => {
  const m = new G.Model({ V: 50 }).init(4), r = G.rng(9);
  const batch = Int32Array.from({ length: 4 * 65 }, () => Math.floor(r() * 50));
  assert.ok(Math.abs(m.loss(batch, 4, 64) - Math.log(50)) < 0.05);
});

test('causal: a later token never changes an earlier prediction', () => {
  const m = new G.Model(tiny, Float64Array).init(5);
  const a = [1, 2, 3, 4, 5, 6], b = [1, 2, 3, 0, 0, 0];
  const la = Array.from(m.forward(a, 0, 6).logits.slice(0, 3 * tiny.V));
  const lb = Array.from(m.forward(b, 0, 6).logits.slice(0, 3 * tiny.V));
  la.forEach((x, i) => assert.ok(Math.abs(x - lb[i]) < 1e-12));
});

test('attention rows are probability distributions over the past only', () => {
  const m = new G.Model(tiny, Float64Array).init(6);
  const ws = m.forward([1, 2, 3, 4, 5], 0, 5);
  const att = ws.layers[0].att, T = 5;
  for (let h = 0; h < tiny.H; h++) for (let t = 0; t < T; t++) {
    let s = 0;
    for (let u = 0; u < T; u++) { const a = att[(h * T + t) * T + u]; if (u > t) assert.equal(a, 0); s += a; }
    assert.ok(Math.abs(s - 1) < 1e-12);
  }
});

test('vocabulary round-trips text and caps rare characters', () => {
  const v = G.vocab('hello, world\n');
  assert.equal(v.decode(v.encode('hello, world\n')), 'hello, world\n');
  const many = Array.from({ length: 200 }, (_, i) => String.fromCharCode(0x4e00 + i)).join('') + 'aaaa';
  assert.equal(G.vocab(many).chars.length, 111);
});

test('learns: memorises a repeating phrase', () => {
  const text = 'the cat sat on the mat. '.repeat(40);
  const v = G.vocab(text), tokens = v.encode(text);
  const m = new G.Model({ V: v.chars.length, C: 24, D: 32, H: 4, L: 1, F: 64 }).init(7), opt = new G.Adam(m, { lr: 1e-2 }), r = G.rng(8);
  let first, last;
  for (let s = 0; s < 120; s++) {
    const b = G.batchFrom(tokens, 0, tokens.length, 8, 24, r);
    const loss = m.lossAndGrad(b, 8, 24);
    if (s === 0) first = loss;
    last = loss;
    opt.step(m.params, m.grads, G.lrAt(s, 1e-2, 10, 120));
  }
  assert.ok(first > 2.2 && last < 0.3, `loss ${first.toFixed(2)} → ${last.toFixed(2)}`);
  assert.equal(v.decode(m.generate(v.encode('the cat sat on '), 8, 0.01)), 'the mat.');
});

test('PCA of embeddings returns finite 2D points', () => {
  const m = new G.Model({ V: 20 }).init(2);
  const pts = G.pca2(m.P.wte, 20, 64);
  assert.equal(pts.length, 40);
  assert.ok([...pts].every(Number.isFinite));
});

test('learning-rate schedule: warm-up, then cosine to a tenth', () => {
  assert.ok(G.lrAt(0, 1, 10, 100) < G.lrAt(9, 1, 10, 100));
  assert.equal(G.lrAt(9, 1, 10, 100), 1);
  assert.ok(Math.abs(G.lrAt(100, 1, 10, 100) - 0.1) < 1e-12);
});
