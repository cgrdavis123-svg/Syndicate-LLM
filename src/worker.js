/* Two jobs, one file.
   Trainer: holds a shard of the work. Each step it receives the current weights, samples its own batch,
   runs forward + backward and sends the gradients back; the page averages them and takes the Adam step
   (synchronous data parallelism, the way multi-GPU training works, with CPU cores as the GPUs).
   Probe: never trains. It takes a snapshot of the weights and writes samples, next-character odds,
   attention maps and the embedding map, so the page itself never stalls. */
importScripts('gpt.js?v=__V__');

let model = null, tokens = null, split = 0, r = null, batch = null, cfg = null;

self.onmessage = e => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      cfg = m.cfg;
      model = new GPT.Model(cfg);
      tokens = m.tokens; split = m.split; r = GPT.rng(m.seed);
      batch = null;
      self.postMessage({ type: 'ready', id: m.id });
    } else if (m.type === 'step') {
      model.params.set(m.params);
      const B = m.B, T = m.T;
      batch = GPT.batchFrom(tokens, 0, split, B, T, r, batch && batch.length === B * (T + 1) ? batch : null);
      const t0 = performance.now();
      const loss = model.lossAndGrad(batch, B, T);
      let val = null;
      if (m.val) {
        const vb = GPT.batchFrom(tokens, split, tokens.length, B, T, r);
        val = model.loss(vb, B, T);
      }
      const grads = model.grads.slice();
      self.postMessage({ type: 'grads', id: m.id, step: m.step, loss, val, grads, ms: performance.now() - t0 }, [grads.buffer]);
    } else if (m.type === 'probe') {
      if (!model || m.cfgKey !== (model.cfgKey || '')) { model = new GPT.Model(m.cfg); model.cfgKey = m.cfgKey; }
      model.params.set(m.params);
      const out = { type: 'probe', tag: m.tag, step: m.step };
      if (m.sample) {
        const rr = GPT.rng(m.sample.seed);
        out.sample = model.generate(m.sample.prompt, m.sample.n, m.sample.temp, rr);
      }
      if (m.inspect) {
        const ctx = m.inspect.prompt.slice(-m.cfg.C);
        out.probs = Array.from(model.next(ctx.length ? ctx : [0], m.inspect.temp));
        // Attention from the last position, per layer and head, over the context it could see.
        if (ctx.length) {
          const ws = model.forward(ctx, 0, ctx.length), T = ctx.length, H = m.cfg.H;
          out.att = ws.layers.map(w => { const rows = []; for (let h = 0; h < H; h++) rows.push(Array.from(w.att.subarray((h * T + T - 1) * T, (h * T + T) * T))); return rows; });
          out.ctx = ctx;
        }
        out.pca = Array.from(GPT.pca2(model.P.wte, m.cfg.V, m.cfg.D));
      }
      self.postMessage(out);
    } else if (m.type === 'generate') {
      if (!model || m.cfgKey !== (model.cfgKey || '')) { model = new GPT.Model(m.cfg); model.cfgKey = m.cfgKey; }
      model.params.set(m.params);
      const rr = GPT.rng(m.seed), ctx = m.prompt.slice();
      for (let i = 0; i < m.n; i += 8) {
        const chunk = model.generate(ctx, Math.min(8, m.n - i), m.temp, rr);
        ctx.push(...chunk);
        self.postMessage({ type: 'gen', id: m.id, chunk, done: i + 8 >= m.n });
      }
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
