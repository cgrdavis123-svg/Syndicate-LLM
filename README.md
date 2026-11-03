# Little Language Model

A GPT-style language model written from scratch in plain JavaScript, training live in your browser across your CPU cores. It starts out knowing nothing and you watch it go from gibberish to something that reads like Shakespeare.

**Live:** https://code-demo.grady.stusyn.com/llm/

No libraries, no autograd, no server. Every layer and every derivative is written by hand.

## What's in it

- **The model** (`src/gpt.js`): token and position embeddings, pre-norm transformer blocks with multi-head causal self-attention and a GELU MLP, a final layer norm and a tied output head. Forward and backward passes are derived and coded manually. Parameters live in one flat `Float32Array` with named slices, so a whole set of weights or gradients moves between threads as a single buffer.
- **Data-parallel training** (`src/worker.js`, `src/app.js`): each Web Worker takes the current weights, samples its own batch, runs forward + backward and sends gradients back. The page averages them and takes the AdamW step (warm-up, then cosine decay) — synchronous data parallelism, the same idea as multi-GPU training, with CPU cores as the GPUs.
- **A probe worker** that never trains: it snapshots the weights and produces samples, next-character odds, attention maps and the embedding map, so the page never stalls.
- **The page** (`src/index.html`, `src/llm.css`): loss curves, live samples, per-head attention heatmaps, a time-lapse of the same prompt at each milestone, and three things to learn from — Shakespeare, the model's own source code, or text you paste in.
- **Tests** (`test/llm.test.mjs`): every hand-written gradient is checked against finite differences in float64, plus causality (a later token never changes an earlier prediction), attention rows summing to one over the past, a fresh model starting at loss ≈ ln V, and the model actually memorising a phrase. The build runs them first and refuses to ship if one fails.

## Run it

Needs Node 18+.

```bash
npm test         # gradient checks and model tests
npm run build    # tests, then src/ → dist/ (generates corpus.js, fingerprints files)
npm run serve    # build and serve dist/ at http://localhost:8000
```

Open it over HTTP, not `file://`, so the workers can load. The page makes no network requests once loaded; its CSP blocks `fetch`, which is why the build packs the training text into `corpus.js`.

## Layout

```
src/        the app: index.html, gpt.js (model), worker.js, app.js (UI + optimizer), llm.css, fonts/
test/       node:test suite, run against src/gpt.js in a vm sandbox
corpus/     shakespeare.txt — the first ~400,000 characters of the "tiny Shakespeare" set (public domain)
build.mjs   zero-dependency build → dist/
```

## How it was made

Built by Grady Davis with AI doing most of the typing; the architecture, the math checks and the engineering decisions are mine. It started as part of the [Syndicate](https://github.com/cgrdavis123-svg/Syndicate-PUBLIC) codebase and lives here on its own.

Fonts: Inter Tight, Instrument Serif and JetBrains Mono, under the SIL Open Font License (`src/fonts/LICENSE.txt`).

© Grady Davis. All rights reserved. Shared for viewing; no license is granted to reuse or redistribute the code.
