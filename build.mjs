// Builds the Little Language Model from src/ into dist/. No dependencies.
// __V__ fingerprints every file (the page and its workers load each other, so one version for the app),
// __LINES__ / __TESTS__ are counted from the source, and the test suite runs first, so a failing
// gradient check stops the build. corpus.js is generated here: the CSP blocks fetch, so the training
// text ships as a script. The build fails on any unfilled __X__.
import { readFileSync, writeFileSync, mkdirSync, rmSync, readdirSync, copyFileSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const SRC = 'src', OUT = 'dist', CODE = ['gpt.js', 'app.js', 'worker.js', 'llm.css'];
rmSync(OUT, { recursive: true, force: true });
mkdirSync(`${OUT}/fonts`, { recursive: true });
for (const f of readdirSync(`${SRC}/fonts`)) copyFileSync(`${SRC}/fonts/${f}`, `${OUT}/fonts/${f}`);

let tap;
try { tap = execFileSync(process.execPath, ['--test', '--test-reporter=tap', 'test/llm.test.mjs'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }); }
catch (e) { process.stdout.write(e.stdout || ''); throw new Error('Tests failed'); }
const pass = +(/^# pass (\d+)/m.exec(tap) || [])[1], fail = +(/^# fail (\d+)/m.exec(tap) || [])[1];
if (!pass || fail) throw new Error('Tests failed');

const files = Object.fromEntries(readdirSync(SRC).filter(f => statSync(`${SRC}/${f}`).isFile()).map(f => [f, readFileSync(`${SRC}/${f}`, 'utf8')]));
files['corpus.js'] = '/* Training text. Shakespeare (public domain) and this model\'s own source code. */\nself.LLM_CORPUS = ' +
  JSON.stringify({ shakespeare: readFileSync('corpus/shakespeare.txt', 'utf8'), code: ['gpt.js', 'worker.js', 'app.js'].map(f => files[f]).join('\n') }) + ';\n';

const counts = {
  __V__: createHash('sha256').update(Object.keys(files).sort().map(f => files[f]).join('')).digest('hex').slice(0, 10),
  __LINES__: CODE.reduce((n, f) => n + files[f].split('\n').filter(l => l.trim()).length, 0).toLocaleString('en-US'),
  __TESTS__: String(pass),
};
for (const [f, text] of Object.entries(files)) {
  let s = text;
  if (f !== 'corpus.js') {
    for (const [k, v] of Object.entries(counts)) s = s.split(k).join(v);
    if (/__[A-Z]+__/.test(s.replace(/__proto__/g, ''))) throw new Error(`Unfilled placeholder in ${f}`);
  }
  writeFileSync(`${OUT}/${f}`, s);
}
console.log(`Little Language Model → ${OUT}/ (${counts.__LINES__} lines, ${pass} tests passed)`);
