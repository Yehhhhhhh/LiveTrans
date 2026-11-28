// Translation: the final translation of each sentence (a queue, a few at a time), and live drafts
// of the segment still being spoken (提前翻译), which the final pass reuses when the text matches.
import * as db from './db.js';
import { $, setPill, showTrMsg, clearTrMsg } from './dom.js';
import { settings, state, glossary } from './state.js';
import { drafts, renderSeg, renderLiveTrans } from './transcript.js';
import { liveKey, splitSentences } from './lib/text.js';
import { buildTranslationRequest, pickContext, postTranslate } from './lib/translate.js';

export function enqueueTranslation(seg) {
  seg.trStatus = 'pending';
  seg.error = '';
  renderSeg(seg);
  state.trQueue.push(seg);
  pumpTranslations();
}

// Final translations run several at a time, so one slow reply doesn't hold up every line after
// it. Each line fills in as soon as its own translation is done; a request that starts while
// earlier lines are still being translated gets their source text as context (see contextFor).
// The local model runs one at a time: it shares the GPU with Whisper, and Ollama would queue
// parallel requests anyway unless configured otherwise.
const TR_PARALLEL_API = 2;
const trParallel = () => (settings.tr.useLocal ? 1 : TR_PARALLEL_API);
const trBusy = () => state.trActive > 0 || state.trQueue.length > 0;

function pumpTranslations() {
  while (state.trQueue.length && state.trActive < trParallel()) {
    const seg = state.trQueue.shift();
    state.trActive++;
    translateSeg(seg).finally(() => {
      state.trActive--;
      pumpTranslations();
    });
  }
  // Live drafts only run while no final translation is waiting.
  if (!trBusy()) pumpLive();
}

export function translationRequest(text, context) {
  return buildTranslationRequest(settings.tr, glossary, text, context);
}

// Context: the previous segments of the same session (all of the current session's when seg is
// null, i.e. for the segment still being spoken); see pickContext.
async function contextFor(seg) {
  const list = !seg || seg.sessionId === state.session?.id ? state.segments : await db.getSegments(seg.sessionId);
  return pickContext(list, seg, settings.tr.contextSize);
}

async function translateSeg(seg) {
  // A live request already translating exactly this text is taken over rather than restarted.
  const req = adopted.get(seg.id);
  adopted.delete(seg.id);
  // An adopted request was sent earlier: its delay counts from then.
  const t0 = req ? req.t0 : performance.now();
  let first = 0;
  const onText = (text) => {
    if (!first) first = performance.now() - t0;
    // With live translation on, text is never streamed in: a draft (or the
    // placeholder) is swapped for the final translation in one go.
    if (drafts.has(seg.id) || settings.tr.live) return;
    seg.translation = text;
    seg.trStatus = 'streaming';
    renderSeg(seg);
  };
  try {
    if (req) {
      req.onText = onText;
      seg.translation = await req.promise;
    } else {
      seg.translation = await postTranslate(translationRequest(seg.text, await contextFor(seg)), onText);
    }
    seg.trStatus = 'done';
    clearTrMsg();
    const total = Math.round(performance.now() - t0);
    seg.trMs = total; // translation delay: text sent -> translation received
    setPill($('stTr'), `翻译：首字 ${Math.round(first || total)} ms · 共 ${total} ms · 队列 ${state.trQueue.length}`, 'ok');
    $('stTr').title = '从文字发送出去翻译，到收到第一个字 / 完整译文所用的时间。';
  } catch (e) {
    seg.trStatus = 'error';
    seg.error = e.message;
    setPill($('stTr'), '翻译：失败', 'err');
  }
  drafts.delete(seg.id);
  renderSeg(seg);
  await db.putSegment(seg);
}

// ---------------------------------------------------------------- live translation

// While a segment is still being spoken, each sentence of its partial transcript
// is translated so drafts are on screen before the final transcript arrives.
// One request at a time, earliest untranslated sentence first (finished sentences
// are stable; only the last one keeps changing). Final sentences whose text was
// already translated reuse that result.
const newLive = (index) => ({
  index,
  sentences: [],
  done: new Map(), // sentence text -> translation
  ms: new Map(), // sentence text -> how long that translation took (sent -> received)
  failed: new Set(), // sentence texts not to retry; the final pass still follows
  shown: [], // per sentence position: the finished draft on screen
  req: null, // the draft translation request in flight
});
export let live = newLive(-1);
const adopted = new Map(); // seg.id -> live request whose text equals the final sentence

export function resetLive(index = -1) {
  live.req?.ctrl.abort();
  live = newLive(index);
}

export function liveTranslate(index, text) {
  if (!settings.tr.enabled || !settings.tr.live || !state.session) return;
  if (index !== live.index) resetLive(index);
  live.sentences = splitSentences(text.trim());
  pumpLive();
}

function showLive() {
  const text = live.shown
    .slice(0, live.sentences.length)
    .map((d) => d || '')
    .join('');
  if (text) renderLiveTrans(live.index, text);
}

function pumpLive() {
  // Final translations go first; this runs again when their queue drains.
  if (live.req || trBusy()) return;
  const pos = live.sentences.findIndex((s) => s && !live.done.has(liveKey(s)) && !live.failed.has(liveKey(s)));
  if (pos < 0) return;
  const text = live.sentences[pos];
  const req = {
    text,
    t0: performance.now(),
    ctrl: new AbortController(),
    // Drafts never stream: each one appears whole and replaces the previous in one go.
    // (Replaced by the segment's handler if this request is adopted for a final sentence.)
    onText: () => {},
  };
  req.promise = contextFor(null).then((ctx) =>
    postTranslate(translationRequest(text, ctx), (t) => req.onText(t), req.ctrl.signal),
  );
  live.req = req;
  req.promise
    .then(
      (out) => {
        if (live.req !== req) return;
        live.done.set(liveKey(text), out);
        live.ms.set(liveKey(text), Math.round(performance.now() - req.t0));
        live.shown[pos] = out;
        showLive();
      },
      () => {
        if (live.req === req) live.failed.add(liveKey(text));
      },
    )
    .finally(() => {
      if (live.req !== req) return;
      live.req = null;
      pumpLive();
    });
}

// Hands the live translations over to the sentences of the segment that just got
// its final transcript: reused when the text matches, otherwise shown as drafts.
export function takeLive(segs, index) {
  if (live.index !== index) {
    if (live.index < index) resetLive();
    return;
  }
  segs.forEach((seg, i) => {
    const key = liveKey(seg.text);
    if (live.done.has(key)) {
      seg.translation = live.done.get(key);
      seg.trStatus = 'done';
      seg.trMs = live.ms.get(key); // the delay of the live request that produced it
      return;
    }
    if (live.req && liveKey(live.req.text) === key) {
      adopted.set(seg.id, live.req);
      live.req = null;
    }
    if (live.shown[i]) drafts.set(seg.id, live.shown[i]);
  });
  resetLive();
}

// ---------------------------------------------------------------- local model

// reload: unload the model first, so that it moves between the GPU and the CPU.
export async function warmupLocal(reload = false) {
  setPill($('stTr'), '翻译：正在加载本地模型…', 'warn');
  try {
    const res = await fetch('/api/translate/warmup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: settings.tr.local, reload }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    if (!settings.tr.useLocal) return; // Switched away while loading.
    setPill($('stTr'), `翻译：本地模型已就绪（${settings.tr.local.model}）`, 'ok');
    clearTrMsg();
  } catch (e) {
    if (!settings.tr.useLocal) return;
    setPill($('stTr'), '翻译：本地模型不可用', 'err');
    showTrMsg(e.message);
  }
}
