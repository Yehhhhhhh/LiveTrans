import * as db from './db.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- settings

const DEFAULTS = {
  source: 'mic',
  micDeviceId: '',
  // Browser noise suppression treats a distant, soft-spoken lecturer as noise, so it is off by default.
  audio: { micGain: 1, sysGain: 1, micProcessing: false },
  asr: { language: 'auto', vadThreshold: 0.5, minSilenceMs: 600, maxSegmentS: 8, glossaryPrompt: true, partial: true },
  tr: {
    enabled: true,
    live: true,
    useLocal: false, // overrides provider with the local Ollama model
    local: { baseUrl: 'http://127.0.0.1:11434', model: 'qwen3:4b-instruct-2507-q4_K_M' },
    provider: 'openai',
    contextSize: 3,
    openai: { baseUrl: 'https://api.openai.com/v1', apiKey: '', model: '' },
    claude: { apiKey: '', model: 'claude-opus-5-5', effort: 'low' },
  },
  display: { origSize: 16, transSize: 22, showOrig: true, autoScroll: true },
  currentSessionId: '',
};

function deepMerge(base, over) {
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(over || {})) {
    if (v && typeof v === 'object' && !Array.isArray(v) && typeof out[k] === 'object') {
      out[k] = deepMerge(out[k], v);
    } else if (k in out) {
      out[k] = v;
    }
  }
  return out;
}

function loadJson(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: settings just won't persist */
  }
}

const settings = deepMerge(DEFAULTS, loadJson('livetrans.settings'));
const saveSettings = () => saveJson('livetrans.settings', settings);

let glossary = loadJson('livetrans.glossary');
const saveGlossary = () => saveJson('livetrans.glossary', glossary);

// ---------------------------------------------------------------- state

const state = {
  ws: null,
  wsReady: false,
  running: false,
  audio: null, // { ctx, node, inputs: { sys?, mic? } }
  switching: false, // a source swap is waiting on a permission prompt
  streamStartWall: 0,
  session: null,
  segments: [], // segments of the current session, in order
  trQueue: [],
  trBusy: false,
  wakeLock: null,
};

// ---------------------------------------------------------------- helpers

function setPill(el, text, cls = '') {
  el.textContent = text;
  el.className = 'pill' + (cls ? ' ' + cls : '');
}

// Live speech probability next to the threshold, so the threshold can be set just below the speaker.
function showVad(msg) {
  const el = $('stVad');
  let text = msg.speaking ? '● 说话中' : '● 静音';
  if (typeof msg.prob === 'number') {
    text += ` 人声 ${msg.prob.toFixed(2)} / 阈值 ${Number(settings.asr.vadThreshold).toFixed(2)}`;
  }
  setPill(el, text, msg.speaking ? 'speaking' : '');
  el.title = '人声概率高于阈值才会开始识别。老师说话时人声值总低于阈值，就把「语音判定阈值」调低一些。';
}

// Time of the last final transcription, plus the backlog once the GPU falls behind the speaker
// (pending counts the segment being transcribed, so 1 is normal).
const asrStat = { ms: null, merged: 1, pending: 0 };
function showAsr() {
  const el = $('stAsr');
  let text = asrStat.ms == null ? '识别：-' : `识别：${asrStat.ms} ms`;
  if (asrStat.merged > 1) text += ` · 合并 ${asrStat.merged} 段追赶`;
  const behind = asrStat.pending >= 2;
  if (behind) text += ` · 待识别 ${asrStat.pending} 段`;
  const warn = behind || asrStat.merged > 1;
  setPill(el, text, warn ? 'warn' : asrStat.ms == null ? '' : 'ok');
  el.title = warn
    ? '识别速度跟不上说话速度，字幕会滞后。积压时自动把排队的几段合并成一次识别来追赶进度（自动语种时合并的几段共用一个语种）。经常出现可在「设置 → 识别」关闭逐词实时显示，或换用更小的模型。'
    : '';
}

// Messages pop up as a toast at the top centre. They fade after a while, except the
// translation-backend problem (trMsg), which stays until it is fixed or dismissed.
const TOAST_MS = 8000;
let toastTimer = 0;
function showMsg(text) {
  $('stMsg').textContent = text || '';
  const toast = $('toast');
  toast.classList.toggle('hidden', !text);
  // Just below the top bar, which wraps to more rows on narrow windows.
  if (text) toast.style.top = `${document.querySelector('.topbar').getBoundingClientRect().bottom + 8}px`;
  clearTimeout(toastTimer);
  if (text && text !== trMsg) toastTimer = setTimeout(() => showMsg(''), TOAST_MS);
}
$('toastClose').onclick = () => showMsg('');

// A translation-backend problem (e.g. Ollama not running) is shown until the
// backend works again or another backend is chosen; other messages are kept.
let trMsg = '';
function showTrMsg(text) {
  trMsg = text;
  showMsg(text);
}
function clearTrMsg() {
  if (trMsg && $('stMsg').textContent === trMsg) showMsg('');
  trMsg = '';
}

const pad = (n) => String(n).padStart(2, '0');
function fmtTime(ts) {
  const d = new Date(ts);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
function fmtDateTime(ts) {
  const d = new Date(ts);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- glossary

async function ensureGlossary() {
  if (Array.isArray(glossary)) return;
  const res = await fetch('glossary_default.json');
  glossary = await res.json();
  saveGlossary();
}

// Whisper prompt: newest (user-added) terms first, within a character budget
// so the 224-token prompt window isn't overflowed.
function buildAsrPrompt() {
  if (!settings.asr.glossaryPrompt || !glossary) return '';
  const seen = new Set();
  const terms = [];
  let len = 0;
  for (let i = glossary.length - 1; i >= 0 && len < 600; i--) {
    for (const t of [glossary[i].en, glossary[i].it]) {
      const k = (t || '').trim();
      if (!k || seen.has(k.toLowerCase())) continue;
      seen.add(k.toLowerCase());
      terms.push(k);
      len += k.length + 2;
    }
  }
  return terms.length ? `Lezione di telecomunicazioni / Telecommunications lecture. ${terms.join(', ')}.` : '';
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function relevantGlossary(texts) {
  const hay = texts.join(' \n ').toLowerCase();
  const out = [];
  for (const g of glossary || []) {
    const hit = [g.it, g.en].some((t) => {
      const k = (t || '').trim().toLowerCase();
      if (k.length < 2) return false;
      return new RegExp(`(^|[^\\p{L}\\p{N}])${escapeRe(k)}($|[^\\p{L}\\p{N}])`, 'u').test(hay);
    });
    if (hit) out.push({ it: g.it, en: g.en, zh: g.zh });
    if (out.length >= 40) break;
  }
  return out;
}

// ---------------------------------------------------------------- websocket

function asrConfig() {
  return {
    type: 'config',
    asr: {
      language: settings.asr.language,
      vad_threshold: Number(settings.asr.vadThreshold),
      min_silence_ms: Number(settings.asr.minSilenceMs),
      max_segment_s: Number(settings.asr.maxSegmentS),
      partial: Boolean(settings.asr.partial),
    },
    prompt: buildAsrPrompt(),
  };
}

function wsSend(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

const sendConfig = debounce(() => wsSend(asrConfig()), 300);

// Resuming after a dropped connection (backend restarted, network blip). Audio sent to the
// backend is kept until the backend reports it "committed": every segment in it answered, or
// silence. Audio captured while the connection is down is kept as well. On reconnect the stream
// resumes at the committed position and the kept audio is sent again, so what was said around
// the drop is still transcribed, with its own timestamps. At most RESEND_MAX_S is kept.
// Positions count samples (16 kHz) since 开始.
const RESEND_MAX_S = 120;
const resend = { chunks: [], pos: 0, committed: 0 };

function resetResend() {
  resend.chunks = [];
  resend.pos = 0;
  resend.committed = 0;
}

function trimResend() {
  const c = resend.chunks;
  while (c.length && c[0].pos + c[0].pcm.length <= resend.committed) c.shift();
  while (c.length && resend.pos - c[0].pos > RESEND_MAX_S * 16000) c.shift();
}

function sendAudio(pcm) {
  resend.chunks.push({ pos: resend.pos, pcm });
  resend.pos += pcm.length;
  trimResend();
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(pcm);
}

function onCommitted(pos) {
  if (pos <= resend.committed) return;
  resend.committed = pos;
  trimResend();
}

// Called on reconnect while capturing: continue the stream where the backend can pick it up.
function resumeStream() {
  const first = resend.chunks[0];
  const from = Math.max(resend.committed, first ? first.pos : resend.pos);
  const lost = (from - resend.committed) / 16000;
  wsSend({ type: 'resume', offset: from });
  resend.committed = from;
  let sent = 0;
  for (const ch of resend.chunks) {
    if (ch.pos + ch.pcm.length <= from) continue;
    const part = ch.pos >= from ? ch.pcm : ch.pcm.subarray(from - ch.pos);
    state.ws.send(part);
    sent += part.length;
  }
  if (lost >= 1) showMsg(`断线超过 ${RESEND_MAX_S / 60} 分钟，约 ${Math.round(lost)} 秒语音未能补回`);
  else if (sent >= 16000) showMsg(`已重新连接，补发断线前后约 ${Math.round(sent / 16000)} 秒语音`);
}

function connectWs() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.wsReady = true;
    setPill($('stWs'), '后端：已连接', 'ok');
    // A new connection starts with an empty queue (a dropped one's backlog is gone),
    // so neither the backlog nor the last merge note applies any more.
    asrStat.pending = 0;
    asrStat.merged = 1;
    showAsr();
    wsSend(asrConfig());
    if (state.running) {
      // Reconnected mid-capture: resend what the backend hasn't answered yet.
      resumeStream();
    }
  };

  ws.onclose = () => {
    state.wsReady = false;
    setPill($('stWs'), '后端：未连接，重连中…', 'err');
    setTimeout(connectWs, 2000);
  };

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (typeof msg.committed === 'number') onCommitted(msg.committed);
    if (msg.type === 'status') onStatus(msg);
    else if (msg.type === 'vad') showVad(msg);
    else if (msg.type === 'segment') onSegment(msg);
    else if (msg.type === 'partial') {
      renderPartial(msg.index, msg.text);
      liveTranslate(msg.index, msg.text);
    } else if (msg.type === 'partial_drop') {
      if (live.index === msg.index) resetLive();
      partialEl(msg.index)?.remove();
    }
    else if (msg.type === 'backlog') {
      asrStat.pending = msg.pending;
      if (!msg.pending) asrStat.merged = 1; // caught up: the merge note no longer applies
      showAsr();
    } else if (msg.type === 'error') showMsg(msg.message);
  };
}

function onStatus(msg) {
  const el = $('stModel');
  if (msg.model === 'ready') setPill(el, `模型：${msg.model_name} · ${msg.device}/${msg.compute_type}`, msg.device === 'cuda' ? 'ok' : 'warn');
  else if (msg.model === 'loading') setPill(el, '模型：加载中（首次需下载）…', 'warn');
  else if (msg.model === 'error') setPill(el, '模型：加载失败', 'err');
  if (msg.error) showMsg(msg.error);
}

// ---------------------------------------------------------------- segments

async function onSegment(msg) {
  const text = (msg.text || '').trim();
  // Partials of earlier segments can no longer be finalized.
  transcriptEl.querySelectorAll('.seg.live').forEach((el) => {
    if (Number(el.dataset.pidx) < msg.index) el.remove();
  });
  if (!text || !state.session) {
    if (live.index <= msg.index) resetLive();
    partialEl(msg.index)?.remove();
    return;
  }
  // One line per sentence; timestamps are spread over the segment by text length.
  const sentences = splitSentences(text);
  const total = sentences.reduce((n, s) => n + s.length, 0);
  let offset = 0;
  const segs = sentences.map((s) => {
    const seg = {
      id: crypto.randomUUID(),
      sessionId: state.session.id,
      ts: state.streamStartWall + (msg.start + (msg.duration * offset) / total) * 1000,
      lang: msg.lang,
      text: s,
      translation: '',
      trStatus: settings.tr.enabled ? 'pending' : 'off',
      error: '',
    };
    offset += s.length;
    return seg;
  });
  asrStat.ms = msg.asr_ms;
  asrStat.merged = msg.merged || 1;
  showAsr();
  if (settings.tr.enabled) takeLive(segs, msg.index);
  if (segs.some((s) => s.trStatus === 'done')) {
    setPill($('stTr'), `翻译：说话时已完成 · 队列 ${state.trQueue.length}`, 'ok');
  }
  let prev = partialEl(msg.index);
  segs.forEach((seg, i) => {
    state.segments.push(seg);
    prev = i === 0 ? renderSeg(seg, prev) : renderSeg(seg, null, prev);
  });
  for (const seg of segs) {
    await db.putSegment(seg);
    if (seg.trStatus === 'pending') enqueueTranslation(seg);
  }
}

const wordCount = (s) => s.split(/\s+/).filter(Boolean).length;

// Whisper often puts several sentences into one VAD segment; each gets its own
// line and translation. Pieces under 3 words stay with a neighbour.
function splitSentences(text) {
  const out = [];
  for (const p of text.split(/(?<=[.?!…])\s+(?=[\p{Lu}¿¡"'])/u)) {
    if (out.length && wordCount(out.at(-1)) < 3) out[out.length - 1] += ` ${p}`;
    else out.push(p);
  }
  if (out.length > 1 && wordCount(out.at(-1)) < 3) out[out.length - 2] += ` ${out.pop()}`;
  return out;
}

function enqueueTranslation(seg) {
  seg.trStatus = 'pending';
  seg.error = '';
  renderSeg(seg);
  state.trQueue.push(seg);
  pumpTranslations();
}

async function pumpTranslations() {
  if (state.trBusy) return;
  state.trBusy = true;
  while (state.trQueue.length) {
    const seg = state.trQueue.shift();
    await translateSeg(seg);
  }
  state.trBusy = false;
  pumpLive();
}

function translationRequest(text, context) {
  const p = settings.tr.useLocal ? 'ollama' : settings.tr.provider;
  return {
    provider: p,
    config: { claude: settings.tr.claude, openai: settings.tr.openai, ollama: settings.tr.local }[p],
    text,
    context,
    glossary: relevantGlossary([text, ...context.map((c) => c.src)]),
  };
}

// Streams the translation: onText receives the full text so far on every update.
async function postTranslate(body, onText = () => {}, signal = undefined) {
  const res = await fetch('/api/translate/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (value) buf += value;
    const lines = buf.split('\n');
    buf = done ? '' : lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const msg = JSON.parse(line);
      if (msg.error) throw new Error(msg.error);
      if (msg.done) return msg.translation;
      onText(msg.text);
    }
    if (done) throw new Error('翻译流意外中断');
  }
}

// Context: the previous translated segments of the same session (all of the
// current session's when seg is null, i.e. for the segment still being spoken).
async function contextFor(seg) {
  const list = !seg || seg.sessionId === state.session?.id ? state.segments : await db.getSegments(seg.sessionId);
  const idx = seg ? list.findIndex((s) => s.id === seg.id) : -1;
  const n = Number(settings.tr.contextSize) || 0;
  return list
    .slice(0, idx < 0 ? list.length : idx)
    .filter((s) => s.trStatus === 'done')
    .slice(-n)
    .map((s) => ({ src: s.text, tgt: s.translation }));
}

async function translateSeg(seg) {
  // A live request already translating exactly this text is taken over rather than restarted.
  const req = adopted.get(seg.id);
  adopted.delete(seg.id);
  const t0 = performance.now();
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
    setPill($('stTr'), `翻译：首字 ${Math.round(first || total)} ms · 共 ${total} ms · 队列 ${state.trQueue.length}`, 'ok');
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
  failed: new Set(), // sentence texts not to retry; the final pass still follows
  shown: [], // per sentence position: the finished draft on screen  req: null,
});
let live = newLive(-1);
// Partials drop Whisper's closing period, so it's ignored when matching sentences.
const liveKey = (s) => s.replace(/[.…]+$/, '').trim();
const drafts = new Map(); // seg.id -> live draft shown until the real translation replaces it
const adopted = new Map(); // seg.id -> live request whose text equals the final sentence

function resetLive(index = -1) {
  live.req?.ctrl.abort();
  live = newLive(index);
}

function liveTranslate(index, text) {
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
  if (live.req || state.trBusy) return;
  const pos = live.sentences.findIndex((s) => s && !live.done.has(liveKey(s)) && !live.failed.has(liveKey(s)));
  if (pos < 0) return;
  const text = live.sentences[pos];
  const req = {
    text,
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
function takeLive(segs, index) {
  if (live.index !== index) {
    if (live.index < index) resetLive();
    return;
  }
  segs.forEach((seg, i) => {
    const key = liveKey(seg.text);
    if (live.done.has(key)) {
      seg.translation = live.done.get(key);
      seg.trStatus = 'done';
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

// ---------------------------------------------------------------- rendering

const transcriptEl = $('transcript');
let stickToBottom = true;

transcriptEl.addEventListener('scroll', () => {
  const nearBottom = transcriptEl.scrollHeight - transcriptEl.scrollTop - transcriptEl.clientHeight < 80;
  stickToBottom = nearBottom;
  $('jumpBtn').classList.toggle('hidden', nearBottom);
});

$('jumpBtn').onclick = () => {
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
};

// The live line of the segment being spoken: words pop in as Whisper hears them.
function partialEl(index) {
  return transcriptEl.querySelector(`.seg.live[data-pidx="${index}"]`);
}

function renderPartial(index, text) {
  if (!state.running) return;
  $('emptyHint').classList.add('hidden');
  let el = partialEl(index);
  if (!el) {
    el = document.createElement('div');
    el.className = 'seg live';
    el.dataset.pidx = index;
    el.innerHTML = `<div class="meta"><span class="time">识别中…</span></div><div class="words"></div>`;
    transcriptEl.appendChild(el);
  }
  // Keep the words that didn't change; animate only the new tail.
  const box = el.querySelector('.words');
  const words = text.split(/\s+/).filter(Boolean);
  const spans = box.children;
  let keep = 0;
  while (keep < spans.length && keep < words.length && spans[keep].textContent.trimEnd() === words[keep]) keep++;
  while (spans.length > keep) spans[spans.length - 1].remove();
  words.slice(keep).forEach((w, i) => {
    const span = document.createElement('span');
    span.className = 'w';
    span.style.animationDelay = `${Math.min(i * 70, 500)}ms`;
    span.textContent = `${w} `;
    box.appendChild(span);
  });
  if (settings.display.autoScroll && stickToBottom) transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// Translations may contain LaTeX: \(...\) / $...$ inline, \[...\] / $$...$$ display.
// Math is rendered with KaTeX; everything else stays plain text (never parsed as HTML).
// An unclosed delimiter (e.g. mid-stream) is left as-is until its closing half arrives.
const MATH_RE = /\\\((.+?)\\\)|\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$/g;

function setRichText(el, text) {
  text = text || '';
  if (!window.katex || !/\\[([]|\$/.test(text)) {
    el.textContent = text;
    return;
  }
  el.textContent = '';
  let last = 0;
  for (const m of text.matchAll(MATH_RE)) {
    if (m.index > last) el.append(text.slice(last, m.index));
    const display = m[2] !== undefined || m[3] !== undefined;
    const tex = m[1] ?? m[2] ?? m[3] ?? m[4];
    const span = document.createElement('span');
    try {
      katex.render(tex, span, { displayMode: display, throwOnError: false, output: 'html' });
    } catch {
      span.textContent = m[0];
    }
    el.append(span);
    last = m.index + m[0].length;
  }
  if (last < text.length) el.append(text.slice(last));
}

function renderLiveTrans(index, text) {
  const el = partialEl(index);
  if (!el) return;
  let tr = el.querySelector('.trans');
  if (!tr) {
    tr = document.createElement('div');
    tr.className = 'trans draft';
    el.appendChild(tr);
  }
  setRichText(tr, text);
  if (settings.display.autoScroll && stickToBottom) transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function renderSeg(seg, replace = null, after = null) {
  if (seg.sessionId !== state.session?.id) return;
  $('emptyHint').classList.add('hidden');
  let el = transcriptEl.querySelector(`[data-id="${seg.id}"]`);
  if (!el) {
    el = document.createElement('div');
    el.className = 'seg';
    el.dataset.id = seg.id;
    el.innerHTML = `<div class="meta"><span class="time"></span><span class="lang"></span></div>
      <div class="orig"></div><div class="trans"></div>`;
    if (replace) replace.replaceWith(el);
    else if (after) after.after(el);
    else transcriptEl.appendChild(el);
  }
  el.querySelector('.time').textContent = fmtTime(seg.ts);
  const lang = el.querySelector('.lang');
  lang.textContent = (seg.lang || '?').toUpperCase();
  lang.className = `lang ${seg.lang || ''}`;
  el.querySelector('.orig').textContent = seg.text;

  const tr = el.querySelector('.trans');
  tr.className = 'trans';
  if (seg.trStatus === 'done') setRichText(tr, seg.translation);
  else if (seg.trStatus === 'streaming') {
    setRichText(tr, seg.translation);
    tr.classList.add('streaming');
  } else if (seg.trStatus === 'pending' && drafts.has(seg.id)) {
    setRichText(tr, drafts.get(seg.id));
    tr.classList.add('draft');
  } else if (seg.trStatus === 'pending') {
    tr.textContent = '翻译中…';
    tr.classList.add('pending');
  } else if (seg.trStatus === 'error') {
    tr.textContent = `翻译失败：${seg.error}`;
    tr.classList.add('error');
  } else tr.textContent = '';

  let retry = el.querySelector('.retry');
  if (seg.trStatus === 'error' || seg.trStatus === 'off' || seg.trStatus === 'done') {
    if (!retry) {
      retry = document.createElement('button');
      retry.className = 'retry';
      retry.onclick = () => enqueueTranslation(seg);
      el.querySelector('.meta').appendChild(retry);
    }
    retry.textContent = seg.trStatus === 'done' ? '重译' : '翻译';
  } else if (retry) retry.remove();

  if (settings.display.autoScroll && stickToBottom) transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return el;
}

function renderAll() {
  transcriptEl.querySelectorAll('.seg').forEach((e) => e.remove());
  $('emptyHint').classList.toggle('hidden', state.segments.length > 0);
  stickToBottom = true;
  for (const s of state.segments) renderSeg(s);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function applyDisplay() {
  const d = settings.display;
  document.documentElement.style.setProperty('--orig-size', `${d.origSize}px`);
  document.documentElement.style.setProperty('--trans-size', `${d.transSize}px`);
  document.body.classList.toggle('hide-orig', !d.showOrig);
}

// ---------------------------------------------------------------- sessions

async function refreshSessionList() {
  const sessions = await db.listSessions();
  const sel = $('sessionSel');
  sel.innerHTML = '';
  for (const s of sessions) {
    const o = document.createElement('option');
    o.value = s.id;
    o.textContent = s.title;
    sel.appendChild(o);
  }
  if (state.session) sel.value = state.session.id;
  return sessions;
}

async function createSession(title) {
  const now = Date.now();
  const s = { id: crypto.randomUUID(), title: title || `课程 ${fmtDateTime(now)}`, createdAt: now };
  await db.putSession(s);
  await openSession(s);
}

async function openSession(s) {
  state.session = s;
  settings.currentSessionId = s.id;
  saveSettings();
  state.segments = await db.getSegments(s.id);
  await refreshSessionList();
  renderAll();
}

async function initSessions() {
  const sessions = await db.listSessions();
  const cur = sessions.find((s) => s.id === settings.currentSessionId) || sessions[0];
  if (cur) await openSession(cur);
  else await createSession();
}

$('sessionSel').onchange = async () => {
  const sessions = await db.listSessions();
  const s = sessions.find((x) => x.id === $('sessionSel').value);
  if (s) await openSession(s);
};

$('newSessionBtn').onclick = async () => {
  const title = prompt('课程名称', `课程 ${fmtDateTime(Date.now())}`);
  if (title !== null) await createSession(title.trim() || undefined);
};

$('renameSessionBtn').onclick = async () => {
  if (!state.session) return;
  const title = prompt('重命名', state.session.title);
  if (!title || !title.trim()) return;
  state.session.title = title.trim();
  await db.putSession(state.session);
  await refreshSessionList();
};

$('deleteSessionBtn').onclick = async () => {
  if (!state.session) return;
  if (!confirm(`删除「${state.session.title}」及其全部 ${state.segments.length} 条记录？此操作不可恢复。`)) return;
  await db.deleteSession(state.session.id);
  state.session = null;
  await initSessions();
};

$('exportBtn').onclick = () => {
  const s = state.session;
  if (!s) return;
  const lines = [`# ${s.title}`, '', `- 创建时间：${fmtDateTime(s.createdAt)}`, `- 段数：${state.segments.length}`, '', '---', ''];
  for (const seg of state.segments) {
    lines.push(`**[${fmtTime(seg.ts)}] ${(seg.lang || '').toUpperCase()}**  `);
    lines.push(seg.text);
    lines.push('');
    if (seg.translation) {
      lines.push(`> ${seg.translation}`);
      lines.push('');
    }
  }
  const safe = s.title.replace(/[\\/:*?"<>|]/g, '_');
  download(`${safe}.md`, lines.join('\n'), 'text/markdown;charset=utf-8');
};

// ---------------------------------------------------------------- audio

async function listMics() {
  const sel = $('micSel');
  let devices = [];
  try {
    devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  } catch {
    /* no permission yet */
  }
  sel.innerHTML = '';
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId;
    o.textContent = d.label || `麦克风 ${i + 1}`;
    sel.appendChild(o);
  });
  if (settings.micDeviceId && devices.some((d) => d.deviceId === settings.micDeviceId)) sel.value = settings.micDeviceId;
}

async function getSystemStream() {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    systemAudio: 'include',
    selfBrowserSurface: 'exclude',
    // Chrome then offers "share this tab instead" in its share bar, switching tabs without a prompt.
    surfaceSwitching: 'include',
  });
  if (stream.getAudioTracks().length === 0) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error('没有获取到音频：请在共享弹窗中勾选「共享系统音频」或「共享标签页音频」');
  }
  // Only the audio is used; keep the video track alive but idle.
  stream.getVideoTracks().forEach((t) => (t.enabled = false));
  return stream;
}

async function getMicStream() {
  const proc = settings.audio.micProcessing;
  return navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: settings.micDeviceId ? { exact: settings.micDeviceId } : undefined,
      channelCount: 1,
      echoCancellation: proc,
      noiseSuppression: proc,
      autoGainControl: proc,
    },
  });
}

const wantedKinds = (src) => (src === 'both' ? ['sys', 'mic'] : [src === 'system' ? 'sys' : 'mic']);

// 'sys' always comes first in wantedKinds, so the sharing prompt opens while the click's user activation is fresh.
const acquire = (kind) => (kind === 'sys' ? getSystemStream() : getMicStream());

function stopStream(stream) {
  stream.getTracks().forEach((t) => t.stop());
}

function acquireError(e) {
  return e.name === 'NotAllowedError' ? '已取消或未授权音频采集' : e.message;
}

// Plugs a stream into the running graph as the current input of its kind,
// connecting the new one before releasing the old, so audio never gaps.
function attachInput(kind, stream) {
  const a = state.audio;
  const source = a.ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
  const gain = a.ctx.createGain();
  gain.gain.value = kind === 'mic' ? settings.audio.micGain : settings.audio.sysGain;
  source.connect(gain).connect(a.node);
  const input = { kind, stream, source, gain };
  input.onEnded = () => onInputEnded(input);
  stream.getTracks().forEach((t) => t.addEventListener('ended', input.onEnded));
  const old = a.inputs[kind];
  a.inputs[kind] = input;
  if (old) releaseInput(old);
  if (kind === 'sys') describeShare(input);
  updateSourceUi();
}

function releaseInput(input) {
  input.stream.getTracks().forEach((t) => t.removeEventListener('ended', input.onEnded));
  input.gain.disconnect();
  input.source.disconnect();
  stopStream(input.stream);
}

function detachInput(kind) {
  const a = state.audio;
  const input = a?.inputs[kind];
  if (!input) return;
  delete a.inputs[kind];
  releaseInput(input);
  updateSourceUi();
}

// The user ended a share (Chrome's "Stop sharing") or a mic went away.
function onInputEnded(input) {
  const a = state.audio;
  if (!state.running || a?.inputs[input.kind] !== input) return;
  const msg = input.kind === 'sys' ? '屏幕/窗口共享已结束' : '麦克风已断开';
  detachInput(input.kind);
  if (Object.keys(a.inputs).length === 0) {
    stop();
    showMsg(msg);
  } else {
    showMsg(input.kind === 'sys' ? `${msg}，麦克风继续识别；点「共享窗口」可重新共享` : `${msg}，系统音频继续识别`);
  }
}

// Swaps inputs while running: the backend keeps one continuous stream, so VAD and transcript carry on.
async function swapInputs(kinds, replace = false) {
  if (!state.running || state.switching) return false;
  state.switching = true;
  const a = state.audio;
  const fresh = [];
  try {
    for (const kind of kinds) {
      if (replace || !a.inputs[kind]) fresh.push({ kind, stream: await acquire(kind) });
    }
  } catch (e) {
    fresh.forEach((f) => stopStream(f.stream));
    if (state.running) showMsg(acquireError(e) + '，已保留原音源');
    return false;
  } finally {
    state.switching = false;
  }
  if (!state.running || state.audio !== a) {
    fresh.forEach((f) => stopStream(f.stream)); // stopped while the prompt was open
    return false;
  }
  showMsg('');
  fresh.forEach((f) => attachInput(f.kind, f.stream));
  return true;
}

async function onSourceChange(prev) {
  updateSourceUi();
  if (!state.running) return;
  const kinds = wantedKinds(settings.source);
  if (!(await swapInputs(kinds))) {
    // Prompt cancelled: keep capturing what we had, and show that in the selector.
    if (state.running && !kinds.every((k) => state.audio.inputs[k])) {
      settings.source = prev;
      $('sourceSel').value = prev;
      $('micSel').classList.toggle('hidden', prev === 'system');
      saveSettings();
      updateSourceUi();
    }
    return;
  }
  for (const kind of Object.keys(state.audio.inputs)) {
    if (!kinds.includes(kind)) detachInput(kind);
  }
  if (kinds.includes('mic')) listMics();
}

function updateSourceUi() {
  const sys = state.audio?.inputs.sys;
  const btn = $('switchShareBtn');
  btn.classList.toggle('hidden', !state.running || settings.source === 'mic');
  btn.textContent = sys ? '换窗口' : '共享窗口';
  const label = $('shareLabel');
  label.classList.toggle('hidden', !sys);
  if (!sys) {
    label.textContent = '';
    label.title = '';
  }
}

// Shows what is being shared. The browser only reveals the surface type; for a window,
// Chrome's track label carries the native window handle, which the backend resolves to a name.
async function describeShare(input) {
  const track = input.stream.getVideoTracks()[0];
  const surface = track?.getSettings().displaySurface;
  const raw = track?.label || '';
  console.debug('[LiveTrans] shared surface:', surface, JSON.stringify(raw));
  const show = (text, title = text, icon = '') => {
    if (state.audio?.inputs.sys !== input) return; // replaced meanwhile
    const label = $('shareLabel');
    label.textContent = text;
    label.title = title;
    if (icon) {
      const img = document.createElement('img');
      img.className = 'share-icon';
      img.alt = '';
      img.src = icon;
      label.prepend(img);
    }
  };
  if (surface === 'monitor' || /^screen:/.test(raw)) {
    show('🖥 整个屏幕');
  } else if (surface === 'browser') {
    show('🌐 浏览器标签页', '浏览器标签页（浏览器不提供标签页标题）');
  } else if (surface === 'window' || /^window:/.test(raw)) {
    show('🪟 窗口');
    const m = /^window:(\d+):/.exec(raw);
    if (!m) return;
    try {
      const r = await fetch(`/api/window?hwnd=${m[1]}`);
      if (!r.ok) return;
      const w = await r.json();
      const app = w.app || '';
      const title = w.title && w.title !== app ? w.title : '';
      const text = [app, title].filter(Boolean).join(' · ');
      // The app's own icon replaces the generic window emoji when the backend could read it.
      const icon = /^data:image\/png;base64,/.test(w.icon || '') ? w.icon : '';
      if (text) show(icon ? text : `🪟 ${text}`, `窗口：${text}${w.exe ? `（${w.exe}）` : ''}`, icon);
    } catch {
      /* keep the generic label */
    }
  } else {
    show('共享中');
  }
}

async function start() {
  if (!state.wsReady) {
    showMsg('后端未连接，请先运行 start.bat');
    return;
  }
  showMsg('');
  const src = settings.source;
  const acquired = [];
  try {
    for (const kind of wantedKinds(src)) acquired.push({ kind, stream: await acquire(kind) });
  } catch (e) {
    acquired.forEach((a) => stopStream(a.stream));
    showMsg(acquireError(e));
    return;
  }

  const ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.audioWorklet.addModule('audio-worklet.js');
  const node = new AudioWorkletNode(ctx, 'pcm-capture');
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink).connect(ctx.destination);

  const meter = $('meterBar');
  node.port.onmessage = (e) => {
    sendAudio(e.data.pcm);
    meter.style.width = `${Math.min(100, e.data.rms * 400)}%`;
  };

  state.audio = { ctx, node, inputs: {} };
  state.running = true;
  acquired.forEach((a) => attachInput(a.kind, a.stream));
  resetLive();
  transcriptEl.querySelectorAll('.seg.live').forEach((el) => el.remove());
  wsSend(asrConfig());
  resetResend();
  wsSend({ type: 'start' });
  state.streamStartWall = Date.now();
  $('startBtn').textContent = '停止';
  $('startBtn').classList.add('running');
  if (src !== 'system') listMics();
  requestWakeLock();
}

function stop() {
  wsSend({ type: 'stop' });
  const a = state.audio;
  if (a) {
    Object.values(a.inputs).forEach(releaseInput);
    a.ctx.close();
  }
  state.audio = null;
  state.running = false;
  updateSourceUi();
  $('startBtn').textContent = '开始';
  $('startBtn').classList.remove('running');
  $('meterBar').style.width = '0';
  setPill($('stVad'), '● 静音');
  state.wakeLock?.release().catch(() => {});
  state.wakeLock = null;
}

async function requestWakeLock() {
  try {
    state.wakeLock = await navigator.wakeLock?.request('screen');
  } catch {
    /* not critical */
  }
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && state.running) requestWakeLock();
});

$('startBtn').onclick = () => (state.running ? stop() : start());

$('sourceSel').value = settings.source;
$('sourceSel').onchange = () => {
  const prev = settings.source;
  settings.source = $('sourceSel').value;
  $('micSel').classList.toggle('hidden', settings.source === 'system');
  saveSettings();
  onSourceChange(prev);
};
$('micSel').classList.toggle('hidden', settings.source === 'system');
$('micSel').onchange = () => {
  settings.micDeviceId = $('micSel').value;
  saveSettings();
  if (state.audio?.inputs.mic) swapInputs(['mic'], true);
};
// Pick another window/tab/screen without stopping recognition.
$('switchShareBtn').onclick = () => swapInputs(['sys'], true);
navigator.mediaDevices.addEventListener?.('devicechange', listMics);

// ---------------------------------------------------------------- settings UI

$('settingsBtn').onclick = () => $('settings').classList.toggle('hidden');
$('closeSettings').onclick = () => $('settings').classList.add('hidden');

document.querySelectorAll('.tab').forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    document.querySelectorAll('.pane').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== tab.dataset.tab));
  };
});

function getPath(path) {
  return path.split('.').reduce((o, k) => o[k], settings);
}
function setPath(path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  keys.reduce((o, k) => o[k], settings)[last] = value;
}

// Two-way bind a control to a settings path.
function bind(id, path, onChange) {
  const el = $(id);
  const isCheck = el.type === 'checkbox';
  const isNum = el.type === 'number' || el.type === 'range';
  const show = () => (isCheck ? (el.checked = getPath(path)) : (el.value = getPath(path)));
  show();
  el.addEventListener(isCheck || el.tagName === 'SELECT' ? 'change' : 'input', () => {
    let v = isCheck ? el.checked : el.value;
    if (isNum) {
      v = Number(v);
      if (Number.isNaN(v)) return;
    }
    setPath(path, v);
    saveSettings();
    onChange?.(v);
  });
}

function updateProviderFields() {
  const claude = settings.tr.provider === 'claude';
  $('openaiFields').classList.toggle('hidden', claude);
  $('claudeFields').classList.toggle('hidden', !claude);
}

bind('trEnabled', 'tr.enabled');
bind('trProvider', 'tr.provider', updateProviderFields);
bind('oaBase', 'tr.openai.baseUrl');
bind('oaKey', 'tr.openai.apiKey');
bind('oaModel', 'tr.openai.model');
bind('clKey', 'tr.claude.apiKey');
bind('clModel', 'tr.claude.model');
bind('clEffort', 'tr.claude.effort');
bind('trContext', 'tr.contextSize');
bind('locBase', 'tr.local.baseUrl');
bind('locModel', 'tr.local.model');
updateProviderFields();

// Toggles shown both in the top bar and in settings.
function bindToggle(ids, path, onChange) {
  const sync = (v) => {
    ids.forEach((id) => ($(id).checked = v));
    onChange?.(v);
  };
  ids.forEach((id) => bind(id, path, sync));
}

async function warmupLocal() {
  setPill($('stTr'), '翻译：正在加载本地模型…', 'warn');
  try {
    const res = await fetch('/api/translate/warmup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: settings.tr.local }),
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

bindToggle(['trLive', 'tbLive'], 'tr.live');
bindToggle(['trLocal', 'tbLocal'], 'tr.useLocal', (on) => {
  if (on) return warmupLocal();
  clearTrMsg();
  setPill($('stTr'), '翻译：-');
});
if (settings.tr.useLocal) warmupLocal();

bind('asrLang', 'asr.language', sendConfig);
bind('vadThr', 'asr.vadThreshold', sendConfig);
bind('vadSil', 'asr.minSilenceMs', sendConfig);
bind('vadMax', 'asr.maxSegmentS', sendConfig);
bind('asrGlossary', 'asr.glossaryPrompt', sendConfig);
bind('asrPartial', 'asr.partial', sendConfig);

const showGain = () => {
  $('micGainV').textContent = `${settings.audio.micGain}×`;
  $('sysGainV').textContent = `${settings.audio.sysGain}×`;
};
bind('micGain', 'audio.micGain', showGain);
bind('sysGain', 'audio.sysGain', showGain);
bind('micProc', 'audio.micProcessing');
showGain();

bind('showOrig', 'display.showOrig', applyDisplay);
bind('autoScroll', 'display.autoScroll');
bind('origSize', 'display.origSize', applyDisplay);
bind('transSize', 'display.transSize', applyDisplay);

document.querySelectorAll('[data-font]').forEach((btn) => {
  btn.onclick = () => {
    const key = btn.dataset.font === 'orig' ? 'origSize' : 'transSize';
    settings.display[key] = Math.max(10, Math.min(64, settings.display[key] + Number(btn.dataset.delta)));
    $(key).value = settings.display[key];
    saveSettings();
    applyDisplay();
  };
});

$('trTest').onclick = async () => {
  const out = $('trTestOut');
  out.textContent = '测试中…';
  const t0 = performance.now();
  let first = 0;
  try {
    const r = await postTranslate(
      translationRequest('Allora, nel sistema OFDM ogni subcarrier is modulated independently, quindi il cyclic prefix elimina la ISI.', []),
      (text) => {
        if (!first) first = performance.now() - t0;
        out.textContent = `… ${text}`;
      },
    );
    const total = Math.round(performance.now() - t0);
    out.textContent = `✓ 首字 ${Math.round(first || total)} ms · 共 ${total} ms：${r}`;
  } catch (e) {
    out.textContent = `✗ ${e.message}`;
  }
};

// ---------------------------------------------------------------- glossary UI

function renderGlossary() {
  const q = $('glSearch').value.trim().toLowerCase();
  const body = $('glBody');
  body.innerHTML = '';
  let shown = 0;
  // Newest first so freshly added rows appear at the top.
  for (let i = glossary.length - 1; i >= 0; i--) {
    const g = glossary[i];
    if (q && ![g.it, g.en, g.zh].some((t) => (t || '').toLowerCase().includes(q))) continue;
    shown++;
    const tr = document.createElement('tr');
    for (const k of ['it', 'en', 'zh']) {
      const td = document.createElement('td');
      const inp = document.createElement('input');
      inp.value = g[k] || '';
      inp.oninput = () => {
        g[k] = inp.value;
        saveGlossaryAndSync();
      };
      td.appendChild(inp);
      tr.appendChild(td);
    }
    const td = document.createElement('td');
    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '✕';
    del.title = '删除';
    del.onclick = () => {
      glossary.splice(glossary.indexOf(g), 1);
      saveGlossaryAndSync();
      renderGlossary();
    };
    td.appendChild(del);
    tr.appendChild(td);
    body.appendChild(tr);
  }
  $('glCount').textContent = `共 ${glossary.length} 条${q ? `，匹配 ${shown} 条` : ''}`;
}

const saveGlossaryAndSync = debounce(() => {
  saveGlossary();
  sendConfig();
}, 400);

$('glSearch').oninput = renderGlossary;
$('glAdd').onclick = () => {
  glossary.push({ it: '', en: '', zh: '' });
  $('glSearch').value = '';
  saveGlossaryAndSync();
  renderGlossary();
  $('glBody').querySelector('input')?.focus();
};
$('glExport').onclick = () => download('glossary.json', JSON.stringify(glossary, null, 1), 'application/json');
$('glReset').onclick = async () => {
  if (!confirm('恢复默认术语表？你添加或修改的条目会丢失。')) return;
  glossary = null;
  await ensureGlossary();
  sendConfig();
  renderGlossary();
};
$('glImport').onclick = () => $('glFile').click();
$('glFile').onchange = async () => {
  const file = $('glFile').files[0];
  $('glFile').value = '';
  if (!file) return;
  const text = await file.text();
  let items = [];
  try {
    if (file.name.toLowerCase().endsWith('.json')) {
      items = JSON.parse(text).map((x) => ({ it: x.it || '', en: x.en || '', zh: x.zh || '' }));
    } else {
      items = text
        .split(/\r?\n/)
        .map((l) => l.split(/[,\t;]/).map((x) => x.trim()))
        .filter((c) => c.some(Boolean))
        .map(([it = '', en = '', zh = '']) => ({ it, en, zh }));
    }
  } catch (e) {
    alert(`导入失败：${e.message}`);
    return;
  }
  const key = (g) => `${(g.it || '').toLowerCase()}|${(g.en || '').toLowerCase()}`;
  const existing = new Map(glossary.map((g) => [key(g), g]));
  let added = 0;
  for (const it of items) {
    const old = existing.get(key(it));
    if (old) Object.assign(old, it);
    else {
      glossary.push(it);
      added++;
    }
  }
  saveGlossaryAndSync();
  renderGlossary();
  alert(`导入 ${items.length} 条，其中新增 ${added} 条`);
};

// ---------------------------------------------------------------- boot

window.addEventListener('beforeunload', (e) => {
  if (state.running) e.preventDefault();
});

(async function boot() {
  applyDisplay();
  await ensureGlossary();
  renderGlossary();
  await initSessions();
  await listMics();
  connectWs();
})();
