// The transcript: a line per sentence with its translation, and the live line of the segment
// still being spoken (its words, and the draft translation); also the display settings.
import { $ } from './dom.js';
import { settings, state } from './state.js';
import { mathParts } from './lib/text.js';
import { fmtLag, fmtTime } from './lib/util.js';

export const transcriptEl = $('transcript');
let stickToBottom = true;

// seg.id -> live draft shown until the real translation replaces it (see takeLive).
export const drafts = new Map();

// What a line's 翻译 / 重译 button does: set by setupTranscript.
let retranslate = () => {};

export function setupTranscript({ onRetry }) {
  retranslate = onRetry;

  transcriptEl.addEventListener('scroll', () => {
    const nearBottom = transcriptEl.scrollHeight - transcriptEl.scrollTop - transcriptEl.clientHeight < 80;
    stickToBottom = nearBottom;
    $('jumpBtn').classList.toggle('hidden', nearBottom);
  });

  $('jumpBtn').onclick = () => {
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  };
}

// The live line of the segment being spoken: words pop in as Whisper hears them.
export function partialEl(index) {
  return transcriptEl.querySelector(`.seg.live[data-pidx="${index}"]`);
}

export function renderPartial(index, text) {
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

// Math (see MATH_RE) is rendered with KaTeX; everything else stays plain text (never parsed as HTML).
function setRichText(el, text) {
  text = text || '';
  if (!window.katex || !/\\[([]|\$/.test(text)) {
    el.textContent = text;
    return;
  }
  el.textContent = '';
  for (const part of mathParts(text)) {
    if (typeof part === 'string') {
      el.append(part);
      continue;
    }
    const span = document.createElement('span');
    try {
      katex.render(part.tex, span, { displayMode: part.display, throwOnError: false, output: 'html' });
    } catch {
      span.textContent = part.src;
    }
    el.append(span);
  }
}

export function renderLiveTrans(index, text) {
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

export function renderSeg(seg, replace = null, after = null) {
  if (seg.sessionId !== state.session?.id) return;
  $('emptyHint').classList.add('hidden');
  let el = transcriptEl.querySelector(`[data-id="${seg.id}"]`);
  if (!el) {
    el = document.createElement('div');
    el.className = 'seg';
    el.dataset.id = seg.id;
    el.innerHTML = `<div class="meta"><span class="time"></span><span class="lang"></span><span class="lag"></span></div>
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
  // Delays: audio sent -> text received, and text sent for translation -> translation received.
  const lag = el.querySelector('.lag');
  const lags = [];
  if (seg.asrLag != null) lags.push(`识别 ${fmtLag(seg.asrLag)}`);
  if (seg.trMs != null) lags.push(`翻译 ${fmtLag(seg.trMs)}`);
  lag.textContent = lags.length ? `延迟 ${lags.join(' · ')}` : '';
  lag.title = lags.length ? '识别：音频发送出去到收到文字的时间；翻译：文字发送出去到收到译文的时间' : '';

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
      retry.onclick = () => retranslate(seg);
      el.querySelector('.meta').appendChild(retry);
    }
    retry.textContent = seg.trStatus === 'done' ? '重译' : '翻译';
  } else if (retry) retry.remove();

  if (settings.display.autoScroll && stickToBottom) transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return el;
}

export function renderAll() {
  transcriptEl.querySelectorAll('.seg').forEach((e) => e.remove());
  $('emptyHint').classList.toggle('hidden', state.segments.length > 0);
  stickToBottom = true;
  for (const s of state.segments) renderSeg(s);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

export function applyDisplay() {
  const d = settings.display;
  document.documentElement.style.setProperty('--orig-size', `${d.origSize}px`);
  document.documentElement.style.setProperty('--trans-size', `${d.transSize}px`);
  document.body.classList.toggle('hide-orig', !d.showOrig);
}
