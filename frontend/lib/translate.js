// Translation requests to the backend (/api/translate/stream), and the context sent with them.
import { relevantGlossary } from './glossary.js';

// The context for translating `seg`: the last `contextSize` segments before it in `list` (all of
// `list` when seg is null or not in it), skipping ones whose translation failed or is off.
// Translated ones come with their translation; ones still being translated (translations run in
// parallel) with their source text only. (A contextSize of 0 still takes them all, as before
// this was moved here: .slice(-0) is .slice(0).)
export function pickContext(list, seg, contextSize) {
  const idx = seg ? list.findIndex((s) => s.id === seg.id) : -1;
  const n = Number(contextSize) || 0;
  return list
    .slice(0, idx < 0 ? list.length : idx)
    .filter((s) => s.trStatus === 'done' || s.trStatus === 'pending' || s.trStatus === 'streaming')
    .slice(-n)
    .map((s) => ({ src: s.text, tgt: s.trStatus === 'done' ? s.translation : '' }));
}

// The request body for `text`, by the translation settings (`tr` is settings.tr).
export function buildTranslationRequest(tr, glossary, text, context) {
  const p = tr.useLocal ? 'ollama' : tr.provider;
  return {
    provider: p,
    config: { claude: tr.claude, openai: tr.openai, ollama: tr.local }[p],
    text,
    context,
    glossary: relevantGlossary(glossary, [text, ...context.map((c) => c.src)]),
    domain: (tr.domain || '').trim(),
  };
}

// Streams the translation: onText receives the full text so far on every update.
export async function postTranslate(body, onText = () => {}, signal = undefined) {
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
