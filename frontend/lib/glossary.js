// The glossary ([{ it, en, zh }], newest last): its terms in the Whisper prompt, the entries a
// translation needs, and import from a file.

// Whisper prompt: newest (user-added) terms first, within a character budget
// so the 224-token prompt window isn't overflowed. `asr` is settings.asr.
export function buildAsrPrompt(asr, glossary) {
  // Format "<hint>. <term>, <term>.", or "<hint>. " alone when there are no terms (empty glossary,
  // or 用术语表提示识别 off): the backend reads the hint as the text before the first ". "
  // (asr.split_prompt) to spot Whisper reciting it. An empty hint leaves just the terms.
  const hint = (asr.hint || '').trim().replace(/[\s.]+$/, '');
  const head = hint ? `${hint}. ` : '';
  if (!asr.glossaryPrompt || !glossary) return head;
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
  return terms.length ? `${head}${terms.join(', ')}.` : head;
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// The entries whose Italian or English term appears as a whole word in any of the texts
// (at most 40), sent with a translation request.
export function relevantGlossary(glossary, texts) {
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

// 导入: a .json file (an array of { it, en, zh }), or any other file as a table with one entry
// per line, its columns (it, en, zh) separated by commas, tabs or semicolons. Throws when the
// file can't be read as either.
export function parseGlossary(filename, text) {
  if (filename.toLowerCase().endsWith('.json')) {
    return JSON.parse(text).map((x) => ({ it: x.it || '', en: x.en || '', zh: x.zh || '' }));
  }
  return text
    .split(/\r?\n/)
    .map((l) => l.split(/[,\t;]/).map((x) => x.trim()))
    .filter((c) => c.some(Boolean))
    .map(([it = '', en = '', zh = '']) => ({ it, en, zh }));
}

// Imported entries replace the entry with the same Italian and English terms (ignoring case), or
// are added at the end. Returns how many were added.
export function mergeGlossary(glossary, items) {
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
  return added;
}
