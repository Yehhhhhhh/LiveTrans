// Text: transcripts split into sentences, and math in translations.

const wordCount = (s) => s.split(/\s+/).filter(Boolean).length;

// Whisper often puts several sentences into one VAD segment; each gets its own
// line and translation. Pieces under 3 words stay with a neighbour.
export function splitSentences(text) {
  const out = [];
  for (const p of text.split(/(?<=[.?!…])\s+(?=[\p{Lu}¿¡"'])/u)) {
    if (out.length && wordCount(out.at(-1)) < 3) out[out.length - 1] += ` ${p}`;
    else out.push(p);
  }
  if (out.length > 1 && wordCount(out.at(-1)) < 3) out[out.length - 2] += ` ${out.pop()}`;
  return out;
}

// Partials drop Whisper's closing period, so it's ignored when matching sentences.
export const liveKey = (s) => s.replace(/[.…]+$/, '').trim();

// Translations may contain LaTeX: \(...\) / $...$ inline, \[...\] / $$...$$ display.
// Math is rendered with KaTeX; everything else stays plain text (never parsed as HTML).
// An unclosed delimiter (e.g. mid-stream) is left as-is until its closing half arrives.
export const MATH_RE = /\\\((.+?)\\\)|\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\$(?!\s)([^$\n]+?)(?<!\s)\$/g;

// The text in order as plain pieces (strings) and math: { tex, display, src }, where `src` is the
// math with its delimiters, shown as it is if KaTeX can't render it.
export function mathParts(text) {
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(MATH_RE)) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const display = m[2] !== undefined || m[3] !== undefined;
    const tex = m[1] ?? m[2] ?? m[3] ?? m[4];
    parts.push({ tex, display, src: m[0] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}
