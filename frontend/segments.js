// Final transcripts: a segment's sentences become lines of the open course record, timed by when
// their audio was captured, saved, and sent for translation.
import * as db from './db.js';
import { $, setPill } from './dom.js';
import { settings, state } from './state.js';
import { audioLag, noteSegment, partialLag } from './status.js';
import { partialEl, renderSeg, transcriptEl } from './transcript.js';
import { enqueueTranslation, live, resetLive, takeLive } from './translation.js';
import { wallAt } from './lib/clock.js';
import { splitSentences } from './lib/text.js';

export async function onSegment(msg) {
  const text = (msg.text || '').trim();
  // A result of an earlier run: its indices don't refer to the current run's partials.
  const current = msg.run === undefined || msg.run === state.run;
  const clock = state.runClock.get(current ? state.run : msg.run) ?? state.runClock.get(state.run);
  if (current) {
    // Partials of earlier segments can no longer be finalized.
    transcriptEl.querySelectorAll('.seg.live').forEach((el) => {
      if (Number(el.dataset.pidx) < msg.index) el.remove();
    });
  }
  if (!text || !state.session) {
    if (current) {
      if (live.index <= msg.index) resetLive();
      partialEl(msg.index)?.remove();
    }
    return;
  }
  // One line per sentence; timestamps are spread over the segment by text length.
  const sentences = splitSentences(text);
  const total = sentences.reduce((n, s) => n + s.length, 0);
  let offset = 0;
  // Recognition delay: audio sent -> its text received. The segment's last partial measured
  // exactly that; the final result also waited for the end of the sentence to be detected, so
  // it only counts for a segment that had no partial.
  const asrLag =
    current && partialLag.index === msg.index && partialLag.lag != null
      ? partialLag.lag
      : audioLag(clock, msg.start + msg.duration);
  const segs = sentences.map((s) => {
    const seg = {
      id: crypto.randomUUID(),
      sessionId: state.session.id,
      ts: wallAt(clock, msg.start + (msg.duration * offset) / total),
      lang: msg.lang,
      text: s,
      translation: '',
      trStatus: settings.tr.enabled ? 'pending' : 'off',
      error: '',
      asrLag,
    };
    offset += s.length;
    return seg;
  });
  noteSegment(msg, asrLag);
  if (settings.tr.enabled && current) takeLive(segs, msg.index);
  if (segs.some((s) => s.trStatus === 'done')) {
    setPill($('stTr'), `翻译：说话时已完成 · 队列 ${state.trQueue.length}`, 'ok');
  }
  let prev = current ? partialEl(msg.index) : null;
  segs.forEach((seg, i) => {
    state.segments.push(seg);
    prev = i === 0 ? renderSeg(seg, prev) : renderSeg(seg, null, prev);
  });
  for (const seg of segs) {
    await db.putSegment(seg);
    if (seg.trStatus === 'pending') enqueueTranslation(seg);
  }
}
