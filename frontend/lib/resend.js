// Resuming after a dropped connection (backend restarted, network blip). Audio sent to the
// backend is kept until the backend reports it "committed": every segment in it answered, or
// silence. Audio captured while the connection is down is kept as well. On reconnect the stream
// resumes at the committed position and the kept audio is sent again, so what was said around
// the drop is still transcribed, with its own timestamps. At most RESEND_MAX_S is kept.
// Positions count samples (16 kHz) since 开始.
//
// A buffer is { chunks: [{ pos, pcm }], pos, committed }: the audio kept, the position after the
// last sample captured, and the position the backend has committed.
export const RESEND_MAX_S = 120;
// A stopped run whose last speech was still being transcribed when 开始 was pressed again keeps
// its audio here until the backend reports the run done ("run_done"); at most this many runs.
export const STOPPED_RUNS_MAX = 5;
export const newResend = () => ({ chunks: [], pos: 0, committed: 0 });

export const unanswered = (buf) => buf.chunks.length > 0 && buf.committed < buf.pos;

export function trimResend(buf) {
  const c = buf.chunks;
  while (c.length && c[0].pos + c[0].pcm.length <= buf.committed) c.shift();
  while (c.length && buf.pos - c[0].pos > RESEND_MAX_S * 16000) c.shift();
}

// A chunk as it is captured (sent now, or later if the connection is down).
export function keepChunk(buf, pcm) {
  buf.chunks.push({ pos: buf.pos, pcm });
  buf.pos += pcm.length;
  trimResend(buf);
}

// The backend reports the stream committed up to `pos`: what comes before needs no resending.
export function commitTo(buf, pos) {
  if (!buf || pos <= buf.committed) return;
  buf.committed = pos;
  trimResend(buf);
}

// When 开始 starts a new run: the ending run's buffer is kept in `stopped` (run -> buffer) while
// any of its audio is unanswered, the oldest run dropped beyond STOPPED_RUNS_MAX.
export function keepStopped(stopped, run, buf) {
  if (!unanswered(buf)) return;
  stopped.set(run, buf);
  while (stopped.size > STOPPED_RUNS_MAX) stopped.delete(stopped.keys().next().value);
}

// Where a run's stream continues on a new connection, and the kept audio to send again from
// there: the committed position, or the oldest audio still kept when more than RESEND_MAX_S went
// unanswered (the audio before it is lost). The buffer counts that position as committed from now.
// Returns the position, the seconds lost, and the parts of the kept chunks to send again.
export function resumeSlice(buf) {
  const first = buf.chunks[0];
  const from = Math.max(buf.committed, first ? first.pos : buf.pos);
  const lost = (from - buf.committed) / 16000;
  buf.committed = from;
  const parts = [];
  for (const ch of buf.chunks) {
    if (ch.pos + ch.pcm.length <= from) continue;
    parts.push(ch.pos >= from ? ch.pcm : ch.pcm.subarray(from - ch.pos));
  }
  return { from, lost, parts };
}
