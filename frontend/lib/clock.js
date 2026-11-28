// A run's clock: when its audio was captured, by the computer's clock. Counting samples from the
// moment 开始 was pressed is not enough: the audio clock runs a little off the computer's, and
// stands still whenever no audio arrives (computer asleep, audio device changed, page frozen), so
// every later sentence would be shown too early. Each entry says "from sample `pos` on, sample 0
// would have been captured at `wall0`" (ms). Positions count samples (16 kHz) since 开始.
export const CLOCK_GAP_MS = 500; // audio arriving this much later than expected starts a new entry
export const CLOCK_MAX = 500;

// Called for every chunk as it arrives; `pos` is the position of its first sample.
export function clockTick(clock, pos, len, now = Date.now()) {
  const wall0 = now - ((pos + len) / 16000) * 1000;
  const last = clock[clock.length - 1];
  if (!last || wall0 > last.wall0 + CLOCK_GAP_MS) {
    clock.push({ pos, wall0 });
    if (clock.length > CLOCK_MAX) clock.shift();
  } else if (wall0 < last.wall0) {
    // Chunks reach the page a little late, never early: the earliest reading is the truest.
    // (After a busy page delivers a burst of late chunks, this also brings the entry back.)
    last.wall0 = wall0;
  }
}

// When the audio `sec` seconds into a run was captured (ms since the epoch).
export function wallAt(clock, sec) {
  if (!clock?.length) return Date.now();
  let at = clock[0];
  for (const c of clock) {
    if (c.pos > sec * 16000) break;
    at = c;
  }
  return at.wall0 + sec * 1000;
}
