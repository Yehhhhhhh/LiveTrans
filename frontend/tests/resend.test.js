import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  RESEND_MAX_S,
  STOPPED_RUNS_MAX,
  commitTo,
  keepChunk,
  keepStopped,
  newResend,
  resumeSlice,
  trimResend,
  unanswered,
} from '../lib/resend.js';

test('keepChunk appends a chunk at the current position and advances pos', () => {
  const buf = newResend();
  const first = new Int16Array([1, 2]);
  const second = new Int16Array([3]);
  keepChunk(buf, first);
  keepChunk(buf, second);
  assert.deepEqual(buf.chunks, [{ pos: 0, pcm: first }, { pos: 2, pcm: second }]);
  assert.equal(buf.pos, 3);
});

test('trimResend drops fully committed chunks and retains a partly committed chunk', () => {
  const buf = newResend();
  keepChunk(buf, new Int16Array([1, 2, 3, 4]));
  keepChunk(buf, new Int16Array([5, 6, 7, 8]));
  buf.committed = 6;
  trimResend(buf);
  assert.equal(buf.chunks.length, 1);
  assert.equal(buf.chunks[0].pos, 4);
  assert.deepEqual([...buf.chunks[0].pcm], [5, 6, 7, 8]);
});

test('trimResend retains only recent chunks, including a small newest chunk', () => {
  const buf = {
    chunks: [
      { pos: 0, pcm: { length: 1 } },
      { pos: 1, pcm: { length: 1 } },
      { pos: RESEND_MAX_S * 16000 + 1, pcm: { length: 1 } },
    ],
    pos: RESEND_MAX_S * 16000 + 2,
    committed: 0,
  };
  trimResend(buf);
  assert.deepEqual(buf.chunks.map((chunk) => chunk.pos), [RESEND_MAX_S * 16000 + 1]);
});

test('trimResend removes a lone chunk larger than the retention window', () => {
  const buf = newResend();
  keepChunk(buf, { length: RESEND_MAX_S * 16000 + 1 });
  assert.deepEqual(buf.chunks, []);
});

test('commitTo only advances and tolerates an undefined buffer', () => {
  const buf = newResend();
  commitTo(undefined, 10);
  commitTo(buf, 10);
  commitTo(buf, 4);
  assert.equal(buf.committed, 10);
});

test('unanswered detects audio beyond the committed position', () => {
  const buf = newResend();
  assert.equal(unanswered(buf), false);
  keepChunk(buf, new Int16Array(10));
  assert.equal(unanswered(buf), true);
  commitTo(buf, 10);
  assert.equal(unanswered(buf), false);
});

test('keepStopped stores only unanswered runs and evicts the oldest beyond the limit', () => {
  const stopped = new Map();
  keepStopped(stopped, 'empty', newResend());
  for (let i = 0; i <= STOPPED_RUNS_MAX; i++) {
    const buf = newResend();
    keepChunk(buf, new Int16Array(1));
    keepStopped(stopped, `run-${i}`, buf);
  }
  assert.equal(stopped.size, STOPPED_RUNS_MAX);
  assert.deepEqual([...stopped.keys()], Array.from({ length: STOPPED_RUNS_MAX }, (_, i) => `run-${i + 1}`));
});

test('resumeSlice starts inside a partly committed chunk and keeps later chunks', () => {
  const buf = newResend();
  keepChunk(buf, new Int16Array([1, 2, 3, 4]));
  keepChunk(buf, new Int16Array([5, 6, 7, 8]));
  commitTo(buf, 2);
  const slice = resumeSlice(buf);
  assert.equal(slice.from, 2);
  assert.equal(slice.lost, 0);
  assert.deepEqual(slice.parts.map((part) => [...part]), [[3, 4], [5, 6, 7, 8]]);
  assert.equal(buf.committed, 2);
});

test('resumeSlice accounts for audio trimmed before the earliest kept chunk', () => {
  const buf = {
    chunks: [
      { pos: 16000, pcm: new Int16Array([1, 2]) },
      { pos: 32000, pcm: new Int16Array([3, 4]) },
    ],
    pos: 121 * 16000,
    committed: 0,
  };
  const slice = resumeSlice(buf);
  assert.equal(slice.from, 16000);
  assert.equal(slice.lost, 1);
  assert.deepEqual(slice.parts.map((part) => [...part]), [[1, 2], [3, 4]]);
  assert.equal(buf.committed, slice.from);
});

test('resumeSlice returns no parts or loss for committed and empty audio', () => {
  const committed = newResend();
  keepChunk(committed, new Int16Array([1, 2]));
  commitTo(committed, 2);
  assert.deepEqual(resumeSlice(committed), { from: 2, lost: 0, parts: [] });

  const empty = newResend();
  assert.deepEqual(resumeSlice(empty), { from: 0, lost: 0, parts: [] });
});
