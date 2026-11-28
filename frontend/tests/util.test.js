import { test } from 'node:test';
import assert from 'node:assert/strict';
import { debounce, fmtDateTime, fmtLag, fmtTime } from '../lib/util.js';

test('fmtTime formats local time as zero-padded HH:MM:SS', () => {
  assert.equal(fmtTime(new Date(2026, 0, 2, 3, 4, 5).getTime()), '03:04:05');
});

test('fmtDateTime formats local date and time with zero padding', () => {
  assert.equal(fmtDateTime(new Date(2026, 0, 2, 3, 4, 5).getTime()), '2026-01-02 03:04');
});

test('fmtLag rounds milliseconds and formats seconds to one decimal place', () => {
  assert.equal(fmtLag(123.4), '123 ms');
  assert.equal(fmtLag(1499), '1.5 s');
});

test('fmtLag rounds a value below one second after choosing the millisecond format', () => {
  assert.equal(fmtLag(999.5), '1000 ms');
});

test('debounce runs once after the final call with its final arguments', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const calls = [];
  const fn = debounce((...args) => calls.push(args), 100);

  fn('first');
  t.mock.timers.tick(50);
  fn('last', 2);
  t.mock.timers.tick(99);
  assert.deepEqual(calls, []);
  t.mock.timers.tick(1);
  assert.deepEqual(calls, [['last', 2]]);
});
