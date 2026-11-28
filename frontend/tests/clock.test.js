import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CLOCK_GAP_MS, CLOCK_MAX, clockTick, wallAt } from '../lib/clock.js';

test('clockTick records the first chunk origin', () => {
  const clock = [];
  clockTick(clock, 320, 1600, 200000);
  assert.deepEqual(clock, [{ pos: 320, wall0: 199880 }]);
});

test('clockTick does not add an on-time chunk', () => {
  const clock = [];
  clockTick(clock, 0, 1600, 100000);
  const firstWall0 = clock[0].wall0;
  clockTick(clock, 1600, 1600, 100200);
  assert.equal(clock.length, 1);
  assert.equal(clock[0].wall0, firstWall0);
});

test('clockTick leaves the origin unchanged for a late chunk within the gap', () => {
  const clock = [];
  clockTick(clock, 0, 1600, 100000);
  clockTick(clock, 1600, 1600, 100500);
  assert.equal(clock.length, 1);
  assert.equal(clock[0].wall0, 99900);
});

test('clockTick moves the origin earlier when a chunk arrives early', () => {
  const clock = [];
  clockTick(clock, 0, 1600, 100000);
  clockTick(clock, 1600, 1600, 99950);
  assert.equal(clock.length, 1);
  assert.equal(clock[0].wall0, 99750);
});

test('clockTick starts a new entry after a gap larger than CLOCK_GAP_MS', () => {
  const clock = [];
  clockTick(clock, 0, 1600, 100000);
  clockTick(clock, 1600, 1600, 100601);
  assert.equal(clock.length, 2);
  assert.ok(clock[1].wall0 - clock[0].wall0 > CLOCK_GAP_MS);
});

test('clockTick keeps at most CLOCK_MAX entries and drops the oldest', () => {
  const clock = [];
  for (let i = 0; i <= CLOCK_MAX; i++) {
    clockTick(clock, i * 1600, 1600, i * 1000 + 100);
  }
  assert.equal(clock.length, CLOCK_MAX);
  assert.equal(clock[0].pos, 1600);
  assert.equal(clock.at(-1).pos, CLOCK_MAX * 1600);
});

test('wallAt uses the first entry before its position and the entry at a boundary', () => {
  const clock = [{ pos: 16000, wall0: 1000 }, { pos: 32000, wall0: 3000 }];
  assert.equal(wallAt(clock, 0.5), 1500);
  assert.equal(wallAt(clock, 1), 2000);
  assert.equal(wallAt(clock, 2), 5000);
});

test('wallAt returns the current time when the clock is empty', () => {
  const before = Date.now();
  const value = wallAt([], 5);
  const after = Date.now();
  assert.ok(value >= before && value <= after);
});
