import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimateVramGb, vramText } from '../lib/vram.js';

test('estimateVramGb calculates large-v3-turbo float16 weights and overhead', () => {
  assert.equal(estimateVramGb('large-v3-turbo', 'float16', false), (809 * 2) / 1024 + 1.5);
});

test('estimateVramGb treats auto as float16 and int8 types as one byte', () => {
  const float16 = (809 * 2) / 1024 + 1.5;
  const int8 = 809 / 1024 + 1.5;
  assert.equal(estimateVramGb('large-v3-turbo', 'auto', false), float16);
  assert.equal(estimateVramGb('large-v3-turbo', 'int8', false), int8);
  assert.equal(estimateVramGb('large-v3-turbo', 'int8_float16', false), int8);
});

test('estimateVramGb returns zero for CPU inference', () => {
  assert.equal(estimateVramGb('unknown', 'unknown', true), 0);
});

test('estimateVramGb returns null for unknown models and compute types', () => {
  assert.equal(estimateVramGb('unknown', 'float16', false), null);
  assert.equal(estimateVramGb('large-v3-turbo', 'unknown', false), null);
});

test('vramText formats used and total GPU memory in gigabytes', () => {
  assert.equal(vramText({ used_mb: 6758, total_mb: 8188 }), '6.6/8GB');
});
