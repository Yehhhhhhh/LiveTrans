import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULTS, deepMerge, getPath, migrateSettings, setPath } from '../lib/settings.js';

test('deepMerge keeps defaults, merges known nested keys, and leaves base unchanged', () => {
  const base = { enabled: true, audio: { gain: 1, processing: false }, list: [1, 2] };
  const original = structuredClone(base);
  const merged = deepMerge(base, {
    audio: { gain: 2, unknown: true },
    extra: 'discard me',
  });

  assert.deepEqual(merged, { enabled: true, audio: { gain: 2, processing: false }, list: [1, 2] });
  assert.deepEqual(base, original);
});

test('deepMerge returns a deep copy when over is null or undefined', () => {
  for (const over of [null, undefined]) {
    const result = deepMerge(DEFAULTS, over);
    assert.deepEqual(result, DEFAULTS);
    assert.notStrictEqual(result, DEFAULTS);
    assert.notStrictEqual(result.asr, DEFAULTS.asr);
    result.asr.language = 'changed';
    assert.equal(DEFAULTS.asr.language, 'it_en');
  }
});

test('deepMerge replaces arrays as a whole', () => {
  assert.deepEqual(deepMerge({ values: [1, 2, 3] }, { values: [9] }), { values: [9] });
});

test('migrateSettings maps the two legacy values', () => {
  const settings = { asr: { language: 'auto', assemblyModel: 'universal-3-6-pro' } };
  migrateSettings(settings);
  assert.deepEqual(settings.asr, {
    language: 'it_en',
    assemblyModel: 'assemblyai-stt-3-7-realtime-agent',
  });
});

test('migrateSettings leaves other values unchanged', () => {
  const settings = { asr: { language: 'fr', assemblyModel: 'custom-model' } };
  migrateSettings(settings);
  assert.deepEqual(settings.asr, { language: 'fr', assemblyModel: 'custom-model' });
});

test('getPath reads a dotted path', () => {
  assert.equal(getPath({ tr: { openai: { model: 'gpt' } } }, 'tr.openai.model'), 'gpt');
});

test('setPath writes a dotted path', () => {
  const settings = { tr: { openai: { model: '' } } };
  setPath(settings, 'tr.openai.model', 'gpt');
  assert.equal(settings.tr.openai.model, 'gpt');
});
