import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildTranslationRequest, pickContext, postTranslate } from '../lib/translate.js';

test('pickContext selects the last eligible segments before the target', () => {
  const list = [
    { id: 1, text: 'done source', trStatus: 'done', translation: 'done target' },
    { id: 2, text: 'failed source', trStatus: 'failed', translation: 'ignored' },
    { id: 3, text: 'pending source', trStatus: 'pending', translation: 'ignored' },
    { id: 4, text: 'streaming source', trStatus: 'streaming', translation: 'ignored' },
    { id: 5, text: 'target', trStatus: 'done', translation: 'target translation' },
  ];
  assert.deepEqual(pickContext(list, list[4], '2'), [
    { src: 'pending source', tgt: '' },
    { src: 'streaming source', tgt: '' },
  ]);
});

test('pickContext uses the whole list when the target is null or absent', () => {
  const list = [
    { id: 1, text: 'one', trStatus: 'done', translation: 'uno' },
    { id: 2, text: 'two', trStatus: 'off', translation: 'ignored' },
  ];
  assert.deepEqual(pickContext(list, null, 5), [{ src: 'one', tgt: 'uno' }]);
  assert.deepEqual(pickContext(list, { id: 9 }, 5), [{ src: 'one', tgt: 'uno' }]);
});

test('pickContext with context size zero (or below one) returns no history', () => {
  const list = [{ id: 1, text: 'one', trStatus: 'done', translation: 'uno' }];
  assert.deepEqual(pickContext(list, null, 0), []);
  assert.deepEqual(pickContext(list, null, '0'), []);
  assert.deepEqual(pickContext(list, null, 0.5), []);
  assert.deepEqual(pickContext(list, null, ''), []);
});

test('pickContext rounds a fractional context size down', () => {
  const list = [
    { id: 1, text: 'one', trStatus: 'done', translation: 'uno' },
    { id: 2, text: 'two', trStatus: 'done', translation: 'due' },
  ];
  assert.deepEqual(pickContext(list, null, 1.9), [{ src: 'two', tgt: 'due' }]);
});

test('buildTranslationRequest uses the local provider and matches glossary across source context', () => {
  const tr = {
    useLocal: true,
    provider: 'claude',
    local: { model: 'local-model' },
    claude: { model: 'claude-model' },
    openai: { model: 'openai-model' },
    domain: '  biology  ',
  };
  const glossary = [
    { it: 'gravità', en: 'gravity', zh: '重力' },
    { it: 'elettrone', en: 'electron', zh: '电子' },
    { it: 'casa', en: 'house', zh: '房子' },
  ];
  assert.deepEqual(buildTranslationRequest(tr, glossary, 'Gravity acts', [{ src: 'An electron moves', tgt: '' }]), {
    provider: 'ollama',
    config: tr.local,
    text: 'Gravity acts',
    context: [{ src: 'An electron moves', tgt: '' }],
    glossary: [glossary[0], glossary[1]],
    domain: 'biology',
  });
});

test('buildTranslationRequest selects the configured provider and defaults an absent domain', () => {
  const tr = { useLocal: false, provider: 'claude', claude: { model: 'claude' }, openai: { model: 'openai' } };
  const request = buildTranslationRequest(tr, [], 'text', []);
  assert.equal(request.provider, 'claude');
  assert.strictEqual(request.config, tr.claude);
  assert.equal(request.domain, '');
  assert.deepEqual(request.glossary, []);
});

function streamResponse(chunks, options = {}) {
  const body = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
      controller.close();
    },
  });
  return new Response(body, options);
}

async function withFetch(fetchImpl, action) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = fetchImpl;
  try {
    return await action();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

test('postTranslate posts JSON and parses split NDJSON lines and updates', async () => {
  const signal = new AbortController().signal;
  const calls = [];
  const updates = [];
  const result = await withFetch(async (...args) => {
    calls.push(args);
    return streamResponse([
      '{"text":"hel',
      'lo"}\n\n{"text":"hello world"}\n{"done":true,"translation":"final"}\n',
    ]);
  }, () => postTranslate({ text: 'source' }, (text) => updates.push(text), signal));

  assert.equal(result, 'final');
  assert.deepEqual(updates, ['hello', 'hello world']);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], '/api/translate/stream');
  assert.equal(calls[0][1].method, 'POST');
  assert.equal(calls[0][1].headers['Content-Type'], 'application/json');
  assert.equal(calls[0][1].body, JSON.stringify({ text: 'source' }));
  assert.strictEqual(calls[0][1].signal, signal);
});

test('postTranslate rejects an HTTP error using JSON detail', async () => {
  await assert.rejects(
    withFetch(async () => streamResponse(['{"detail":"backend unavailable"}'], {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    }), () => postTranslate({})),
    { message: 'backend unavailable' },
  );
});

test('postTranslate falls back to the HTTP status when the error body is not JSON', async () => {
  await assert.rejects(
    withFetch(async () => streamResponse(['not json'], { status: 500 }), () => postTranslate({})),
    { message: 'HTTP 500' },
  );
});

test('postTranslate rejects an error message from the stream', async () => {
  await assert.rejects(
    withFetch(async () => streamResponse(['{"error":"translation failed"}\n']), () => postTranslate({})),
    { message: 'translation failed' },
  );
});

test('postTranslate rejects a stream that ends without a done message', async () => {
  await assert.rejects(
    withFetch(async () => streamResponse(['{"text":"partial"}\n']), () => postTranslate({})),
    { message: '\u7ffb\u8bd1\u6d41\u610f\u5916\u4e2d\u65ad' },
  );
});
