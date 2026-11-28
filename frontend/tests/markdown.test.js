import { test } from 'node:test';
import assert from 'node:assert/strict';
import { markdownFilename, sessionMarkdown } from '../lib/markdown.js';

test('sessionMarkdown writes the full session and segment content', () => {
  const createdAt = new Date(2026, 0, 2, 3, 4, 5).getTime();
  const segments = [
    {
      ts: new Date(2026, 0, 2, 3, 5, 6).getTime(),
      lang: 'it',
      text: 'Ciao a tutti.',
      translation: 'Hello everyone.',
    },
    {
      ts: new Date(2026, 0, 2, 3, 6, 7).getTime(),
      text: 'Second segment.',
    },
  ];
  assert.equal(sessionMarkdown({ title: 'Biology', createdAt }, segments), [
    '# Biology',
    '',
    '- 创建时间：2026-01-02 03:04',
    '- 段数：2',
    '',
    '---',
    '',
    '**[03:05:06] IT**  ',
    'Ciao a tutti.',
    '',
    '> Hello everyone.',
    '',
    '**[03:06:07] **  ',
    'Second segment.',
    '',
  ].join('\n'));
});

test('sessionMarkdown writes an empty session header without segment rows', () => {
  const createdAt = new Date(2026, 0, 2, 3, 4, 5).getTime();
  assert.equal(sessionMarkdown({ title: 'Empty', createdAt }, []), [
    '# Empty',
    '',
    '- 创建时间：2026-01-02 03:04',
    '- 段数：0',
    '',
    '---',
    '',
  ].join('\n'));
});

test('sessionMarkdown uppercases language and omits a quote for untranslated text', () => {
  const output = sessionMarkdown(
    { title: 'History', createdAt: new Date(2026, 0, 2, 3, 4, 5).getTime() },
    [{ ts: new Date(2026, 0, 2, 3, 4, 6).getTime(), lang: 'it', text: 'La storia continua.' }],
  );
  assert.ok(output.includes('**[03:04:06] IT**  \nLa storia continua.\n'));
  assert.ok(!output.includes('> '));
});

test('sessionMarkdown leaves an absent segment language blank', () => {
  const output = sessionMarkdown(
    { title: 'Notes', createdAt: new Date(2026, 0, 2, 3, 4, 5).getTime() },
    [{ ts: new Date(2026, 0, 2, 3, 4, 6).getTime(), text: 'Text without a language.' }],
  );
  assert.ok(output.includes('**[03:04:06] **  \nText without a language.'));
});

test('markdownFilename replaces Windows-invalid filename characters', () => {
  assert.equal(markdownFilename('a\\b/c:d*e?f"g<h>i|j'), 'a_b_c_d_e_f_g_h_i_j.md');
});
