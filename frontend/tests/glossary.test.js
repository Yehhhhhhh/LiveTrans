import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildAsrPrompt, mergeGlossary, parseGlossary, relevantGlossary } from '../lib/glossary.js';

test('buildAsrPrompt puts the cleaned hint first and newest English and Italian terms next', () => {
  const asr = { hint: '  Biology lecture...  ', glossaryPrompt: true };
  const glossary = [
    { it: 'cellula', en: 'cell', zh: '细胞' },
    { it: 'gravità', en: 'gravity', zh: '重力' },
  ];
  assert.equal(buildAsrPrompt(asr, glossary), 'Biology lecture. gravity, gravità, cell, cellula.');
});

test('buildAsrPrompt returns only its head when terms are unavailable or disabled', () => {
  assert.equal(buildAsrPrompt({ hint: '  Physics.  ', glossaryPrompt: true }, null), 'Physics. ');
  assert.equal(buildAsrPrompt({ hint: 'Physics', glossaryPrompt: false }, [{ it: 'fisica', en: 'physics' }]), 'Physics. ');
  assert.equal(buildAsrPrompt({ hint: '  ', glossaryPrompt: true }, null), '');
});

test('buildAsrPrompt skips empty terms and deduplicates without regard to case', () => {
  const glossary = [
    { it: 'cella', en: 'Cell', zh: '' },
    { it: '', en: 'cell', zh: '' },
  ];
  assert.equal(buildAsrPrompt({ glossaryPrompt: true }, glossary), 'cell, cella.');
});

test('buildAsrPrompt stops before older entries once the character budget is reached', () => {
  const glossary = [
    { it: 'vecchio', en: 'old', zh: '' },
    { it: 'NEWITALIAN', en: 'N'.repeat(598), zh: '' },
  ];
  const prompt = buildAsrPrompt({ glossaryPrompt: true }, glossary);
  assert.ok(prompt.includes('NEWITALIAN'));
  assert.ok(!prompt.includes('old'));
});

test('relevantGlossary matches whole words case-insensitively and escapes special characters', () => {
  const glossary = [
    { it: 'isi', en: 'isi', zh: 'term', extra: true },
    { it: 'crisi', en: 'crisis', zh: 'crisis' },
    { it: 'caffè', en: 'coffee', zh: 'coffee' },
    { it: 'C++', en: 'plus', zh: 'plus' },
    { it: 'a', en: 'x', zh: 'short' },
  ];
  assert.deepEqual(relevantGlossary(glossary, ['CRISIS', 'A caffè break', 'C++ is useful']), [
    { it: 'crisi', en: 'crisis', zh: 'crisis' },
    { it: 'caffè', en: 'coffee', zh: 'coffee' },
    { it: 'C++', en: 'plus', zh: 'plus' },
  ]);
});

test('relevantGlossary matches across any text and returns an empty array for null', () => {
  const glossary = [{ it: 'elettrone', en: 'electron', zh: '电子' }];
  assert.deepEqual(relevantGlossary(glossary, ['unrelated', 'The ELECTRON moves']), glossary);
  assert.deepEqual(relevantGlossary(null, ['electron']), []);
});

test('relevantGlossary returns at most 40 matching entries', () => {
  const glossary = Array.from({ length: 45 }, (_, i) => ({ it: '', en: `term${i}`, zh: '' }));
  const text = glossary.map((item) => item.en).join(' ');
  assert.equal(relevantGlossary(glossary, [text]).length, 40);
});

test('parseGlossary reads JSON extensions case-insensitively and fills missing columns', () => {
  assert.deepEqual(parseGlossary('terms.JSON', '[{"it":"casa","en":"house"},{"zh":"房子"}]'), [
    { it: 'casa', en: 'house', zh: '' },
    { it: '', en: '', zh: '房子' },
  ]);
});

test('parseGlossary parses delimited rows and rejects non-array JSON', () => {
  assert.deepEqual(parseGlossary('terms.csv', ' casa , house ; 房子\n\nlibro\tbook\t书\nsolo'), [
    { it: 'casa', en: 'house', zh: '房子' },
    { it: 'libro', en: 'book', zh: '书' },
    { it: 'solo', en: '', zh: '' },
  ]);
  assert.throws(() => parseGlossary('terms.json', '{"it":"casa"}'), TypeError);
});

test('mergeGlossary updates matching objects and appends each repeated new import', () => {
  const existing = { it: 'Casa', en: 'House', zh: '旧', keep: true };
  const glossary = [existing];
  const addedA = { it: 'Libro', en: 'Book', zh: '书' };
  const addedB = { it: 'libro', en: 'BOOK', zh: '本' };
  const added = mergeGlossary(glossary, [
    { it: 'casa', en: 'HOUSE', zh: '房子' },
    addedA,
    addedB,
  ]);
  assert.equal(added, 2);
  assert.strictEqual(glossary[0], existing);
  assert.deepEqual(existing, { it: 'casa', en: 'HOUSE', zh: '房子', keep: true });
  assert.strictEqual(glossary[1], addedA);
  assert.strictEqual(glossary[2], addedB);
});
