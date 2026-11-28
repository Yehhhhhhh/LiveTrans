import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MATH_RE, liveKey, mathParts, splitSentences } from '../lib/text.js';

test('splitSentences breaks at sentence punctuation before capitals and opening marks', () => {
  const text = 'This sentence has enough words. ¿Quién habla ahora mismo? ¡Todos escuchan muy bien! "Quoted words continue onward. \'Other words continue onward.';
  assert.deepEqual(splitSentences(text), [
    'This sentence has enough words.',
    '¿Quién habla ahora mismo?',
    '¡Todos escuchan muy bien!',
    '"Quoted words continue onward.',
    "'Other words continue onward.",
  ]);
});

test('splitSentences joins short opening, middle, and ending fragments to neighbours', () => {
  assert.deepEqual(splitSentences('Hi. This sentence has enough words. Yes. Another sentence has enough words. End.'), [
    'Hi. This sentence has enough words.',
    'Yes. Another sentence has enough words. End.',
  ]);
});

test('splitSentences does not split before lowercase text or inside decimals', () => {
  assert.deepEqual(splitSentences('This sentence has three words. then lowercase continues with 3.14 points. Next sentence has four words.'), [
    'This sentence has three words. then lowercase continues with 3.14 points.',
    'Next sentence has four words.',
  ]);
});

test('splitSentences handles Italian sentences', () => {
  assert.deepEqual(splitSentences('La lezione inizia presto. Il professore spiega la storia.'), [
    'La lezione inizia presto.',
    'Il professore spiega la storia.',
  ]);
});

test('splitSentences recognizes ellipsis as sentence punctuation', () => {
  assert.deepEqual(splitSentences('Three words finish… Four words start next.'), [
    'Three words finish…',
    'Four words start next.',
  ]);
});

test('liveKey strips punctuation at the end before trimming whitespace', () => {
  assert.equal(liveKey('A sentence…'), 'A sentence');
  assert.equal(liveKey('  A sentence…  '), 'A sentence…');
  assert.equal(liveKey('A sentence...'), 'A sentence');
  assert.equal(liveKey('A question?  '), 'A question?');
});

test('mathParts returns inline and display math in order across multiline display blocks', () => {
  const input = 'before \\(x+1\\) and $y^2$ then \\[a +\nb\\] finally $$c +\nd$$ after';
  assert.deepEqual(mathParts(input), [
    'before ',
    { tex: 'x+1', display: false, src: '\\(x+1\\)' },
    ' and ',
    { tex: 'y^2', display: false, src: '$y^2$' },
    ' then ',
    { tex: 'a +\nb', display: true, src: '\\[a +\nb\\]' },
    ' finally ',
    { tex: 'c +\nd', display: true, src: '$$c +\nd$$' },
    ' after',
  ]);
});

test('mathParts leaves spaced dollars, prices, unclosed delimiters, and multiline inline math as text', () => {
  const input = '$ x$ / $x $ / costs $5 and $10 / $unfinished / \\(a\nb\\)';
  assert.deepEqual(mathParts(input), [input]);
  assert.equal([...input.matchAll(MATH_RE)].length, 0);
});
