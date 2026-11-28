// The page's shared state: the settings and the glossary (kept in localStorage), and the state of
// the connection, the capture, the open course record and its translations.
import { DEFAULTS, deepMerge, migrateSettings } from './lib/settings.js';

function loadJson(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: settings just won't persist */
  }
}

export const settings = deepMerge(DEFAULTS, loadJson('livetrans.settings'));
export const saveSettings = () => saveJson('livetrans.settings', settings);
migrateSettings(settings);

// [{ it, en, zh }]; ensureGlossary() loads the default one when none is saved.
export let glossary = loadJson('livetrans.glossary');
export const saveGlossary = () => saveJson('livetrans.glossary', glossary);

export async function ensureGlossary() {
  if (Array.isArray(glossary)) return;
  const res = await fetch('glossary_default.json');
  glossary = await res.json();
  saveGlossary();
}

// 恢复默认: the default glossary in place of the user's.
export async function resetGlossary() {
  glossary = null;
  await ensureGlossary();
}

export const state = {
  ws: null,
  wsReady: false,
  running: false,
  audio: null, // { ctx, node, inputs: { sys?, mic? } }
  switching: false, // a source swap is waiting on a permission prompt
  run: 0, // capture run sent with every 'start'; the backend echoes it in its results
  runClock: new Map(), // run -> its clock (see lib/clock.js); kept for results of runs that already ended
  session: null,
  segments: [], // segments of the current session, in order
  trQueue: [],
  trActive: 0, // final translations in flight
  wakeLock: null,
};
