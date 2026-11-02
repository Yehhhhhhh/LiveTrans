// Minimal IndexedDB wrapper: sessions + segments, so transcripts survive reloads.
const DB_NAME = 'livetrans';
const DB_VERSION = 1;

let dbPromise = null;

function open() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('sessions')) {
        db.createObjectStore('sessions', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('segments')) {
        const s = db.createObjectStore('segments', { keyPath: 'id' });
        s.createIndex('bySession', 'sessionId');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function wrap(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function store(name, mode = 'readonly') {
  const db = await open();
  return db.transaction(name, mode).objectStore(name);
}

export async function listSessions() {
  const all = await wrap((await store('sessions')).getAll());
  return all.sort((a, b) => b.createdAt - a.createdAt);
}

export async function putSession(session) {
  return wrap((await store('sessions', 'readwrite')).put(session));
}

export async function deleteSession(id) {
  const db = await open();
  const tx = db.transaction(['sessions', 'segments'], 'readwrite');
  tx.objectStore('sessions').delete(id);
  const idx = tx.objectStore('segments').index('bySession');
  const keys = await wrap(idx.getAllKeys(id));
  for (const k of keys) tx.objectStore('segments').delete(k);
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

export async function getSegments(sessionId) {
  const idx = (await store('segments')).index('bySession');
  const all = await wrap(idx.getAll(sessionId));
  return all.sort((a, b) => a.ts - b.ts || a.id.localeCompare(b.id));
}

export async function putSegment(seg) {
  return wrap((await store('segments', 'readwrite')).put(seg));
}
