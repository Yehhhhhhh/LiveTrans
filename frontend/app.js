// LiveTrans page: captures the lecture's audio, streams it to the local backend over a WebSocket
// (speech detection and recognition run there), shows each sentence with its translation, and
// keeps every course record in IndexedDB. This file only wires the modules together:
//
//   state.js        settings, glossary and shared state      dom.js          DOM helpers, toast
//   audio.js        capture, sources, 开始 / 停止             stream.js       what is sent: config, audio, runs
//   connection.js   the WebSocket, what the backend sends    segments.js     final transcripts -> lines
//   translation.js  final translations and live drafts       transcript.js   rendering the lines
//   status.js       the status bar                           sessions.js     course records, 导出 MD
//   settings-ui.js  the 设置 panel                           glossary-ui.js  the 术语表 tab
//   db.js           IndexedDB                                lib/            pure logic (tests/)
import { setupAudio, listMics } from './audio.js';
import { connectWs } from './connection.js';
import { setupToast } from './dom.js';
import { renderGlossary, setupGlossary } from './glossary-ui.js';
import { initSessions, setupSessions } from './sessions.js';
import { setupSettings } from './settings-ui.js';
import { ensureGlossary } from './state.js';
import { setupStatus } from './status.js';
import { applyDisplay, setupTranscript } from './transcript.js';
import { enqueueTranslation } from './translation.js';

setupToast();
setupStatus();
setupTranscript({ onRetry: enqueueTranslation });
setupSessions();
setupAudio();
setupSettings();
setupGlossary();

(async function boot() {
  applyDisplay();
  await ensureGlossary();
  renderGlossary();
  await initSessions();
  await listMics();
  connectWs();
})();
