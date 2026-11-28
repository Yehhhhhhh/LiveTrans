// The WebSocket to the backend: connecting (again, after a drop), and what the page does with each
// message the backend sends.
import { $, setPill, showMsg } from './dom.js';
import { state } from './state.js';
import { onSegment } from './segments.js';
import { loadOtherLangs } from './settings-ui.js';
import { notePartialLag, onBacklog, onEngine, onStatus, resetBacklog, showVad } from './status.js';
import { asrConfig, onCommitted, onRunDone, resumeAll, wsSend } from './stream.js';
import { partialEl, renderPartial, transcriptEl } from './transcript.js';
import { live, liveTranslate, resetLive } from './translation.js';

export function connectWs() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.wsReady = true;
    setPill($('stWs'), '后端：已连接', 'ok');
    resetBacklog();
    wsSend(asrConfig());
    resumeAll();
    loadOtherLangs(); // in case the backend wasn't up when the page loaded
  };

  ws.onclose = () => {
    state.wsReady = false;
    setPill($('stWs'), '后端：未连接，重连中…', 'err');
    clearLiveLines();
    setTimeout(connectWs, 2000);
  };

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    // Each position goes to its own run's buffer: an earlier run's would drop this run's audio.
    if (typeof msg.committed === 'number') onCommitted(msg.committed, msg.committed_run);
    if (msg.type === 'run_done') {
      onRunDone(msg.run);
      return;
    }
    if (msg.type === 'status') onStatus(msg);
    else if (msg.type === 'vad') showVad(msg);
    else if (msg.type === 'segment') onSegment(msg);
    else if (msg.type === 'partial') {
      notePartialLag(msg);
      renderPartial(msg.index, msg.text);
      liveTranslate(msg.index, msg.text);
    } else if (msg.type === 'partial_drop') {
      if (live.index === msg.index) resetLive();
      partialEl(msg.index)?.remove();
    }
    else if (msg.type === 'backlog') onBacklog(msg);
    else if (msg.type === 'engine') onEngine(msg);
    else if (msg.type === 'error') showMsg(msg.message);
  };
}

// When the connection drops, the backend loses the segment being spoken (and its queue),
// and the next connection numbers segments from 1 again: live lines left over from the old
// connection would never be finalized or removed, so they go now, with their live translation.
function clearLiveLines() {
  resetLive();
  transcriptEl.querySelectorAll('.seg.live').forEach((el) => el.remove());
  $('emptyHint').classList.toggle('hidden', transcriptEl.querySelector('.seg') !== null);
}
