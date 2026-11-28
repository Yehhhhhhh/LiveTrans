// What the page sends the backend over the WebSocket: the recognition settings, the start and
// stop of each capture run, and its audio, kept until the backend has answered it so that it can
// be sent again after a dropped connection (see lib/resend.js).
import { showMsg } from './dom.js';
import { settings, state, glossary } from './state.js';
import { resetPartialLag } from './status.js';
import { clockTick } from './lib/clock.js';
import { buildAsrPrompt } from './lib/glossary.js';
import { RESEND_MAX_S, commitTo, keepChunk, keepStopped, newResend, resumeSlice, unanswered } from './lib/resend.js';
import { debounce } from './lib/util.js';

export function asrConfig() {
  return {
    type: 'config',
    asr: {
      engine: settings.asr.engine,
      assemblyai_key: settings.asr.assemblyKey,
      assemblyai_model: settings.asr.assemblyModel,
      language: settings.asr.language,
      model: settings.asr.model,
      compute: settings.asr.compute,
      vad_threshold: Number(settings.asr.vadThreshold),
      min_silence_ms: Number(settings.asr.minSilenceMs),
      max_segment_s: Number(settings.asr.maxSegmentS),
      partial: Boolean(settings.asr.partial),
      context: Boolean(settings.asr.context),
    },
    prompt: buildAsrPrompt(settings.asr, glossary),
  };
}

export function wsSend(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

export const sendConfig = debounce(() => wsSend(asrConfig()), 300);

let resend = newResend(); // the current run
const stoppedResend = new Map(); // run -> its buffer, for runs ended with unanswered audio

// Called by start() before the new run is numbered: keep what the ending run still needs.
export function resetResend() {
  keepStopped(stoppedResend, state.run, resend);
  resend = newResend();
}

export function sendAudio(pcm) {
  const clock = state.runClock.get(state.run);
  if (clock) clockTick(clock, resend.pos, pcm.length);
  keepChunk(resend, pcm);
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(pcm);
}

export function onCommitted(pos, run) {
  commitTo(run === state.run ? resend : stoppedResend.get(run), pos);
}

// The backend has answered all of a stopped run's audio.
export function onRunDone(run) {
  stoppedResend.delete(run);
}

// Continue a run's stream where the backend can pick it up and resend its kept audio. Its
// results keep the run's clock (start times continue from `from`, and the audio keeps the time it
// was captured at, not the time it is resent). Returns seconds resent / lost.
function resumeStream(buf = resend, run = state.run) {
  const { from, lost, parts } = resumeSlice(buf);
  wsSend({ type: 'resume', offset: from, run, ...runEngine.get(run) });
  let sent = 0;
  for (const part of parts) {
    state.ws.send(part);
    sent += part.length;
  }
  return { sent: sent / 16000, lost };
}

// On reconnect: resend the stopped runs' unanswered speech (each stopped again), then continue
// the current run, so nothing said around the drop is lost.
export function resumeAll() {
  const stopped = [...stoppedResend].sort((a, b) => a[0] - b[0]);
  if (!state.running && !stopped.length && !unanswered(resend)) return;
  let sent = 0;
  let lost = 0;
  const add = (r) => {
    sent += r.sent;
    lost += r.lost;
  };
  for (const [run, buf] of stopped) {
    add(resumeStream(buf, run));
    wsSend({ type: 'stop' });
  }
  // Always end on the current run: it makes the backend report the stopped runs done.
  add(resumeStream());
  if (!state.running) wsSend({ type: 'stop' });
  if (lost >= 1) showMsg(`断线超过 ${RESEND_MAX_S / 60} 分钟，约 ${Math.round(lost)} 秒语音未能补回`);
  else if (sent >= 1) showMsg(`已重新连接，补发断线前后约 ${Math.round(sent)} 秒语音`);
}

// A new capture run on the backend, with its own stream clock. Runs stopped earlier may still be
// transcribing (model loading, backlog): their results keep their own clock and stay out of the
// current run's live state.
export function startRun() {
  state.run += 1;
  resetPartialLag(); // indices start over with the run
  state.runClock.set(state.run, []);
  for (const r of state.runClock.keys()) if (r < state.run - 20) state.runClock.delete(r);
  runEngine.set(state.run, {
    engine: settings.asr.engine,
    assemblyai_key: settings.asr.assemblyKey,
    assemblyai_model: settings.asr.assemblyModel,
  });
  for (const r of runEngine.keys()) if (r < state.run - 20) runEngine.delete(r);
  wsSend({ type: 'start', run: state.run, ...runEngine.get(state.run) });
}

// run -> the recognition engine (with its key and model) it was started with. Sent again when the
// run is resumed after a dropped connection: a change in the settings applies from the next 开始,
// and a run captured for the local engine must not have its kept audio sent to the cloud.
const runEngine = new Map();
