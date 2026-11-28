// Audio capture: the microphone and/or a shared window's sound, mixed into one 16 kHz stream for
// the backend; switching sources while capturing; 开始 / 停止.
import { $, setPill, showMsg } from './dom.js';
import { settings, saveSettings, state } from './state.js';
import { asrConfig, resetResend, sendAudio, startRun, wsSend } from './stream.js';
import { transcriptEl } from './transcript.js';
import { resetLive } from './translation.js';

export async function listMics() {
  const sel = $('micSel');
  let devices = [];
  try {
    devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  } catch {
    /* no permission yet */
  }
  sel.innerHTML = '';
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId;
    o.textContent = d.label || `麦克风 ${i + 1}`;
    sel.appendChild(o);
  });
  if (settings.micDeviceId && devices.some((d) => d.deviceId === settings.micDeviceId)) sel.value = settings.micDeviceId;
}

async function getSystemStream() {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    systemAudio: 'include',
    // For a shared window, capture that window's own sound. Chrome's default is the whole
    // system's audio, so every other program playing sound got subtitles too. (Chrome 141+;
    // older versions ignore the option and still capture everything.)
    windowAudio: 'window',
    selfBrowserSurface: 'exclude',
    // Chrome then offers "share this tab instead" in its share bar, switching tabs without a prompt.
    surfaceSwitching: 'include',
  });
  if (stream.getAudioTracks().length === 0) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error('没有获取到音频：请在共享弹窗中勾选「共享系统音频」或「共享标签页音频」');
  }
  // Only the audio is used; keep the video track alive but idle.
  stream.getVideoTracks().forEach((t) => (t.enabled = false));
  return stream;
}

async function getMicStream() {
  const proc = settings.audio.micProcessing;
  return navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: settings.micDeviceId ? { exact: settings.micDeviceId } : undefined,
      channelCount: 1,
      echoCancellation: proc,
      noiseSuppression: proc,
      autoGainControl: proc,
    },
  });
}

const wantedKinds = (src) => (src === 'both' ? ['sys', 'mic'] : [src === 'system' ? 'sys' : 'mic']);

// 'sys' always comes first in wantedKinds, so the sharing prompt opens while the click's user activation is fresh.
const acquire = (kind) => (kind === 'sys' ? getSystemStream() : getMicStream());

function stopStream(stream) {
  stream.getTracks().forEach((t) => t.stop());
}

function acquireError(e) {
  return e.name === 'NotAllowedError' ? '已取消或未授权音频采集' : e.message;
}

// Plugs a stream into the running graph as the current input of its kind,
// connecting the new one before releasing the old, so audio never gaps.
function attachInput(kind, stream) {
  const a = state.audio;
  const source = a.ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
  const gain = a.ctx.createGain();
  gain.gain.value = kind === 'mic' ? settings.audio.micGain : settings.audio.sysGain;
  source.connect(gain).connect(a.node);
  const input = { kind, stream, source, gain };
  input.onEnded = () => onInputEnded(input);
  stream.getTracks().forEach((t) => t.addEventListener('ended', input.onEnded));
  const old = a.inputs[kind];
  a.inputs[kind] = input;
  if (old) releaseInput(old);
  if (kind === 'sys') describeShare(input);
  updateSourceUi();
}

function releaseInput(input) {
  input.stream.getTracks().forEach((t) => t.removeEventListener('ended', input.onEnded));
  input.gain.disconnect();
  input.source.disconnect();
  stopStream(input.stream);
}

function detachInput(kind) {
  const a = state.audio;
  const input = a?.inputs[kind];
  if (!input) return;
  delete a.inputs[kind];
  releaseInput(input);
  updateSourceUi();
}

// The user ended a share (Chrome's "Stop sharing") or a mic went away.
function onInputEnded(input) {
  const a = state.audio;
  if (!state.running || a?.inputs[input.kind] !== input) return;
  const msg = input.kind === 'sys' ? '屏幕/窗口共享已结束' : '麦克风已断开';
  detachInput(input.kind);
  if (Object.keys(a.inputs).length === 0) {
    stop();
    showMsg(msg);
  } else {
    showMsg(input.kind === 'sys' ? `${msg}，麦克风继续识别；点「共享窗口」可重新共享` : `${msg}，系统音频继续识别`);
  }
}

// Swaps inputs while running: the backend keeps one continuous stream, so VAD and transcript carry on.
async function swapInputs(kinds, replace = false) {
  if (!state.running || state.switching) return false;
  state.switching = true;
  const a = state.audio;
  const fresh = [];
  try {
    for (const kind of kinds) {
      if (replace || !a.inputs[kind]) fresh.push({ kind, stream: await acquire(kind) });
    }
  } catch (e) {
    fresh.forEach((f) => stopStream(f.stream));
    if (state.running) showMsg(acquireError(e) + '，已保留原音源');
    return false;
  } finally {
    state.switching = false;
  }
  if (!state.running || state.audio !== a) {
    fresh.forEach((f) => stopStream(f.stream)); // stopped while the prompt was open
    return false;
  }
  showMsg('');
  fresh.forEach((f) => attachInput(f.kind, f.stream));
  return true;
}

async function onSourceChange(prev) {
  updateSourceUi();
  if (!state.running) return;
  const kinds = wantedKinds(settings.source);
  if (!(await swapInputs(kinds))) {
    // Prompt cancelled: keep capturing what we had, and show that in the selector.
    if (state.running && !kinds.every((k) => state.audio.inputs[k])) {
      settings.source = prev;
      $('sourceSel').value = prev;
      $('micSel').classList.toggle('hidden', prev === 'system');
      saveSettings();
      updateSourceUi();
    }
    return;
  }
  for (const kind of Object.keys(state.audio.inputs)) {
    if (!kinds.includes(kind)) detachInput(kind);
  }
  if (kinds.includes('mic')) listMics();
}

function updateSourceUi() {
  const sys = state.audio?.inputs.sys;
  const btn = $('switchShareBtn');
  btn.classList.toggle('hidden', !state.running || settings.source === 'mic');
  btn.textContent = sys ? '换窗口' : '共享窗口';
  const label = $('shareLabel');
  label.classList.toggle('hidden', !sys);
  if (!sys) {
    label.textContent = '';
    label.title = '';
  }
}

// Shows what is being shared. The browser only reveals the surface type; for a window,
// Chrome's track label carries the native window handle, which the backend resolves to a name.
async function describeShare(input) {
  const track = input.stream.getVideoTracks()[0];
  const surface = track?.getSettings().displaySurface;
  const raw = track?.label || '';
  console.debug('[LiveTrans] shared surface:', surface, JSON.stringify(raw));
  const show = (text, title = text, icon = '') => {
    if (state.audio?.inputs.sys !== input) return; // replaced meanwhile
    const label = $('shareLabel');
    label.textContent = text;
    label.title = title;
    if (icon) {
      const img = document.createElement('img');
      img.className = 'share-icon';
      img.alt = '';
      img.src = icon;
      label.prepend(img);
    }
  };
  if (surface === 'monitor' || /^screen:/.test(raw)) {
    show('🖥 整个屏幕');
  } else if (surface === 'browser') {
    show('🌐 浏览器标签页', '浏览器标签页（浏览器不提供标签页标题）');
  } else if (surface === 'window' || /^window:/.test(raw)) {
    show('🪟 窗口');
    const m = /^window:(\d+):/.exec(raw);
    if (!m) return;
    try {
      const r = await fetch(`/api/window?hwnd=${m[1]}`);
      if (!r.ok) return;
      const w = await r.json();
      const app = w.app || '';
      const title = w.title && w.title !== app ? w.title : '';
      const text = [app, title].filter(Boolean).join(' · ');
      // The app's own icon replaces the generic window emoji when the backend could read it.
      const icon = /^data:image\/png;base64,/.test(w.icon || '') ? w.icon : '';
      if (text) show(icon ? text : `🪟 ${text}`, `窗口：${text}${w.exe ? `（${w.exe}）` : ''}`, icon);
    } catch {
      /* keep the generic label */
    }
  } else {
    show('共享中');
  }
}

async function start() {
  if (!state.wsReady) {
    showMsg('后端未连接，请先运行 start.bat');
    return;
  }
  if (settings.asr.engine === 'assemblyai' && !settings.asr.assemblyKey.trim()) {
    showMsg('请先在「设置 → 识别」填写 AssemblyAI API Key，或把识别引擎改为本地 Whisper');
    return;
  }
  showMsg('');
  // The backend connects to AssemblyAI while the sources are being asked for (the window picker,
  // a permission prompt): audio that had to wait for the connection would keep the whole run's
  // subtitles that far behind.
  const cloud = settings.asr.engine === 'assemblyai';
  if (cloud) {
    wsSend({
      type: 'prepare',
      engine: settings.asr.engine,
      assemblyai_key: settings.asr.assemblyKey,
      assemblyai_model: settings.asr.assemblyModel,
    });
  }
  const src = settings.source;
  const acquired = [];
  try {
    for (const kind of wantedKinds(src)) acquired.push({ kind, stream: await acquire(kind) });
  } catch (e) {
    acquired.forEach((a) => stopStream(a.stream));
    if (cloud) wsSend({ type: 'unprepare' });
    showMsg(acquireError(e));
    return;
  }

  const ctx = new AudioContext({ sampleRate: 16000 });
  await ctx.audioWorklet.addModule('audio-worklet.js');
  const node = new AudioWorkletNode(ctx, 'pcm-capture');
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink).connect(ctx.destination);

  const meter = $('meterBar');
  node.port.onmessage = (e) => {
    sendAudio(e.data.pcm);
    meter.style.width = `${Math.min(100, e.data.rms * 400)}%`;
  };

  state.audio = { ctx, node, inputs: {} };
  state.running = true;
  acquired.forEach((a) => attachInput(a.kind, a.stream));
  resetLive();
  transcriptEl.querySelectorAll('.seg.live').forEach((el) => el.remove());
  wsSend(asrConfig());
  resetResend();
  startRun();
  $('startBtn').textContent = '停止';
  $('startBtn').classList.add('running');
  if (src !== 'system') listMics();
  requestWakeLock();
}

function stop() {
  wsSend({ type: 'stop' });
  const a = state.audio;
  if (a) {
    Object.values(a.inputs).forEach(releaseInput);
    a.ctx.close();
  }
  state.audio = null;
  state.running = false;
  updateSourceUi();
  $('startBtn').textContent = '开始';
  $('startBtn').classList.remove('running');
  $('meterBar').style.width = '0';
  setPill($('stVad'), '● 静音');
  state.wakeLock?.release().catch(() => {});
  state.wakeLock = null;
}

async function requestWakeLock() {
  try {
    state.wakeLock = await navigator.wakeLock?.request('screen');
  } catch {
    /* not critical */
  }
}

export function setupAudio() {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && state.running) requestWakeLock();
  });

  $('startBtn').onclick = () => (state.running ? stop() : start());

  $('sourceSel').value = settings.source;
  $('sourceSel').onchange = () => {
    const prev = settings.source;
    settings.source = $('sourceSel').value;
    $('micSel').classList.toggle('hidden', settings.source === 'system');
    saveSettings();
    onSourceChange(prev);
  };
  $('micSel').classList.toggle('hidden', settings.source === 'system');
  $('micSel').onchange = () => {
    settings.micDeviceId = $('micSel').value;
    saveSettings();
    if (state.audio?.inputs.mic) swapInputs(['mic'], true);
  };
  // Pick another window/tab/screen without stopping recognition.
  $('switchShareBtn').onclick = () => swapInputs(['sys'], true);
  navigator.mediaDevices.addEventListener?.('devicechange', listMics);

  // Leaving the page would end the capture: the browser asks first.
  window.addEventListener('beforeunload', (e) => {
    if (state.running) e.preventDefault();
  });
}
