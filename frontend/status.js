// The status bar: speech detection, recognition time / delay / backlog, the recognition model
// (and a model download), and the GPU's memory; also the model choices in 设置 → 识别.
import { $, setPill, showMsg } from './dom.js';
import { settings, saveSettings, state } from './state.js';
import { wallAt } from './lib/clock.js';
import { fmtLag } from './lib/util.js';
import { estimateVramGb, vramText } from './lib/vram.js';

// Live speech probability next to the threshold, so the threshold can be set just below the speaker.
export function showVad(msg) {
  const el = $('stVad');
  let text = msg.speaking ? '● 说话中' : '● 静音';
  if (typeof msg.prob === 'number') {
    text += ` 人声 ${msg.prob.toFixed(2)} / 阈值 ${Number(settings.asr.vadThreshold).toFixed(2)}`;
  }
  setPill(el, text, msg.speaking ? 'speaking' : '');
  el.title = '人声概率高于阈值才会开始识别。老师说话时人声值总低于阈值，就把「语音判定阈值」调低一些。';
}

// Time of the last final transcription, plus the backlog once the GPU falls behind the speaker
// (pending counts the segment being transcribed, so 1 is normal).
// `lag`: how long after a piece of audio was sent its text came back (see audioLag).
const asrStat = { ms: null, lag: null, merged: 1, pending: 0 };
// Audio is sent as it is captured, so the audio at `audioEnd` (seconds on a run's stream) left
// the page when the run's clock says it was captured (see wallAt). A text that covers the audio
// up to there arrived now.
export const audioLag = (clock, audioEnd) => Math.max(0, Math.round(Date.now() - wallAt(clock, audioEnd)));
// The delay of the newest partial transcript, and the segment it belongs to.
export let partialLag = { index: -1, lag: null };
export function notePartialLag(msg) {
  if (typeof msg.audio_end !== 'number') return;
  partialLag = { index: msg.index, lag: audioLag(state.runClock.get(state.run), msg.audio_end) };
  asrStat.lag = partialLag.lag;
  showAsr();
}
// A new run: its segment indices start over.
export function resetPartialLag() {
  partialLag = { index: -1, lag: null };
}
function showAsr() {
  const el = $('stAsr');
  // AssemblyAI reports no computation time (asr_ms is 0): only the delay is shown.
  const parts = [];
  if (asrStat.ms) parts.push(`${asrStat.ms} ms`);
  if (asrStat.lag != null) parts.push(`延迟 ${fmtLag(asrStat.lag)}`);
  let text = `识别：${parts.join(' · ') || (asrStat.ms == null ? '-' : '0 ms')}`;
  if (asrStat.merged > 1) text += ` · 合并 ${asrStat.merged} 段追赶`;
  const behind = asrStat.pending >= 2;
  if (behind) text += ` · 待识别 ${asrStat.pending} 段`;
  const warn = behind || asrStat.merged > 1;
  setPill(el, text, warn ? 'warn' : parts.length || asrStat.ms != null ? 'ok' : '');
  el.title = warn
    ? '识别速度跟不上说话速度，字幕会滞后。积压时自动把排队的几段合并成一次识别来追赶进度（自动语种时合并的几段共用一个语种）。经常出现可在「设置 → 识别」关闭逐词实时显示，或换用更小的模型。'
    : '延迟：音频发送出去，到收到它对应的文字所用的时间。';
}

// A final transcription: its computation time and delay, and how many queued segments it merged.
export function noteSegment(msg, lag) {
  asrStat.ms = msg.asr_ms;
  asrStat.lag = lag;
  asrStat.merged = msg.merged || 1;
  showAsr();
}

export function onBacklog(msg) {
  asrStat.pending = msg.pending;
  if (!msg.pending) asrStat.merged = 1; // caught up: the merge note no longer applies
  showAsr();
}

// A new connection starts with an empty queue (a dropped one's backlog is gone),
// so neither the backlog nor the last merge note applies any more.
export function resetBacklog() {
  asrStat.pending = 0;
  asrStat.merged = 1;
  showAsr();
}

const ASSEMBLY_LABELS = {
  '': 'Universal-Streaming English',
  'universal-streaming-multilingual': 'Universal-Streaming Multilingual',
  'assemblyai-stt-3-7-realtime': 'STT-3.7 Realtime',
  'assemblyai-stt-3-7-realtime-agent': 'STT-3.7 Realtime-Agent',
};
let whisperStatus = null; // the backend's last 'status' message
let engineState = ''; // of the current run's AssemblyAI stream: connecting | ready | error | ''

// The AssemblyAI stream of a run: connecting / ready / error / closed.
export function onEngine(msg) {
  if (msg.run === state.run) {
    engineState = msg.state === 'closed' ? '' : msg.state;
    showModel();
  }
}

// The engine was changed while not capturing: no stream of the new one yet.
export function clearEngineState() {
  engineState = '';
}

// The status bar's model pill: the Whisper model, or AssemblyAI when that engine is chosen.
export function showModel() {
  const el = $('stModel');
  if (settings.asr.engine === 'assemblyai') {
    const name = ASSEMBLY_LABELS[settings.asr.assemblyModel] || settings.asr.assemblyModel;
    if (engineState === 'connecting') setPill(el, '模型：AssemblyAI 连接中…', 'warn');
    else if (engineState === 'error') setPill(el, '模型：AssemblyAI 连接失败', 'err');
    else setPill(el, `模型：AssemblyAI · ${name}`, engineState === 'ready' ? 'ok' : '');
    return;
  }
  const msg = whisperStatus;
  if (!msg) return;
  if (msg.model === 'ready') setPill(el, `模型：${msg.model_name} · ${msg.device}/${msg.compute_type}`, msg.device === 'cuda' ? 'ok' : 'warn');
  else if (msg.model === 'loading') {
    const dl = msg.download;
    setPill(el, dl ? `模型：下载中 ${Math.floor((dl.done / dl.total) * 100)}%` : '模型：加载中（首次需下载）…', 'warn');
  }
  else if (msg.model === 'error') setPill(el, '模型：加载失败', 'err');
  else setPill(el, '模型：未加载'); // idle: the backend loads Whisper once it is sent this engine
}

export function onStatus(msg) {
  whisperStatus = msg;
  showModel();
  showDownload(msg);
  if (msg.error) showMsg(msg.error);
  showAsrModels(msg);
}

const MODEL_LABELS = {
  'large-v3-turbo': 'large-v3-turbo（快）',
  'large-v3': 'large-v3（更准，更慢，更占显存）',
  'alignatt-large-v3-turbo': 'large-v3-turbo · AlignAtt 同传（实验，实时文字只增不改）',
  'alignatt-large-v3': 'large-v3 · AlignAtt 同传（实验，更准，更慢，更占显存）',
};
const COMPUTE_LABELS = {
  auto: '自动（GPU 用 float16，CPU 用 int8）',
  float16: 'float16（标准）',
  int8_float16: 'int8_float16（省显存，准确度略降）',
  int8: 'int8（省显存，准确度略降）',
};

// Progress of a model download, shown where the messages are (below one that is showing).
function showDownload(msg) {
  const dl = msg.model === 'loading' ? msg.download : null;
  const box = $('dlToast');
  box.classList.toggle('hidden', !dl);
  if (!dl) return;
  const gb = (b) => (b / 1024 ** 3).toFixed(1);
  const speed = dl.rate ? `，${(dl.rate / 1024 ** 2).toFixed(1)}MB/s` : '';
  $('dlText').textContent = `正在下载模型 ${msg.model_name}：${gb(dl.done)}/${gb(dl.total)}GB${speed}`;
  $('dlBar').value = dl.total ? dl.done / dl.total : 0;
  const top = document.querySelector('.topbar').getBoundingClientRect().bottom + 8;
  const toast = $('toast');
  box.style.top = `${toast.classList.contains('hidden') ? top : toast.getBoundingClientRect().bottom + 8}px`;
}

// 设置 → 识别 → 模型: the choices come from the backend, which loads nothing else.
function showAsrModels(msg) {
  if (!Array.isArray(msg.models) || !Array.isArray(msg.compute_types)) return;
  const before = state.asrStatus;
  state.asrStatus = msg;
  // Status messages also arrive twice a second during a download: leave the choices alone
  // (one may be open) unless the lists changed, and only re-read the GPU when the model did.
  const lists = (m) => JSON.stringify([m.models, m.default_model, m.compute_types, m.default_compute]);
  const loaded = (m) => JSON.stringify([m.model, m.model_name, m.compute_type, m.device]);
  if (before && lists(before) === lists(msg)) {
    if (loaded(before) !== loaded(msg)) refreshVram();
    else showVram();
    return;
  }
  const fill = (id, key, all, def, label, defLabel = label) => {
    const others = all.filter((v) => v !== def);
    $(id).replaceChildren(new Option(`默认（${defLabel(def)}）`, ''), ...others.map((v) => new Option(label(v), v)));
    // A saved choice that is now the default, or no longer offered, becomes "默认".
    if (settings.asr[key] && !others.includes(settings.asr[key])) {
      settings.asr[key] = '';
      saveSettings();
    }
    $(id).value = settings.asr[key];
  };
  fill('asrModel', 'model', msg.models, msg.default_model, (m) => MODEL_LABELS[m] || m, (m) => m);
  fill(
    'asrCompute', 'compute', msg.compute_types, msg.default_compute,
    (c) => COMPUTE_LABELS[c] || c,
    (c) => (c === 'auto' ? '自动：GPU 用 float16，CPU 用 int8' : c),
  );
  refreshVram(); // a model was just loaded or is loading: the GPU's memory changed
}

// The line under the model choices, and the pill at the end of the status bar.
export function showVram() {
  const st = state.asrStatus;
  const gpu = state.gpu;
  const hasGpu = Boolean(gpu && gpu.total_mb);
  if (st) {
    const model = settings.asr.model || st.default_model;
    const compute = settings.asr.compute || st.default_compute;
    const est = estimateVramGb(model, compute, st.model === 'ready' && st.device === 'cpu');
    let text = `预计显存占用：${est === null ? '未知' : `${est.toFixed(1)}GB`}`;
    if (hasGpu) text += `，目前显存占用：${vramText(gpu)}`;
    $('asrVram').textContent = `${text}。`;
  }
  const pill = $('stVram');
  if (!hasGpu) {
    pill.classList.add('hidden');
    return;
  }
  const used = gpu.used_mb / gpu.total_mb;
  setPill(pill, `显存占用：${vramText(gpu)}`, used >= VRAM_ERR ? 'err' : used >= VRAM_WARN ? 'warn' : '');
}

// Share of the GPU's memory in use from which the pill turns yellow, then red. Near full,
// Whisper slows down sharply (Windows starts paging VRAM to system RAM) or fails to load.
const VRAM_WARN = 0.8;
const VRAM_ERR = 0.9;
const VRAM_POLL_MS = 5000;

async function refreshVram() {
  const seq = (state.vramSeq = (state.vramSeq || 0) + 1);
  let gpu = null;
  try {
    gpu = await (await fetch('/api/gpu')).json();
  } catch {
    /* backend not reachable: no reading */
  }
  if (seq !== state.vramSeq) return; // a newer call is on its way
  state.gpu = gpu;
  showVram();
}

export function setupStatus() {
  setInterval(() => {
    if (!document.hidden) refreshVram();
  }, VRAM_POLL_MS);
}
