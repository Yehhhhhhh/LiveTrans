// Settings: the defaults, saved settings read over them, and access by dotted path.
// No DOM or storage here (state.js loads and saves them), so the unit tests run it in Node.

export const DEFAULTS = {
  source: 'mic',
  micDeviceId: '',
  // Browser noise suppression treats a distant, soft-spoken lecturer as noise, so it is off by default.
  audio: { micGain: 1, sysGain: 1, micProcessing: false },
  asr: {
    // Recognition engine: 'assemblyai' (cloud, streaming) or 'local' (Whisper on this PC, which
    // the backend loads only once this is chosen), with AssemblyAI's API key and model ('' is
    // the backend's default, Universal-Streaming English).
    engine: 'assemblyai',
    assemblyKey: '',
    assemblyModel: '',
    language: 'it_en',
    lastOther: 'fr',
    // Whisper model and compute type (weight precision), from those the backend offers;
    // '' is its default (LIVETRANS_MODEL, LIVETRANS_COMPUTE_TYPE).
    model: '',
    compute: '',
    vadThreshold: 0.5,
    minSilenceMs: 600,
    maxSegmentS: 8,
    glossaryPrompt: true,
    // Give Whisper the end of the previous sentence as context for the next one.
    context: true,
    // Course name in the lecture's language, put before the glossary terms in the Whisper prompt.
    hint: '',
    partial: true,
  },
  tr: {
    enabled: true,
    live: true,
    useLocal: false, // overrides provider with the local Ollama model
    domain: '', // course domain named in the translation prompt; '' = unspecified
    // cpu: run the model on the CPU, so it doesn't share the GPU with Whisper.
    local: { baseUrl: 'http://127.0.0.1:11434', model: 'qwen3:4b-instruct-2507-q4_K_M', cpu: false },
    provider: 'openai',
    contextSize: 3,
    openai: { baseUrl: 'https://api.openai.com/v1', apiKey: '', model: '' },
    claude: { apiKey: '', model: 'claude-opus-5-5', effort: 'low' },
  },
  display: { origSize: 16, transSize: 22, showOrig: true, autoScroll: true },
  currentSessionId: '',
};

// A copy of `base` with the values of `over` (saved settings) put in: objects are merged key by
// key, and keys `base` doesn't have (settings that no longer exist) are left out.
export function deepMerge(base, over) {
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(over || {})) {
    if (v && typeof v === 'object' && !Array.isArray(v) && typeof out[k] === 'object') {
      out[k] = deepMerge(out[k], v);
    } else if (k in out) {
      out[k] = v;
    }
  }
  return out;
}

// Settings saved by an earlier version, read as what they meant then.
export function migrateSettings(settings) {
  // Before "自动（所有语言）" existed, "auto" meant Italian / English: keep what it did.
  if (settings.asr.language === 'auto') settings.asr.language = 'it_en';
  // Universal-3.6 Pro is no longer offered: AssemblyAI names STT-3.7 Realtime-Agent as its successor.
  if (settings.asr.assemblyModel === 'universal-3-6-pro') settings.asr.assemblyModel = 'assemblyai-stt-3-7-realtime-agent';
}

// A setting by its dotted path, e.g. 'tr.openai.model'.
export function getPath(obj, path) {
  return path.split('.').reduce((o, k) => o[k], obj);
}
export function setPath(obj, path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  keys.reduce((o, k) => o[k], obj)[last] = value;
}
