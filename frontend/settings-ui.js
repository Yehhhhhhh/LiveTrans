// The 设置 panel (and the toggles also in the top bar): each control bound to its setting, and what
// changing one does.
import { $, clearTrMsg, setPill, showMsg } from './dom.js';
import { settings, saveSettings, state } from './state.js';
import { clearEngineState, showModel, showVram } from './status.js';
import { sendConfig } from './stream.js';
import { applyDisplay } from './transcript.js';
import { translationRequest, warmupLocal } from './translation.js';
import { getPath, setPath } from './lib/settings.js';
import { postTranslate } from './lib/translate.js';

// Two-way bind a control to a settings path.
function bind(id, path, onChange) {
  const el = $(id);
  const isCheck = el.type === 'checkbox';
  const isNum = el.type === 'number' || el.type === 'range';
  const show = () => (isCheck ? (el.checked = getPath(settings, path)) : (el.value = getPath(settings, path)));
  show();
  el.addEventListener(isCheck || el.tagName === 'SELECT' ? 'change' : 'input', () => {
    let v = isCheck ? el.checked : el.value;
    if (isNum) {
      v = Number(v);
      if (Number.isNaN(v)) return;
    }
    setPath(settings, path, v);
    saveSettings();
    onChange?.(v);
  });
}

function updateProviderFields() {
  const claude = settings.tr.provider === 'claude';
  $('openaiFields').classList.toggle('hidden', claude);
  $('claudeFields').classList.toggle('hidden', !claude);
}

// Toggles shown both in the top bar and in settings.
function bindToggle(ids, path, onChange) {
  const sync = (v) => {
    ids.forEach((id) => ($(id).checked = v));
    onChange?.(v);
  };
  ids.forEach((id) => bind(id, path, sync));
}

// Recognition language: four fixed choices, or "其他…" plus one language from Whisper's list.
const ASR_LANG_MAIN = ['any', 'it', 'en', 'it_en'];
const langName = (() => {
  let names = null;
  try {
    names = new Intl.DisplayNames(['zh-CN'], { type: 'language' });
  } catch {
    /* old browser: codes only */
  }
  return (code) => {
    const n = names?.of(code);
    return n && n !== code ? `${n} (${code})` : code;
  };
})();

function addOtherLangOption(code) {
  const sel = $('asrLangOther');
  if ([...sel.options].some((o) => o.value === code)) return;
  sel.append(new Option(langName(code), code));
}

function showAsrLang() {
  const v = settings.asr.language;
  const main = ASR_LANG_MAIN.includes(v) ? v : 'other';
  $('asrLang').value = main;
  $('asrLangOtherField').classList.toggle('hidden', main !== 'other');
  if (main === 'other') {
    addOtherLangOption(v); // shown even before the list has loaded
    $('asrLangOther').value = v;
  }
}

function setAsrLang(v) {
  settings.asr.language = v;
  if (!ASR_LANG_MAIN.includes(v)) settings.asr.lastOther = v; // offered again next time "其他…" is picked
  saveSettings();
  showAsrLang();
  sendConfig();
}

// The list comes from the backend (Whisper's languages); Italian and English have their own entries.
export async function loadOtherLangs() {
  const sel = $('asrLangOther');
  if (sel.dataset.loaded) return;
  try {
    const { languages } = await (await fetch('/api/languages')).json();
    const keep = settings.asr.language;
    sel.textContent = '';
    languages
      .filter((c) => c !== 'it' && c !== 'en')
      .map((c) => [langName(c), c])
      .sort((a, b) => a[0].localeCompare(b[0], 'zh-CN'))
      .forEach(([name, c]) => sel.append(new Option(name, c)));
    sel.dataset.loaded = '1';
    if (!ASR_LANG_MAIN.includes(keep)) addOtherLangOption(keep);
    showAsrLang();
  } catch {
    /* backend not up yet: retried when the WebSocket connects */
  }
}

const onModelChoice = () => {
  sendConfig();
  showVram();
};
// Settings that only the local Whisper engine uses are hidden while AssemblyAI is chosen.
function showEngineFields() {
  const cloud = settings.asr.engine === 'assemblyai';
  $('assemblyFields').classList.toggle('hidden', !cloud);
  document.querySelectorAll('.whisperOnly').forEach((el) => el.classList.toggle('hidden', cloud));
}
const onEngineChoice = () => {
  showEngineFields();
  // The pill shows the state of the running capture's engine, which stays as it is until 开始.
  if (state.running) showMsg('识别引擎的改动在下一次「开始」生效');
  else clearEngineState();
  showModel();
  sendConfig();
};

const showGain = () => {
  $('micGainV').textContent = `${settings.audio.micGain}×`;
  $('sysGainV').textContent = `${settings.audio.sysGain}×`;
};

export function setupSettings() {
  $('settingsBtn').onclick = () => $('settings').classList.toggle('hidden');
  $('closeSettings').onclick = () => $('settings').classList.add('hidden');

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      document.querySelectorAll('.pane').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== tab.dataset.tab));
    };
  });

  bind('trEnabled', 'tr.enabled');
  bind('trProvider', 'tr.provider', updateProviderFields);
  bind('oaBase', 'tr.openai.baseUrl');
  bind('oaKey', 'tr.openai.apiKey');
  bind('oaModel', 'tr.openai.model');
  bind('clKey', 'tr.claude.apiKey');
  bind('clModel', 'tr.claude.model');
  bind('clEffort', 'tr.claude.effort');
  bind('trContext', 'tr.contextSize');
  bind('trDomain', 'tr.domain');
  bind('locBase', 'tr.local.baseUrl');
  bind('locModel', 'tr.local.model');
  updateProviderFields();

  bindToggle(['trLive', 'tbLive'], 'tr.live');
  bind('locCpu', 'tr.local.cpu', () => {
    if (settings.tr.useLocal) warmupLocal(true);
  });
  bindToggle(['trLocal', 'tbLocal'], 'tr.useLocal', (on) => {
    // Reloaded: the CPU setting may have been changed while the local model was off, and the
    // model may still be loaded where the old setting put it.
    if (on) return warmupLocal(true);
    clearTrMsg();
    setPill($('stTr'), '翻译：-');
  });
  if (settings.tr.useLocal) warmupLocal();

  $('asrLang').addEventListener('change', () => {
    const v = $('asrLang').value;
    setAsrLang(v === 'other' ? settings.asr.lastOther || 'fr' : v);
  });
  $('asrLangOther').addEventListener('change', () => setAsrLang($('asrLangOther').value));
  showAsrLang();
  loadOtherLangs();
  bind('asrEngine', 'asr.engine', onEngineChoice);
  bind('aaiKey', 'asr.assemblyKey', sendConfig);
  bind('aaiModel', 'asr.assemblyModel', onEngineChoice);
  showEngineFields();
  showModel();
  bind('asrModel', 'asr.model', onModelChoice);
  bind('asrCompute', 'asr.compute', onModelChoice);
  bind('vadThr', 'asr.vadThreshold', sendConfig);
  bind('vadSil', 'asr.minSilenceMs', sendConfig);
  bind('vadMax', 'asr.maxSegmentS', sendConfig);
  bind('asrGlossary', 'asr.glossaryPrompt', sendConfig);
  bind('asrHint', 'asr.hint', sendConfig);
  bind('asrContext', 'asr.context', sendConfig);
  bind('asrPartial', 'asr.partial', sendConfig);

  bind('micGain', 'audio.micGain', showGain);
  bind('sysGain', 'audio.sysGain', showGain);
  bind('micProc', 'audio.micProcessing');
  showGain();

  bind('showOrig', 'display.showOrig', applyDisplay);
  bind('autoScroll', 'display.autoScroll');
  bind('origSize', 'display.origSize', applyDisplay);
  bind('transSize', 'display.transSize', applyDisplay);

  document.querySelectorAll('[data-font]').forEach((btn) => {
    btn.onclick = () => {
      const key = btn.dataset.font === 'orig' ? 'origSize' : 'transSize';
      settings.display[key] = Math.max(10, Math.min(64, settings.display[key] + Number(btn.dataset.delta)));
      $(key).value = settings.display[key];
      saveSettings();
      applyDisplay();
    };
  });

  $('trTest').onclick = async () => {
    const out = $('trTestOut');
    out.textContent = '测试中…';
    const t0 = performance.now();
    let first = 0;
    try {
      const r = await postTranslate(
        translationRequest('Allora, nel sistema OFDM ogni subcarrier is modulated independently, quindi il cyclic prefix elimina la ISI.', []),
        (text) => {
          if (!first) first = performance.now() - t0;
          out.textContent = `… ${text}`;
        },
      );
      const total = Math.round(performance.now() - t0);
      out.textContent = `✓ 首字 ${Math.round(first || total)} ms · 共 ${total} ms：${r}`;
    } catch (e) {
      out.textContent = `✗ ${e.message}`;
    }
  };
}
