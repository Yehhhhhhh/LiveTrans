// DOM helpers shared by the page's modules: element lookup, status-bar pills, the message toast
// and file downloads.

export const $ = (id) => document.getElementById(id);

export function setPill(el, text, cls = '') {
  el.textContent = text;
  el.className = 'pill' + (cls ? ' ' + cls : '');
}

// Messages pop up as a toast at the top centre. They fade after a while, except the
// translation-backend problem (trMsg), which stays until it is fixed or dismissed.
const TOAST_MS = 8000;
let toastTimer = 0;
export function showMsg(text) {
  $('stMsg').textContent = text || '';
  const toast = $('toast');
  toast.classList.toggle('hidden', !text);
  // Just below the top bar, which wraps to more rows on narrow windows.
  if (text) toast.style.top = `${document.querySelector('.topbar').getBoundingClientRect().bottom + 8}px`;
  clearTimeout(toastTimer);
  if (text && text !== trMsg) toastTimer = setTimeout(() => showMsg(''), TOAST_MS);
}

export function setupToast() {
  $('toastClose').onclick = () => showMsg('');
}

// A translation-backend problem (e.g. Ollama not running) is shown until the
// backend works again or another backend is chosen; other messages are kept.
let trMsg = '';
export function showTrMsg(text) {
  trMsg = text;
  showMsg(text);
}
export function clearTrMsg() {
  if (trMsg && $('stMsg').textContent === trMsg) showMsg('');
  trMsg = '';
}

export function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
