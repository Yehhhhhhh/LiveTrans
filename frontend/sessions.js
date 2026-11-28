// Course records (课程), kept in IndexedDB: the list, opening one, 新建 / 重命名 / 删除, 导出 MD.
import * as db from './db.js';
import { $, download } from './dom.js';
import { settings, saveSettings, state } from './state.js';
import { renderAll } from './transcript.js';
import { markdownFilename, sessionMarkdown } from './lib/markdown.js';
import { fmtDateTime } from './lib/util.js';

async function refreshSessionList() {
  const sessions = await db.listSessions();
  const sel = $('sessionSel');
  sel.innerHTML = '';
  for (const s of sessions) {
    const o = document.createElement('option');
    o.value = s.id;
    o.textContent = s.title;
    sel.appendChild(o);
  }
  if (state.session) sel.value = state.session.id;
  return sessions;
}

async function createSession(title) {
  const now = Date.now();
  const s = { id: crypto.randomUUID(), title: title || `课程 ${fmtDateTime(now)}`, createdAt: now };
  await db.putSession(s);
  await openSession(s);
}

async function openSession(s) {
  state.session = s;
  settings.currentSessionId = s.id;
  saveSettings();
  state.segments = await db.getSegments(s.id);
  await refreshSessionList();
  renderAll();
}

// Opens the course record that was open last (or the newest), or starts one when there is none.
export async function initSessions() {
  const sessions = await db.listSessions();
  const cur = sessions.find((s) => s.id === settings.currentSessionId) || sessions[0];
  if (cur) await openSession(cur);
  else await createSession();
}

export function setupSessions() {
  $('sessionSel').onchange = async () => {
    const sessions = await db.listSessions();
    const s = sessions.find((x) => x.id === $('sessionSel').value);
    if (s) await openSession(s);
  };

  $('newSessionBtn').onclick = async () => {
    const title = prompt('课程名称', `课程 ${fmtDateTime(Date.now())}`);
    if (title !== null) await createSession(title.trim() || undefined);
  };

  $('renameSessionBtn').onclick = async () => {
    if (!state.session) return;
    const title = prompt('重命名', state.session.title);
    if (!title || !title.trim()) return;
    state.session.title = title.trim();
    await db.putSession(state.session);
    await refreshSessionList();
  };

  $('deleteSessionBtn').onclick = async () => {
    if (!state.session) return;
    if (!confirm(`删除「${state.session.title}」及其全部 ${state.segments.length} 条记录？此操作不可恢复。`)) return;
    await db.deleteSession(state.session.id);
    state.session = null;
    await initSessions();
  };

  $('exportBtn').onclick = () => {
    const s = state.session;
    if (!s) return;
    download(markdownFilename(s.title), sessionMarkdown(s, state.segments), 'text/markdown;charset=utf-8');
  };
}
