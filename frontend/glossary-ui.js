// The 术语表 tab: the glossary as an editable table, with search, 添加, 导入, 导出 and 恢复默认.
import { $, download } from './dom.js';
import { glossary, saveGlossary, resetGlossary } from './state.js';
import { sendConfig } from './stream.js';
import { mergeGlossary, parseGlossary } from './lib/glossary.js';
import { debounce } from './lib/util.js';

export function renderGlossary() {
  const q = $('glSearch').value.trim().toLowerCase();
  const body = $('glBody');
  body.innerHTML = '';
  let shown = 0;
  // Newest first so freshly added rows appear at the top.
  for (let i = glossary.length - 1; i >= 0; i--) {
    const g = glossary[i];
    if (q && ![g.it, g.en, g.zh].some((t) => (t || '').toLowerCase().includes(q))) continue;
    shown++;
    const tr = document.createElement('tr');
    for (const k of ['it', 'en', 'zh']) {
      const td = document.createElement('td');
      const inp = document.createElement('input');
      inp.value = g[k] || '';
      inp.oninput = () => {
        g[k] = inp.value;
        saveGlossaryAndSync();
      };
      td.appendChild(inp);
      tr.appendChild(td);
    }
    const td = document.createElement('td');
    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '✕';
    del.title = '删除';
    del.onclick = () => {
      glossary.splice(glossary.indexOf(g), 1);
      saveGlossaryAndSync();
      renderGlossary();
    };
    td.appendChild(del);
    tr.appendChild(td);
    body.appendChild(tr);
  }
  $('glCount').textContent = `共 ${glossary.length} 条${q ? `，匹配 ${shown} 条` : ''}`;
}

const saveGlossaryAndSync = debounce(() => {
  saveGlossary();
  sendConfig();
}, 400);

export function setupGlossary() {
  $('glSearch').oninput = renderGlossary;
  $('glAdd').onclick = () => {
    glossary.push({ it: '', en: '', zh: '' });
    $('glSearch').value = '';
    saveGlossaryAndSync();
    renderGlossary();
    $('glBody').querySelector('input')?.focus();
  };
  $('glExport').onclick = () => download('glossary.json', JSON.stringify(glossary, null, 1), 'application/json');
  $('glReset').onclick = async () => {
    if (!confirm('恢复默认术语表？你添加或修改的条目会丢失。')) return;
    await resetGlossary();
    sendConfig();
    renderGlossary();
  };
  $('glImport').onclick = () => $('glFile').click();
  $('glFile').onchange = async () => {
    const file = $('glFile').files[0];
    $('glFile').value = '';
    if (!file) return;
    const text = await file.text();
    let items;
    try {
      items = parseGlossary(file.name, text);
    } catch (e) {
      alert(`导入失败：${e.message}`);
      return;
    }
    const added = mergeGlossary(glossary, items);
    saveGlossaryAndSync();
    renderGlossary();
    alert(`导入 ${items.length} 条，其中新增 ${added} 条`);
  };
}
