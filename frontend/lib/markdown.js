// 导出 MD: a course record as a Markdown file.
import { fmtDateTime, fmtTime } from './util.js';

export function sessionMarkdown(s, segments) {
  const lines = [`# ${s.title}`, '', `- 创建时间：${fmtDateTime(s.createdAt)}`, `- 段数：${segments.length}`, '', '---', ''];
  for (const seg of segments) {
    lines.push(`**[${fmtTime(seg.ts)}] ${(seg.lang || '').toUpperCase()}**  `);
    lines.push(seg.text);
    lines.push('');
    if (seg.translation) {
      lines.push(`> ${seg.translation}`);
      lines.push('');
    }
  }
  return lines.join('\n');
}

// The course title, with the characters Windows doesn't allow in file names replaced.
export const markdownFilename = (title) => `${title.replace(/[\\/:*?"<>|]/g, '_')}.md`;
