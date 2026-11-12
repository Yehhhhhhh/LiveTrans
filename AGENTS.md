# AGENTS.md

本文件是给在本仓库工作的 AI 编码代理（Claude Code、Codex 等）的规则，人工贡献者也适用。

## 处理 Codex PR 审查意见

Codex 审查结果发布在 PR 的评论里（标题为 “Codex review of <提交>”），每条问题都标有严重程度。

| 严重程度 | 怎么处理 |
|---|---|
| **Critical** | 必须修复。 |
| **High** | 必须修复。 |
| **Medium** | 非必要不修。默认不改代码，在汇报中列出并说明理由，由用户决定是否修复。 |

- 审查结论为 `NO_BLOCKING_ISSUES` 时，不要再为了审查改代码。
- 不要修复审查没有指出的“顺手问题”，不要因为审查意见去做重构或风格调整。
- 认为某条 Critical/High 判断有误时，不要硬改，向用户说明依据，由用户决定。
- 修复后照常跑本地检查（见 README 的“持续集成”一节）：
  - `.venv\Scripts\python .github\scripts\lint.py`
  - `.venv\Scripts\python -m pytest backend\tests`

## 重新审查的次数上限

PR 创建时 Codex 会自动审查一次（不计入下面的轮数）。之后的推送不会自动触发审查，需要手动重新审查：

```
gh workflow run codex-review.yml -f pr_number=<PR 编号>
```

- **同一个 PR 最多手动重新审查 2 轮。**
- 触发前先数已经进行过的轮数：
  ```
  gh run list --workflow codex-review.yml --event workflow_dispatch --limit 100 --json displayTitle --jq '[.[] | select(.displayTitle == "Codex review of PR #<PR 编号>")] | length'
  ```
- 2 轮用完后，不要再触发。把仍然存在的问题和你的判断汇报给用户，是否继续修改、再审查或合并由用户决定。
- 只有在修复了 Critical/High 问题之后才需要重新审查；只改了文档或 Medium 问题时不需要。

## 不允许的操作

- 未经用户明确同意，不要合并 PR，不要删除分支。
- 除非用户明确要求，不要修改 `.github/codex/review-prompt.md` 或 `.github/workflows/`，也不要关闭、绕过 Codex 审查或 CI。
- `main` 有分支保护：所有改动都通过 PR 合并，并且 `CI result` 必须通过。
