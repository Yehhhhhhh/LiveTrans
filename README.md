# LiveTrans 同传字幕

课堂实时字幕：意大利语 / 英语混合语音 → 本地 Whisper 识别 → 云端 API 或本地 Ollama 模型翻译成中文，在 Chrome 中显示。

## 特性

- 本地 GPU 运行 faster-whisper（`large-v3-turbo`），默认在意大利语 / 英语间逐段自动识别，也可自动识别所有语言或固定为任一种语言
- Silero VAD 流式断句，说话时逐词实时显示，句末替换为精确结果；老师声音小、离得远也能检测到
- 自动过滤 Whisper 幻觉（重复循环、「Grazie a tutti」之类的套话、复读术语表提示）
- 翻译支持 OpenAI 兼容接口、Claude，以及本机 Ollama 本地模型（不经过云端），带上下文与术语表，流式输出
- 译文中的数学公式用 KaTeX 渲染（离线可用）
- 音源支持麦克风、系统音频、某个窗口或标签页，无需虚拟声卡；识别中可随时换窗口 / 换音源，不中断
- 状态栏显示当前共享的是哪个程序（含图标和窗口标题）
- 记录自动保存在浏览器，可导出 Markdown
- 术语表默认为空，可在「设置 → 术语表」中自行添加，或导入 JSON / CSV

## 环境要求

- Windows 10 / 11
- Python 3.12
- NVIDIA 显卡 + 驱动（无显卡可用 CPU，但很慢）
- Chrome / Edge 浏览器
- 翻译二选一：云端 API（OpenAI 兼容接口或 Claude），或本机安装 [Ollama](https://ollama.com/download) 使用本地模型

## 安装与启动

1. 双击 `install.bat`：创建 `.venv` 并安装依赖（含 CUDA/cuDNN 运行库，不需要单独装 CUDA，只需 NVIDIA 显卡驱动）。安装的是 `backend/requirements.lock` 中锁定的、测试过的版本，过一段时间重新安装也不会装到不兼容的新版本；锁定版本装不上时（例如 Python 版本不同）会自动改用 `requirements.txt` 中的版本范围。
2. 双击 `start.bat`：启动后端，等后端就绪后自动打开 http://127.0.0.1:8765 。
   首次启动会下载 Whisper `large-v3-turbo` 模型（约 1.6GB），状态栏显示「模型：large-v3-turbo · cuda/float16」即就绪。
   下载困难时，编辑 `start.bat`，取消 `HF_ENDPOINT` 那行的注释使用镜像。
3. 在「设置 → 翻译」中填写 API，或开启「本地模型」（见下文），点「测试翻译」确认可用。

更新代码后（`git pull`）需要关掉 `start.bat` 窗口重新启动，后端的改动才会生效；只改了页面的话刷新浏览器即可。如果 `backend/requirements.lock` 有变化，先重新运行一次 `install.bat`。

升级依赖：在 `.venv` 中升级并测试通过后，运行 `.venv\Scripts\python backend\tools\freeze_lock.py` 重新生成 `requirements.lock`（只包含 `requirements.txt` 及其依赖，不含 pytest 等开发工具）。

## 音源

| 选项 | 用途 | 操作 |
|---|---|---|
| 麦克风 | 现场上课 | 选择麦克风设备即可 |
| 系统 / 标签页音频 | Teams 桌面版、浏览器里的网课 | 共享弹窗中选择要采集的**窗口**（只采集该程序的声音）、**标签页**或**整个屏幕**（采集全部系统声音），并勾选「共享音频」 |
| 麦克风 + 系统音频 | 两路混合 | 同上，两者都会采集 |

不需要装虚拟声卡。现场上课时，离讲台越近、麦克风越好，识别越准；声音小可在「设置 → 音频」调高麦克风增益。

### 识别中切换音源（不中断）

- 识别过程中，音源下拉框旁会出现「**换窗口**」按钮：点击后重新弹出共享窗口，选好新的窗口 / 标签页 / 屏幕，识别直接接着进行，正在说的句子和记录都不受影响。
- 识别中修改「音源」下拉框或更换麦克风设备，同样不会中断；从「麦克风 + 系统音频」改成其中一种时，保留的那一路不会重新弹窗。
- 在共享弹窗里点了取消，会保留原来的音源。
- 共享标签页时，Chrome 共享提示栏里的「改为共享此标签页」也能直接切换标签页，不用弹窗。
- 「麦克风 + 系统音频」模式下，在 Chrome 提示栏点「停止共享」只停掉系统音频，麦克风继续识别；按钮变成「共享窗口」，可以重新共享。

切换窗口必须经过浏览器的共享弹窗，这是浏览器的安全限制，网页无法自行静默切换。

### 当前共享的内容

底部状态栏会显示当前共享的是什么：

- **窗口**：显示程序图标、程序名和窗口标题，例如「Microsoft Teams · 会议标题」，鼠标悬停可看到完整名称和 exe 文件名。
  浏览器只告诉网页窗口句柄，名称和图标由本机后端通过 Windows API 查出（`/api/window`）。读不到时显示「🪟 窗口」，不影响识别。
- **整个屏幕**：显示「🖥 整个屏幕」。
- **标签页**：显示「🌐 浏览器标签页」（浏览器不提供标签页标题）。

少数商店应用（如 Windows「设置」）外面套了一层系统宿主进程，会显示为「Application Frame Host」和通用图标。

## 翻译

- **OpenAI 兼容接口**：OpenAI、DeepSeek、通义千问、Ollama / LM Studio 等任何 `/chat/completions` 接口。
- **Claude**：默认 `claude-opus-5-5`，effort 设为 `low` 以降低延迟；想更快更省可改为 `claude-haiku-5-5`。
- **本地模型（Ollama）**：见下一节，开启后忽略上面的服务商。
- 译文流式显示，边生成边出现。
- **提前翻译**（顶栏开关）：说话过程中就翻译实时识别出的半句话，译文更早出现；需开启「逐词实时显示」，API 调用次数会增加数倍。
- **课程领域**（「设置 → 翻译」，默认留空）：写进翻译提示词，告诉模型这是什么课，帮助理解专业内容；填写相应领域（如「通信工程」「机器学习」「经济学」）即可，留空则不指定。
- 每段翻译会带上前 N 段（默认 3）的原文和译文作上下文，并附上该段命中的术语表条目。
- 使用云端 API 时，最多 2 段同时翻译，某一段回复慢不会拖住后面的段落；每段的译文一完成就显示在自己那一行。前面还没译完的段落只带原文作上下文。本地模型和 Whisper 共用显卡，仍然一次翻译一段。
- 译文里的数学公式（`\(..\)`、`\[..\]`、`$..$`、`$$..$$`）用 KaTeX 渲染，KaTeX 已打包在 `frontend/vendor/` 中，离线也能用。
- 翻译失败的段落可点「翻译」重试，已翻译的可点「重译」。

## 本地翻译（Ollama）

翻译完全在本机运行，不需要 API Key，也不经过云端。

1. 安装 [Ollama](https://ollama.com/download) 并保持运行。
2. 拉取模型（默认推荐，约 2.5GB，显存占用约 3GB）：
   ```
   ollama pull qwen3:4b-instruct-2507-q4_K_M
   ```
3. 在顶栏勾选「本地模型」，或在「设置 → 翻译」中勾选「使用本地模型」。
   默认地址 `http://127.0.0.1:11434`，模型名可改成任何已拉取的 Ollama 模型。
4. 开启时会自动预加载模型，状态栏显示「翻译：本地模型已就绪」即可使用。

说明：

- 模型常驻显存 1 小时（`keep_alive`），避免第一句翻译变慢；上下文窗口为 4096，给 Whisper 留出显存。
- 显存要和 Whisper 共用：`large-v3-turbo` 约占 4GB，8GB 显卡上本地模型建议控制在 3.5GB 以内，否则会互相抢显存变慢。
- 自动关闭模型的思考模式；推理模型输出的 `<think>` 内容会被过滤掉。
- 本地 4B 模型翻译质量不如云端大模型，专业术语主要靠术语表纠正。
- 常见错误：「无法连接 Ollama」→ 确认 Ollama 已启动；「本地没有模型 …」→ 先执行上面的 `ollama pull`。翻译恢复正常或改用 API 后，错误提示会自动消失。

## 识别参数（设置 → 识别）

- **识别语言**：
  - **意大利语 / 英语**（默认）：每段只在这两种语言之间判断，混用两种语言的课用这个。
  - **意大利语**、**英语**：整节课只用一种语言时固定下来，更准，也更快（省掉每段的语种判断）。
  - **自动（所有语言）**：在 Whisper 支持的约 100 种语言中判断。范围大，口音重或很短的句子更容易判断错，只在不确定讲什么语言时使用。
  - **其他…**：从列表中选一种语言固定识别（语言名称由浏览器显示为中文）。
  - 以前保存的「自动（意大利语 / 英语）」设置会继续按意大利语 / 英语识别。术语表和幻觉过滤主要针对意大利语 / 英语，其他语言下作用有限。
- **语音判定阈值**（默认 0.5）：人声概率高于它才开始识别。状态栏会实时显示「人声 0.xx / 阈值 0.xx」，老师说话时人声值总达不到阈值，就把阈值调低一些。
- **句末静音** 600ms、**单段最长** 8 秒：超过 4.8 秒后遇到 200ms 停顿即切分，到 8 秒强制切分。
- **说话时逐词实时显示**：说话过程中每约 0.7 秒做一次快速识别，单词逐个出现在虚线框里；句子结束后替换为精确结果再翻译。显卡紧张时可关掉。
- **用术语表提示识别**：把术语表中的词作为 Whisper 提示，提高专业词识别率。
- **识别提示语**（默认留空）：用讲课的语言写的一句课程名称，放在术语前面一起提示 Whisper；填写后会一起发送。Whisper 偶尔会把提示语原样「念」出来，这类输出会被自动过滤。

识别质量相关的自动处理：

- VAD 看到的音频会自动放大音量（最多 20 倍），轻声、远距离的讲课也能检测到；送给 Whisper 的仍是原始音频。
- 浏览器自带的麦克风降噪 / 自动增益默认关闭（它会把远处的人声当成噪音去掉），可在「设置 → 音频」中开启。
- 长时间上课时定期重置 VAD 状态，避免越到后面越检测不到人声。
- 识别跟不上说话速度时（显卡慢、显存被其他程序占用或用 CPU），状态栏「识别」会变黄，并显示「待识别 N 段」。正在识别的那段后面又排了 2 段或以上时，排队的几段会合并成一次识别来追赶（最长 24 秒），追上后黄色提示自动消失。Whisper 每次识别的耗时主要是固定开销，实测合并 3 段只比识别 1 段多花约三成时间。自动语种模式下，合并的几段共用一个语种。
- 模型还在加载（首次启动要下载）时就点了「开始」，说的话也不会丢：语音先排队，模型就绪后再一起识别。
- 过滤 Whisper 幻觉：连续重复的词句会合并，只有套话（如「Grazie a tutti」「Thank you」）、只在复读术语表提示、或可信度过低的段落会被丢弃。

## 记录

- 所有记录自动保存在浏览器 IndexedDB 中，刷新或关闭页面不会丢失（同一浏览器、同一地址 `127.0.0.1:8765`）。
- 每节课可「新建」一条记录，「导出 MD」导出为 Markdown。
- **断线续传**：识别过程中后端断开（重启了 `start.bat`、后端崩溃等）时，页面继续采集并保留音频，重新连上后把后端还没处理完的语音补发过去，断线前后说的话照常出字幕，时间也接得上，已经出来的句子不会重复。最多保留 2 分钟；断线更久时，更早的部分无法补回，页面会提示丢了多少秒。刷新或关闭页面不在此列（页面里保留的音频会随之丢失）。

## 术语表

术语表默认为空（「恢复默认」会清空），可在「设置 → 术语表」中搜索、编辑、新增、删除，支持导入 JSON（`[{"it","en","zh"}]`）或 CSV（`意大利语,英语,中文` 每行一条）。

## 提示信息

错误和提示会在页面上方正中弹出：一般提示 8 秒后自动消失，翻译后端的错误（如 Ollama 未启动）会一直显示到问题解决；都可以点 × 关闭。

## 环境变量（可写入 start.bat）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LIVETRANS_MODEL` | `large-v3-turbo` | 可改为 `large-v3`（更准更慢）、`medium` 等 |
| `LIVETRANS_DEVICE` | `auto` | `cuda` / `cpu` |
| `LIVETRANS_COMPUTE_TYPE` | `auto` | Whisper 计算精度，如 `float16` / `int8_float16` / `int8` |
| `LIVETRANS_HOST` | `127.0.0.1` | 监听地址 |
| `LIVETRANS_PORT` | `8765` | 端口 |
| `LIVETRANS_ALLOWED_HOSTS` | - | 除 `localhost` 和 IP 地址外，还允许用哪些主机名打开页面，逗号分隔（例如 `LIVETRANS_HOST=0.0.0.0` 时本机的网络名） |

后端只接受来自自己页面的请求：浏览器里打开的其他网站不能连接 `/ws` 或调用 API（会被拒绝并在日志中记录），命令行工具、脚本不受影响。用 IP 地址或 `localhost` 打开页面都可以；用其他主机名打开时，需要把它加到 `LIVETRANS_ALLOWED_HOSTS`。

注意：设置 `LIVETRANS_HOST=0.0.0.0` 让局域网访问时，同一网络中的任何人都可以使用这个后端，包括用它转发翻译请求，请只在可信网络中这样做。
| `LIVETRANS_RECORD` | - | 设为 `1` 时把每次采集的原始 16kHz 音频保存到 `recordings/`，用于调参 |
| `HF_ENDPOINT` | - | HuggingFace 镜像 |

## 测试

```
.venv\Scripts\python -m pip install -r backend\requirements-dev.txt
.venv\Scripts\python -m pytest backend\tests
```

## 结构

```
backend/requirements.txt  Python 依赖（版本范围）
backend/requirements.lock 测试过的精确版本，install.bat 用它安装
backend/tools/         freeze_lock.py：由当前环境生成 requirements.lock
backend/server.py      FastAPI：/ws 音频流、/api/translate 翻译代理、/api/window 窗口信息、静态页面
backend/vad.py         Silero VAD 流式断句
backend/asr.py         faster-whisper 识别（识别语言模式）、幻觉过滤
backend/translate.py   OpenAI 兼容接口 / Claude / Ollama 本地模型
backend/winfo.py       由窗口句柄查程序名、窗口标题和图标（Windows API）
backend/tests/         pytest 单元测试（依赖见 requirements-dev.txt）
frontend/              页面、音频采集（AudioWorklet 16kHz）、IndexedDB 记录
frontend/vendor/katex/ KaTeX 公式渲染（离线）
install.bat            创建 .venv 并安装依赖
start.bat              启动后端，就绪后打开页面
```

## 持续集成（GitHub Actions）

Pull Request 上有两个工作流：CI 在每次创建或更新时都运行；Codex PR Review 只在创建时自动运行一次，之后手动触发。

- **CI**（`.github/workflows/ci.yml`）
  - Lint：`python .github/scripts/lint.py` —— Python 语法、前端 JS 语法（`node --check`）、`.bat` 必须是纯 ASCII + CRLF。
  - Build & Test（Windows、Python 3.12）：按 install.bat 的方式安装依赖，检查 `import server`，有 `backend/tests` 时运行 pytest。
  - 项目没有配置类型检查器，所以不做 Type Check。
  - `CI result` 汇总上面所有任务，分支保护只需把它设为必需检查。
- **Codex PR Review**（`.github/workflows/codex-review.yml`）：用 OpenAI Codex 只读审查 PR 的改动，结果发在 PR 评论里（始终更新同一条评论）。
  - PR **创建时**自动审查一次；之后的推送（包括 Claude Code / Codex 的推送）只跑 CI，不再自动审查，避免“推送 → 审查 → 修改 → 推送”无限循环。
  - 需要重新审查时：Actions → Codex PR Review → Run workflow，填 PR 编号。会审查该 PR 最新的提交，并逐条核对上一次审查的问题是否已修复。
  - 没有阻止合并的问题时，结论为 `NO_BLOCKING_ISSUES`。审查要求在 `.github/codex/review-prompt.md`，以目标分支上的版本为准。
  - 来自 fork 的 PR 不审查。需要仓库 Secret `OPENAI_API_KEY`。

本地跑与 CI 相同的检查（提交前 Claude Code / Codex 也用这两条）：

```
.venv\Scripts\python .github\scripts\lint.py
.venv\Scripts\python -m pytest backend\tests
```
