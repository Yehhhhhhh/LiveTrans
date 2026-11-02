# LiveTrans 同传字幕

课堂实时字幕：意大利语 / 英语混合语音 → 本地 Whisper 识别 → API 翻译成中文，在 Chrome 中显示。

## 特性

- 本地 GPU 运行 faster-whisper（`large-v3-turbo`），意大利语 / 英语逐段自动识别
- Silero VAD 流式断句，说话时逐词实时显示，句末替换为精确结果
- 翻译支持 OpenAI 兼容接口与 Claude，带上下文与术语表
- 音源支持麦克风、系统 / 标签页音频，无需虚拟声卡
- 记录自动保存在浏览器，可导出 Markdown
- 内置 225 条通信领域意 / 英 / 中术语表

## 环境要求

- Windows 10 / 11
- Python 3.12
- NVIDIA 显卡 + 驱动（无显卡可用 CPU，但很慢）
- Chrome / Edge 浏览器
- 一个翻译 API（OpenAI 兼容接口或 Claude）

## 安装与启动

1. 双击 `install.bat`：创建 `.venv` 并安装依赖（含 CUDA/cuDNN 运行库，不需要单独装 CUDA，只需 NVIDIA 显卡驱动）。
2. 双击 `start.bat`：启动后端并打开 http://127.0.0.1:8765 。
   首次启动会下载 Whisper `large-v3-turbo` 模型（约 1.6GB），状态栏显示「模型：large-v3-turbo · cuda/float16」即就绪。
   下载困难时，编辑 `start.bat`，取消 `HF_ENDPOINT` 那行的注释使用镜像。
3. 在「设置 → 翻译」中填写 API，点「测试翻译」确认可用。

## 音源

| 选项 | 用途 | 操作 |
|---|---|---|
| 麦克风 | 现场上课 | 选择麦克风设备即可 |
| 系统 / 标签页音频 | Teams 桌面版、浏览器里的网课 | 弹窗中选「整个屏幕」并勾选「共享系统音频」；网页版可直接选该标签页并勾选「共享标签页音频」 |
| 麦克风 + 系统音频 | 两路混合 | 同上，两者都会采集 |

不需要装虚拟声卡。现场上课时，离讲台越近、麦克风越好，识别越准；声音小可在「设置 → 音频」调高麦克风增益。

## 翻译

- **OpenAI 兼容接口**：OpenAI、DeepSeek、通义千问、Ollama / LM Studio 等任何 `/chat/completions` 接口。
- **Claude**：默认 `claude-opus-5-5`，effort 设为 `low` 以降低延迟；想更快更省可改为 `claude-haiku-5-5`。
- 每段翻译会带上前 N 段（默认 3）的原文和译文作上下文，并附上该段命中的术语表条目。
- 翻译失败的段落可点「翻译」重试，已翻译的可点「重译」。

## 识别参数（设置 → 识别）

- **识别语言**：默认在意大利语和英语之间逐段自动判断；整节课只用一种语言时固定下来会更准。
- **句末静音** 600ms、**单段最长** 8 秒：超过 4.8 秒后遇到 200ms 停顿即切分，到 8 秒强制切分。
- **说话时逐词实时显示**：说话过程中每约 0.7 秒做一次快速识别，单词逐个出现在虚线框里；句子结束后替换为精确结果再翻译。显卡紧张时可关掉。
- **用术语表提示识别**：把术语表中的词作为 Whisper 提示，提高专业词识别率。

## 记录

- 所有记录自动保存在浏览器 IndexedDB 中，刷新或关闭页面不会丢失（同一浏览器、同一地址 `127.0.0.1:8765`）。
- 每节课可「新建」一条记录，「导出 MD」导出为 Markdown。

## 术语表

内置 225 条通信领域意 / 英 / 中术语，可在「设置 → 术语表」中搜索、编辑、新增、删除，支持导入 JSON（`[{"it","en","zh"}]`）或 CSV（`意大利语,英语,中文` 每行一条）。

## 环境变量（可写入 start.bat）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LIVETRANS_MODEL` | `large-v3-turbo` | 可改为 `large-v3`（更准更慢）、`medium` 等 |
| `LIVETRANS_DEVICE` | `auto` | `cuda` / `cpu` |
| `LIVETRANS_PORT` | `8765` | 端口 |
| `HF_ENDPOINT` | - | HuggingFace 镜像 |

## 结构

```
backend/requirements.txt  Python 依赖
backend/server.py      FastAPI：/ws 音频流、/api/translate 翻译代理、静态页面
backend/vad.py         Silero VAD 流式断句
backend/asr.py         faster-whisper 识别（意/英二选一）
backend/translate.py   OpenAI 兼容接口 / Claude
frontend/              页面、音频采集（AudioWorklet 16kHz）、IndexedDB 记录
install.bat            创建 .venv 并安装依赖
start.bat              启动后端并打开页面
```
