# VoiceToSRT 使用说明

这是一个高精度字幕对齐工具，可以将给定的文本稿件与音频文件精确对齐，生成 SRT 字幕文件。

## 功能特点
- **图形化界面 (GUI)**：**[NEW]** 支持拖拽操作，无需敲命令！
- **高精度对齐**：优先使用 stable-whisper 强制对齐原文，逐行输出时间戳。
- **高鲁棒性**：强制对齐失败时自动回退到转写 + 字符插值对齐。
- **模型可选**：GUI 支持选择 Whisper 模型（base/small/medium/large-v2/large-v3）。
- **多语言支持**：支持所有 Whisper 支持的语言（如韩语、日语、中文等）。

## 技术实现原理
1. 读取文本文件，每一行作为一个字幕段落。
2. 使用 stable-whisper 的 `model.align` 对原文进行强制对齐，并保持行级切分（`original_split=True`）。
3. 若强制对齐不可用或失败，则回退到 Whisper 转写 + 段级时间戳 + 字符插值对齐。
4. 输出标准 SRT 文件（HH:MM:SS,mmm 格式）。

## 如何使用

### 方式一：使用图形化界面 (推荐)
为了方便操作并避免环境报错，请直接使用启动脚本：

1. **双击运行** `run_gui.sh` (或者在终端运行 `./run_gui.sh`)
2. 浏览器会自动打开操作界面 (默认地址 `http://127.0.0.1:7860`)。
3. **拖入** 您的音频文件 (如 `.mp3`) 和文案文件 (`.txt`)。
4. 选择语言 (或保持默认 `Auto`)。
5. 选择模型（建议 `large-v3` 以获得更准确的对齐）。
6. 点击 **Generate SRT**。
7. 完成后直接点击右侧文件下载生成的 SRT。

---

### 方式二：命令行使用

#### 环境准备
依赖包括：`openai-whisper`, `ffmpeg`
(推荐使用提供的 Conda 环境)

#### 命令说明
使用 `app.py` 脚本：

```bash
python3 app.py --audio "音频文件路径" --text "文案文件路径" --output "输出路径" --language "语言代码" --model "模型名称"
```

#### 示例
**生成韩语字幕：**
```bash
python3 app.py --audio "韩语音频.MP3" --text "韩语文案.txt" --output "韩语.srt" --language "ko" --model "medium"
```

**生成日语字幕：**
```bash
python3 app.py --audio "日语音频.mp3" --text "日语文案.txt" --output "日文.srt" --language "ja" --model "large-v3"
```

## 模型下载
首次使用大模型会自动下载到缓存目录（默认 `~/.cache/whisper`）。也可以手动预下载：

```bash
/opt/miniconda3/bin/python3 - <<'PY'
import whisper
whisper.load_model("large-v3")
print("downloaded")
PY
```

## Docker 运行（Windows 也可用）
如果你不熟 Docker，推荐用 `docker-compose` 一键启动（会自动持久化模型缓存）。

### 方式一：Dockerfile + docker run
1. 构建镜像：
```bash
docker build -t voice-to-srt .
```
2. 运行容器：
```bash
docker run --rm -p 7860:7860 -v whisper-cache:/root/.cache/whisper voice-to-srt
```
3. 打开浏览器访问：`http://localhost:7860`

### 方式二：docker-compose（推荐）
1. 构建并启动：
```bash
docker compose up --build
```
2. 打开浏览器访问：`http://localhost:7860`
3. 停止服务：
```bash
docker compose down
```

## 常见问题
- **ModuleNotFoundError**: 请务必使用 `./run_gui.sh` 启动，它会自动调用正确的 Python 环境。
- **OMP Error**: `run_gui.sh` 已内置 OpenMP 处理；如遇报错，请确保通过启动脚本运行。
