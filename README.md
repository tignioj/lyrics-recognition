# Lyrics Sync：音频与歌词自动对齐

输入一段音频和“每行一句”的标准歌词，服务会用 Whisper 识别歌声的时间位置，再把识别结果与标准歌词进行字符级顺序对齐。输出文本始终使用你提供的歌词，Whisper 只负责提供时间锚点。

可导出：

- `LRC`：播放器、歌词程序常用格式
- `SRT`：Premiere、剪映及大多数字幕软件可用
- `CSV`：表格或批处理
- `JSON`：其他程序/API 集成
- `JSX`：在 After Effects 中直接生成逐句文字图层

## 本地启动

需要 Python 3.10 或更高版本。`faster-whisper` 通过 PyAV 解码音频，不要求单独安装系统 FFmpeg。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。接口文档在 <http://127.0.0.1:8000/docs>。

如果 Windows 报端口权限或占用错误，可换一个端口，例如 `--port 8765`，并打开 <http://127.0.0.1:8765>。

首次使用某个 Whisper 模型时会下载模型文件。中文歌曲建议先用 `small`；准确度不足时换成 `medium` 或 `large-v3`。CPU 可以运行但速度较慢，有 NVIDIA CUDA 时会自动使用 GPU 和 FP16。

也可以不启动服务，直接进行批处理：

```powershell
.venv\Scripts\python.exe cli.py "song\亲爱的，那不是爱情_张韶涵.mp3" `
  "song\亲爱的，那不是爱情_张韶涵.txt" --model small
```

五种导出文件会写入 `output/<歌曲文件名>/`。

## API 调用

`POST /api/align` 使用 `multipart/form-data`：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/align `
  -F "audio=@song/亲爱的，那不是爱情_张韶涵.mp3" `
  -F "lyrics=<lyrics.txt" `
  -F "model=small" `
  -F "language=zh" `
  -o result.json
```

响应主体包含：

```json
{
  "metadata": {
    "filename": "song.mp3",
    "duration": 250.2,
    "diagnostics": { "overall_confidence": 0.87 }
  },
  "lines": [
    {
      "index": 1,
      "text": "教室里那台风琴 叮咚叮咚叮咛",
      "start": 18.42,
      "end": 24.91,
      "confidence": 0.91
    }
  ],
  "exports": { "lrc": "...", "srt": "...", "csv": "...", "json": "...", "jsx": "..." }
}
```

`overall_confidence` 是标准歌词字符与识别字符的整体精确匹配率；每一行另有 `confidence`。歌唱识别比普通讲话困难，低匹配行应在网页中点击试听并人工复核。

## 导入 After Effects

1. 在网页结果区下载 `AE · JSX`。
2. 在 AE 中打开或选中目标合成。
3. 选择 **文件 → 脚本 → 运行脚本文件…**，选择下载的 `.jsx`。
4. 脚本会为每句歌词创建一个文字图层，并设置 `inPoint`、`outPoint` 和底部居中的初始位置。
5. 批量选中文字图层后即可统一修改字体、字号、动画预设等样式。

如果你的下游工具有自己的格式，优先读取 JSON 中的 `lines[].start` / `lines[].end`（单位为秒）。

## 精度建议

- 歌词需要包含所有重复段落，顺序必须和歌曲一致。
- 不要加入歌曲里没有唱出的制作信息、歌手名或章节标题。
- 现场版、混响很重或伴奏盖过人声时，使用更大的模型。
- 当前版本以逐行时间戳为目标；低置信度行适合人工微调。

## 测试

核心对齐算法的测试不需要下载 Whisper 模型：

```powershell
python -m unittest discover -s tests -v
```
