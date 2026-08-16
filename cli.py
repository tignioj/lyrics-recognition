from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from lyrics_sync.alignment import align_lyrics, clean_lyrics
from lyrics_sync.exporters import build_exports
from lyrics_sync.transcriber import audio_duration, transcribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将音频与标准歌词对齐并导出时间戳")
    parser.add_argument("audio", type=Path, help="音频文件路径")
    parser.add_argument("lyrics", type=Path, help="UTF-8 歌词文本，一行一句")
    parser.add_argument("--model", default="small", help="Whisper 模型（默认：small）")
    parser.add_argument("--language", default="zh", help="语言代码（默认：zh）")
    parser.add_argument("--output", type=Path, default=Path("output"), help="输出目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        print(f"找不到音频：{args.audio}", file=sys.stderr)
        return 2
    if not args.lyrics.is_file():
        print(f"找不到歌词：{args.lyrics}", file=sys.stderr)
        return 2

    raw_lyrics = args.lyrics.read_text(encoding="utf-8-sig")
    lyric_lines = clean_lyrics(raw_lyrics)
    print(f"载入 {len(lyric_lines)} 行歌词；模型：{args.model}")
    started = time.perf_counter()
    duration = audio_duration(args.audio)
    words, transcription = transcribe(args.audio, raw_lyrics, args.model, args.language)
    print(f"识别到 {len(words)} 个带时间词组，开始与标准歌词对齐……")
    timed_lines, diagnostics = align_lyrics(lyric_lines, words, duration)

    metadata = {
        "filename": args.audio.name,
        "duration": round(duration, 3),
        "model": args.model,
        "language": args.language,
        "processing_seconds": round(time.perf_counter() - started, 3),
        "diagnostics": diagnostics,
        "transcription": transcription,
    }
    exports = build_exports(timed_lines, metadata)
    target = args.output / args.audio.stem
    target.mkdir(parents=True, exist_ok=True)
    for extension, content in exports.items():
        (target / f"{args.audio.stem}.{extension}").write_text(content, encoding="utf-8")
    (target / "result.json").write_text(
        json.dumps(
            {"metadata": metadata, "lines": [line.to_dict() for line in timed_lines]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    confidence = diagnostics["overall_confidence"] * 100
    print(f"完成：{len(timed_lines)} 行，整体匹配 {confidence:.1f}%")
    print(f"输出目录：{target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

