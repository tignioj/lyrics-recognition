from __future__ import annotations

import os
from pathlib import Path
import tempfile
from threading import Thread
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from lyrics_sync import __version__
from lyrics_sync.alignment import align_lyrics, clean_lyrics
from lyrics_sync.exporters import build_exports
from lyrics_sync.transcriber import (
    TranscriberUnavailable,
    audio_duration,
    ensure_model_downloaded,
    model_cache_status,
    transcribe,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "200")) * 1024 * 1024
ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3", "turbo"}
MODEL_ORDER = ("small", "medium", "large-v3", "turbo", "base", "tiny")
MINIMUM_USEFUL_MATCH = 0.05

app = FastAPI(
    title="Lyrics Sync API",
    version=__version__,
    description="上传音频与标准歌词，返回逐行时间戳及 AE/LRC/SRT/CSV 导出。",
)


def _process_audio(
    audio_path: Path,
    lyrics: str,
    lines: list[str],
    model: str,
    language: str,
) -> tuple[float, list, dict, dict]:
    """Run CPU/GPU-heavy work away from FastAPI's event-loop thread."""
    duration = audio_duration(audio_path)
    words, transcription = transcribe(audio_path, lyrics, model, language)
    if duration <= 0 and words:
        duration = max(word.end for word in words)
    timed_lines, diagnostics = align_lyrics(lines, words, duration)

    if model != "small" and diagnostics["overall_confidence"] < MINIMUM_USEFUL_MATCH:
        fallback_words, fallback_transcription = transcribe(
            audio_path,
            lyrics,
            "small",
            language,
        )
        fallback_lines, fallback_diagnostics = align_lyrics(
            lines,
            fallback_words,
            duration,
        )
        if fallback_diagnostics["overall_confidence"] > diagnostics["overall_confidence"]:
            timed_lines = fallback_lines
            diagnostics = fallback_diagnostics
            transcription = fallback_transcription
            transcription.update(
                {
                    "requested_model": model,
                    "effective_model": "small",
                    "fallback_reason": "requested model had almost no lyric matches",
                }
            )

    transcription.setdefault("requested_model", model)
    transcription.setdefault("effective_model", model)
    return duration, timed_lines, diagnostics, transcription


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/models")
def models() -> dict:
    return {"models": [model_cache_status(name) for name in MODEL_ORDER]}


def _download_model_in_background(model_name: str) -> None:
    try:
        ensure_model_downloaded(model_name)
    except Exception:
        # The transcriber records the error for /api/models. The download is a
        # background convenience task and must not terminate the web server.
        return


@app.post("/api/models/{model_name}/download", status_code=202)
def download_model(model_name: str) -> dict:
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(status_code=404, detail=f"不支持的模型：{model_name}")
    status = model_cache_status(model_name)
    if status["status"] == "downloaded":
        return status
    if status["status"] != "downloading":
        Thread(
            target=_download_model_in_background,
            args=(model_name,),
            name=f"model-download-{model_name}",
            daemon=True,
        ).start()
    return {**status, "status": "downloading"}


@app.post("/api/align")
async def align(
    audio: UploadFile = File(..., description="MP3/WAV/M4A/FLAC 等音频文件"),
    lyrics: str = Form(..., description="每行一句的标准歌词"),
    model: str = Form("small", description="Whisper 模型"),
    language: str = Form("zh", description="ISO 语言代码"),
) -> dict:
    started = time.perf_counter()
    lines = clean_lyrics(lyrics)
    if not lines:
        raise HTTPException(status_code=422, detail="请至少输入一行歌词")
    if model not in ALLOWED_MODELS:
        raise HTTPException(status_code=422, detail=f"不支持的模型：{model}")

    suffix = Path(audio.filename or "audio.mp3").suffix or ".audio"
    total = 0
    try:
        with tempfile.TemporaryDirectory(prefix="lyrics-sync-") as temp_dir:
            audio_path = Path(temp_dir) / f"input{suffix}"
            with audio_path.open("wb") as output:
                while chunk := await audio.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise HTTPException(status_code=413, detail="音频文件超过大小限制")
                    output.write(chunk)

            duration, timed_lines, diagnostics, transcription = await run_in_threadpool(
                _process_audio,
                audio_path,
                lyrics,
                lines,
                model,
                language,
            )
    except HTTPException:
        raise
    except TranscriberUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"处理音频失败：{exc}") from exc
    finally:
        await audio.close()

    metadata = {
        "filename": audio.filename or "audio",
        "duration": round(duration, 3),
        "model": model,
        "effective_model": transcription.get("effective_model", model),
        "language": language,
        "processing_seconds": round(time.perf_counter() - started, 3),
        "diagnostics": diagnostics,
        "transcription": transcription,
    }
    return {
        "metadata": metadata,
        "lines": [line.to_dict() for line in timed_lines],
        "exports": build_exports(timed_lines, metadata),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
