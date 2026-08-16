from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .alignment import WordSpan


class TranscriberUnavailable(RuntimeError):
    pass


def _runtime_device() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@lru_cache(maxsize=4)
def load_model(model_name: str, device: str = "", compute_type: str = ""):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriberUnavailable(
            f"语音识别运行依赖不完整（{exc}）。请运行：python -m pip install -r requirements.txt"
        ) from exc

    if not device or not compute_type:
        device, compute_type = _runtime_device()
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def audio_duration(path: str | Path) -> float:
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            durations = [
                float(stream.duration * stream.time_base)
                for stream in container.streams.audio
                if stream.duration is not None and stream.time_base is not None
            ]
            if durations:
                return max(durations)
    except Exception:
        pass
    return 0.0


def transcribe(
    audio_path: str | Path,
    lyrics: str,
    model_name: str = "small",
    language: str = "zh",
) -> tuple[list[WordSpan], dict]:
    model = load_model(model_name)
    prompt = lyrics[:6000]

    def run(active_model):
        segments, info = active_model.transcribe(
            str(audio_path),
            language=language or None,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=True,
            initial_prompt=prompt,
        )

        words: list[WordSpan] = []
        segment_count = 0
        for segment in segments:
            segment_count += 1
            if segment.words:
                for word in segment.words:
                    if word.start is None or word.end is None:
                        continue
                    words.append(
                        WordSpan(
                            text=word.word,
                            start=float(word.start),
                            end=float(word.end),
                            probability=float(word.probability or 0.0),
                        )
                    )
        return words, info, segment_count

    runtime = "gpu"
    try:
        words, info, segment_count = run(model)
    except RuntimeError as exc:
        message = str(exc).casefold()
        cuda_runtime_missing = any(name in message for name in ("cublas", "cudnn", "cuda"))
        if not cuda_runtime_missing:
            raise
        runtime = "cpu-fallback"
        words, info, segment_count = run(load_model(model_name, "cpu", "int8"))

    details = {
        "detected_language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "segments": segment_count,
        "words": len(words),
        "runtime": runtime,
    }
    return words, details
