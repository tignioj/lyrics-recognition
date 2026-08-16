from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import site

from .alignment import WordSpan, normalized_characters


class TranscriberUnavailable(RuntimeError):
    pass


_CUDA_DLL_HANDLES: list[object] = []


def _configure_cuda_dll_directories() -> None:
    """Expose CUDA DLLs installed by NVIDIA's Windows Python packages."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    relative_directories = (
        Path("nvidia/cublas/bin"),
        Path("nvidia/cuda_nvrtc/bin"),
    )
    for package_root in map(Path, site.getsitepackages()):
        for relative_directory in relative_directories:
            dll_directory = package_root / relative_directory
            if dll_directory.is_dir():
                _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(dll_directory)))
                if relative_directory == Path("nvidia/cublas/bin"):
                    # CTranslate2 loads cuBLAS by basename at first inference.
                    # Preloading by absolute path makes the pip-installed DLLs
                    # visible to that native loader on Windows.
                    for filename in ("cublasLt64_12.dll", "cublas64_12.dll"):
                        dll_path = dll_directory / filename
                        if dll_path.is_file():
                            _CUDA_DLL_HANDLES.append(ctypes.WinDLL(str(dll_path)))


_configure_cuda_dll_directories()


def _cached_model_path(model_name: str) -> str | None:
    """Return a complete cached model without contacting Hugging Face."""
    from faster_whisper.utils import download_model
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        model_path = Path(download_model(model_name, local_files_only=True))
    except LocalEntryNotFoundError:
        return None

    required_files = ("config.json", "model.bin")
    if not all((model_path / filename).is_file() for filename in required_files):
        return None
    return str(model_path)


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
    cached_path = _cached_model_path(model_name)
    model_source = cached_path or model_name
    return WhisperModel(
        model_source,
        device=device,
        compute_type=compute_type,
        local_files_only=bool(cached_path),
    )


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

    def run(active_model, active_prompt: str | None):
        segments, info = active_model.transcribe(
            str(audio_path),
            language=language or None,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=True,
            initial_prompt=active_prompt,
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
    active_model = model
    try:
        words, info, segment_count = run(active_model, prompt)
    except RuntimeError as exc:
        message = str(exc).casefold()
        cuda_runtime_missing = any(name in message for name in ("cublas", "cudnn", "cuda"))
        if not cuda_runtime_missing:
            raise
        runtime = "cpu-fallback"
        active_model = load_model(model_name, "cpu", "int8")
        words, info, segment_count = run(active_model, prompt)

    attempts = 1
    usable_text = "".join(word.text for word in words)
    if prompt and not normalized_characters(usable_text):
        # Some singing voices make Whisper echo only invisible prompt tokens.
        # A prompt-free retry recovers useful timestamps for those files.
        words, info, segment_count = run(active_model, None)
        attempts += 1

    details = {
        "detected_language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "segments": segment_count,
        "words": len(words),
        "runtime": runtime,
        "attempts": attempts,
        "prompt_retry": attempts > 1,
    }
    return words, details
