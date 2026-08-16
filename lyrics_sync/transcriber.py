from __future__ import annotations

import ctypes
from functools import lru_cache
import os
from pathlib import Path
import site
from threading import Lock
from typing import Any

from .alignment import WordSpan, normalized_characters


class TranscriberUnavailable(RuntimeError):
    pass


_CUDA_DLL_HANDLES: list[object] = []

MODEL_REPOSITORIES = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

# Exact total of the files selected by faster-whisper's download allow-list.
# Keeping this local makes the status endpoint fast and available while offline.
MODEL_DOWNLOAD_BYTES = {
    "tiny": 78_203_619,
    "base": 147_882_941,
    "small": 486_212_372,
    "medium": 1_530_571_735,
    "large-v3": 3_090_835_702,
    "turbo": 1_621_665_983,
}

_MODEL_FILES = {
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.json",
    "vocabulary.txt",
}
_DOWNLOAD_PROGRESS: dict[str, dict[str, Any]] = {}
_DOWNLOAD_PROGRESS_LOCK = Lock()


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


def _model_repo_path(model_name: str, cache_root: str | Path | None = None) -> Path:
    if model_name not in MODEL_REPOSITORIES:
        raise ValueError(f"Unsupported model: {model_name}")
    if cache_root is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_root = HF_HUB_CACHE
    owner, repository = MODEL_REPOSITORIES[model_name].split("/", 1)
    return Path(cache_root) / f"models--{owner}--{repository}"


def _snapshot_candidates(repo_path: Path) -> list[Path]:
    snapshots = repo_path / "snapshots"
    candidates: list[Path] = []
    reference = repo_path / "refs" / "main"
    try:
        candidate = snapshots / reference.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            candidates.append(candidate)
    except OSError:
        pass
    if snapshots.is_dir():
        candidates.extend(
            path for path in snapshots.iterdir() if path.is_dir() and path not in candidates
        )
    return candidates


def _cache_details(model_name: str, cache_root: str | Path | None = None) -> dict:
    repo_path = _model_repo_path(model_name, cache_root)
    candidates = _snapshot_candidates(repo_path)
    complete_path = next(
        (
            path
            for path in candidates
            if (path / "config.json").is_file() and (path / "model.bin").is_file()
        ),
        None,
    )
    snapshot_path = complete_path or (candidates[0] if candidates else None)

    downloaded_bytes = 0
    if snapshot_path:
        for filename in _MODEL_FILES:
            file_path = snapshot_path / filename
            try:
                if file_path.is_file():
                    downloaded_bytes += file_path.stat().st_size
            except OSError:
                pass
    blobs_path = repo_path / "blobs"
    if blobs_path.is_dir():
        for file_path in blobs_path.glob("*.incomplete"):
            try:
                downloaded_bytes += file_path.stat().st_size
            except OSError:
                pass

    return {
        "complete": complete_path is not None,
        "downloaded_bytes": downloaded_bytes,
        "path": str(complete_path or snapshot_path or repo_path),
    }


def model_cache_status(model_name: str, cache_root: str | Path | None = None) -> dict:
    """Describe cache and live download state for one model."""
    details = _cache_details(model_name, cache_root)
    total_bytes = MODEL_DOWNLOAD_BYTES[model_name]
    with _DOWNLOAD_PROGRESS_LOCK:
        live = dict(_DOWNLOAD_PROGRESS.get(model_name, {}))

    downloaded_bytes = details["downloaded_bytes"]
    if live.get("status") == "downloading":
        downloaded_bytes = max(downloaded_bytes, int(live.get("downloaded_bytes", 0)))
        status = "downloading"
    elif details["complete"]:
        downloaded_bytes = total_bytes
        status = "downloaded"
    elif live.get("status") == "error":
        status = "error"
    elif downloaded_bytes:
        status = "partial"
    else:
        status = "not_downloaded"

    downloaded_bytes = min(downloaded_bytes, total_bytes)
    progress = round(downloaded_bytes / total_bytes * 100, 1) if total_bytes else None
    result = {
        "name": model_name,
        "repository": MODEL_REPOSITORIES[model_name],
        "status": status,
        "downloaded": status == "downloaded",
        "downloaded_bytes": downloaded_bytes,
        "total_bytes": total_bytes,
        "progress": progress,
        "path": details["path"],
    }
    if live.get("error"):
        result["error"] = live["error"]
    return result


def _tracked_tqdm(model_name: str):
    from tqdm.auto import tqdm

    class ModelDownloadTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_progress_id = id(self)
            self._record()

        def _record(self) -> None:
            # Ignore the small "Fetching N files" counter; byte bars are what
            # users need and their totals are at least several hundred bytes.
            if not self.total or self.total < 256:
                return
            with _DOWNLOAD_PROGRESS_LOCK:
                state = _DOWNLOAD_PROGRESS.setdefault(model_name, {})
                jobs = state.setdefault("jobs", {})
                jobs[self._model_progress_id] = int(self.n)
                state["downloaded_bytes"] = sum(jobs.values())

        def update(self, n=1):
            result = super().update(n)
            self._record()
            return result

        def close(self):
            self._record()
            return super().close()

    return ModelDownloadTqdm


def _download_model_with_progress(model_name: str) -> str:
    from huggingface_hub import snapshot_download

    if model_name not in MODEL_REPOSITORIES:
        raise ValueError(f"Unsupported model: {model_name}")
    with _DOWNLOAD_PROGRESS_LOCK:
        _DOWNLOAD_PROGRESS[model_name] = {
            "status": "downloading",
            "downloaded_bytes": _cache_details(model_name)["downloaded_bytes"],
            "jobs": {},
        }
    try:
        model_path = snapshot_download(
            MODEL_REPOSITORIES[model_name],
            allow_patterns=list(_MODEL_FILES),
            tqdm_class=_tracked_tqdm(model_name),
        )
    except Exception as exc:
        with _DOWNLOAD_PROGRESS_LOCK:
            state = _DOWNLOAD_PROGRESS.setdefault(model_name, {})
            state.update({"status": "error", "error": str(exc)})
        raise
    with _DOWNLOAD_PROGRESS_LOCK:
        _DOWNLOAD_PROGRESS.pop(model_name, None)
    return model_path


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
            f"语音识别运行依赖不完整（{exc}）。请运行：uv sync"
        ) from exc

    if not device or not compute_type:
        device, compute_type = _runtime_device()
    cached_path = _cached_model_path(model_name)
    model_source = cached_path or _download_model_with_progress(model_name)
    return WhisperModel(
        model_source,
        device=device,
        compute_type=compute_type,
        local_files_only=True,
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
