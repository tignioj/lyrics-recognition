import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError

from lyrics_sync.transcriber import _cached_model_path, model_cache_status, transcribe


class TranscriberTests(unittest.TestCase):
    def test_model_cache_status_reports_complete_snapshot_and_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir) / "models--Systran--faster-whisper-small"
            snapshot = repo_dir / "snapshots" / "commit-id"
            snapshot.mkdir(parents=True)
            (repo_dir / "refs").mkdir()
            (repo_dir / "refs" / "main").write_text("commit-id", encoding="utf-8")
            (snapshot / "config.json").write_bytes(b"{}")
            (snapshot / "model.bin").write_bytes(b"model")

            status = model_cache_status("small", temp_dir)

            self.assertEqual(status["status"], "downloaded")
            self.assertTrue(status["downloaded"])
            self.assertEqual(status["progress"], 100.0)
            self.assertEqual(status["path"], str(snapshot))

    def test_model_cache_status_reports_partial_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir) / "models--Systran--faster-whisper-large-v3"
            blobs = repo_dir / "blobs"
            blobs.mkdir(parents=True)
            (blobs / "model.incomplete").write_bytes(b"partial")

            status = model_cache_status("large-v3", temp_dir)

            self.assertEqual(status["status"], "partial")
            self.assertEqual(status["downloaded_bytes"], 7)
            self.assertEqual(status["path"], str(repo_dir))

    @patch("faster_whisper.utils.download_model")
    def test_cached_model_path_uses_complete_local_snapshot(self, mocked_download):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").touch()
            (model_dir / "model.bin").touch()
            mocked_download.return_value = str(model_dir)

            self.assertEqual(_cached_model_path("small"), str(model_dir))
            mocked_download.assert_called_once_with("small", local_files_only=True)

    @patch("faster_whisper.utils.download_model")
    def test_cached_model_path_rejects_incomplete_snapshot(self, mocked_download):
        with tempfile.TemporaryDirectory() as temp_dir:
            mocked_download.return_value = temp_dir
            self.assertIsNone(_cached_model_path("turbo"))

    @patch("faster_whisper.utils.download_model")
    def test_cached_model_path_handles_cache_miss(self, mocked_download):
        mocked_download.side_effect = LocalEntryNotFoundError("not cached")
        self.assertIsNone(_cached_model_path("medium"))

    @patch("lyrics_sync.transcriber.load_model")
    def test_transcribe_retries_without_prompt_for_invisible_output(self, mocked_load):
        class FakeModel:
            def __init__(self):
                self.prompts = []

            def transcribe(self, _audio_path, **kwargs):
                prompt = kwargs["initial_prompt"]
                self.prompts.append(prompt)
                text = "\u200b" if prompt else "歌词"
                word = SimpleNamespace(start=1.0, end=2.0, word=text, probability=0.9)
                segment = SimpleNamespace(words=[word])
                info = SimpleNamespace(language="zh", language_probability=1.0)
                return iter([segment]), info

        model = FakeModel()
        mocked_load.return_value = model

        words, details = transcribe("demo.wav", "歌词", "small", "zh")

        self.assertEqual([word.text for word in words], ["歌词"])
        self.assertEqual(model.prompts, ["歌词", None])
        self.assertEqual(details["attempts"], 2)
        self.assertTrue(details["prompt_retry"])


if __name__ == "__main__":
    unittest.main()
