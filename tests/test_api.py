import io
import unittest
from unittest.mock import AsyncMock, patch

try:
    from fastapi.testclient import TestClient
    from app import _process_audio, app
    from lyrics_sync.alignment import WordSpan
except ImportError:  # Core unit tests remain runnable before optional web deps are installed.
    TestClient = None


@unittest.skipIf(TestClient is None, "FastAPI dependencies are not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    @patch("app.run_in_threadpool", new_callable=AsyncMock)
    def test_align_dispatches_blocking_work_to_threadpool(self, mocked_threadpool):
        mocked_threadpool.return_value = (
            8.0,
            [],
            {"overall_confidence": 1.0},
            {"runtime": "gpu"},
        )
        response = self.client.post(
            "/api/align",
            files={"audio": ("demo.mp3", io.BytesIO(b"fake-audio"), "audio/mpeg")},
            data={"lyrics": "第一句", "model": "small", "language": "zh"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        mocked_threadpool.assert_awaited_once()
        dispatched_function = mocked_threadpool.await_args.args[0]
        self.assertEqual(dispatched_function.__name__, "_process_audio")

    @patch("app.audio_duration", return_value=8.0)
    @patch("app.transcribe")
    def test_align_contract(self, mocked_transcribe, _mocked_duration):
        mocked_transcribe.return_value = (
            [
                WordSpan("第一句", 1.0, 2.0, 0.9),
                WordSpan("第二句", 3.0, 4.0, 0.9),
            ],
            {"detected_language": "zh", "language_probability": 1.0, "segments": 2, "words": 2},
        )
        response = self.client.post(
            "/api/align",
            files={"audio": ("demo.mp3", io.BytesIO(b"fake-audio"), "audio/mpeg")},
            data={"lyrics": "第一句\n第二句", "model": "small", "language": "zh"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["metadata"]["filename"], "demo.mp3")
        self.assertEqual(len(payload["lines"]), 2)
        self.assertEqual(set(payload["exports"]), {"lrc", "srt", "csv", "json", "jsx"})
        self.assertIn("第一句", payload["exports"]["lrc"])

    @patch("app.audio_duration", return_value=8.0)
    @patch("app.transcribe")
    def test_low_match_model_falls_back_to_small(self, mocked_transcribe, _mocked_duration):
        mocked_transcribe.side_effect = [
            ([WordSpan("词词词词", 0.0, 1.0, 0.1)], {"runtime": "gpu"}),
            ([WordSpan("第一句第二句", 1.0, 4.0, 0.9)], {"runtime": "gpu"}),
        ]

        _, lines, diagnostics, transcription = _process_audio(
            None,
            "第一句\n第二句",
            ["第一句", "第二句"],
            "turbo",
            "zh",
        )

        self.assertGreater(diagnostics["overall_confidence"], 0.9)
        self.assertEqual(len(lines), 2)
        self.assertEqual(transcription["requested_model"], "turbo")
        self.assertEqual(transcription["effective_model"], "small")
        self.assertEqual(mocked_transcribe.call_args_list[1].args[2], "small")


if __name__ == "__main__":
    unittest.main()
