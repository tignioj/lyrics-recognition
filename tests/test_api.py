import io
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from app import app
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


if __name__ == "__main__":
    unittest.main()
