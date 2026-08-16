import unittest

from lyrics_sync.alignment import TimedLine, WordSpan, _repair_repeated_transitions, align_lyrics, clean_lyrics
from lyrics_sync.exporters import export_ae_jsx, export_csv, export_lrc, export_srt


class AlignmentTests(unittest.TestCase):
    def test_clean_lyrics(self):
        self.assertEqual(clean_lyrics("第一句\r\n\r\n 第二句 \n"), ["第一句", "第二句"])

    def test_exact_chinese_alignment(self):
        lyrics = ["教室里那台风琴", "像你告白的声音"]
        words = [
            WordSpan("教室里", 1.0, 2.2, 0.9),
            WordSpan("那台风琴", 2.2, 4.0, 0.9),
            WordSpan("像你告白", 5.0, 6.5, 0.8),
            WordSpan("的声音", 6.5, 7.6, 0.8),
        ]
        lines, diagnostics = align_lyrics(lyrics, words, 10.0)
        self.assertEqual(len(lines), 2)
        self.assertAlmostEqual(lines[0].start, 1.0, places=2)
        self.assertGreaterEqual(lines[1].start, 5.0)
        self.assertGreater(lines[0].confidence, 0.8)
        self.assertEqual(diagnostics["overall_confidence"], 1.0)

    def test_alignment_tolerates_wrong_recognition(self):
        lyrics = ["亲爱的那不是爱情", "再怎么美丽也只能是曾经"]
        words = [
            WordSpan("亲爱的那不是爱琴", 12.0, 15.0, 0.65),
            WordSpan("再怎么美丽也只能是曾今", 17.0, 21.0, 0.6),
        ]
        lines, diagnostics = align_lyrics(lyrics, words, 24.0)
        self.assertEqual(len(lines), 2)
        self.assertLess(lines[0].start, lines[1].start)
        self.assertGreater(diagnostics["overall_confidence"], 0.8)

    def test_phonetic_alignment_uses_homophones_as_time_anchors(self):
        lyrics = [
            "可是流光 赶来答疑",
            "用身高举例 拿告别作比",
            "先生秃着头顶 说这叫川流不息",
            "假装没听见我问 流去到哪里",
        ]
        words = [
            WordSpan("可是流光敢来大意", 1.0, 4.0, 0.8),
            WordSpan("用深刻距离拉高别作弊", 4.0, 8.0, 0.8),
            WordSpan("先生吐着头顶说着脚传流不息", 8.0, 13.0, 0.8),
            WordSpan("假装没听见我问流去到哪里", 13.0, 17.0, 0.8),
        ]

        lines, diagnostics = align_lyrics(lyrics, words, 20.0)

        self.assertEqual([line.confidence for line in lines], [1.0, 0.8, 1.0, 1.0])
        self.assertLess(lines[0].text_confidence, lines[0].confidence)
        self.assertLess(lines[2].text_confidence, lines[2].confidence)
        self.assertGreater(diagnostics["phonetic_confidence"], diagnostics["text_confidence"])
        self.assertEqual(diagnostics["overall_confidence"], diagnostics["phonetic_confidence"])

    def test_low_probability_noise_is_not_used_as_an_anchor(self):
        lines, diagnostics = align_lyrics(
            ["正确歌词"],
            [
                WordSpan("幻觉噪声", 0.0, 1.0, 0.05),
                WordSpan("正确歌词", 2.0, 3.0, 0.9),
            ],
            4.0,
        )

        self.assertEqual(lines[0].start, 2.0)
        self.assertEqual(lines[0].confidence, 1.0)
        self.assertEqual(diagnostics["recognized_text"], "正确歌词")

    def test_exports(self):
        lines, _ = align_lyrics(
            ["第一句", "第二句"],
            [WordSpan("第一句", 1.0, 2.0, 1.0), WordSpan("第二句", 3.0, 4.0, 1.0)],
            5.0,
        )
        self.assertIn("[00:01.00]第一句", export_lrc(lines))
        self.assertIn("00:00:01,000 -->", export_srt(lines))
        self.assertTrue(export_csv(lines).startswith("\ufeff"))
        self.assertIn("comp.layers.addText", export_ae_jsx(lines))
        self.assertIn("第一句", export_ae_jsx(lines))

    def test_repeated_transition_repairs_held_syllable_outlier(self):
        lines = [
            TimedLine(1, "甲歌词", 0.0, 3.0, 1.0, 10, 10),
            TimedLine(2, "乙歌词", 12.0, 15.0, 1.0, 10, 10),
            TimedLine(3, "过渡歌词", 17.0, 20.0, 1.0, 10, 10),
            TimedLine(4, "甲歌词", 30.0, 33.0, 1.0, 10, 10),
            TimedLine(5, "乙歌词", 35.0, 38.0, 1.0, 10, 10),
            TimedLine(6, "丙歌词", 40.0, 43.0, 1.0, 10, 10),
            TimedLine(7, "丁歌词", 40.2, 43.0, 1.0, 10, 10),
            TimedLine(8, "过渡歌词", 46.0, 49.0, 1.0, 10, 10),
            TimedLine(9, "丙歌词", 55.0, 58.0, 1.0, 10, 10),
            TimedLine(10, "丁歌词", 60.0, 63.0, 1.0, 10, 10),
        ]
        _repair_repeated_transitions(lines)
        self.assertAlmostEqual(lines[1].start, 5.0, places=2)
        self.assertAlmostEqual(lines[6].start, 45.0, places=2)


if __name__ == "__main__":
    unittest.main()
