"""Regression tests for VLM stream early-stop heuristics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import ocr_engine


class OcrEngineStreamStopTests(unittest.TestCase):
    def test_stops_on_complete_json_object_before_closing_fence(self) -> None:
        text = '```json\n{\n"text": "Record ID: 8011571A"\n}'
        stop = ocr_engine._stream_stop_completion(text)
        self.assertIsNotNone(stop)
        self.assertEqual(stop.get("stop_reason"), "json_completion")
        start = int(stop["start_index"])
        end = int(stop["end_index"])
        self.assertEqual(text[start:end], '{\n"text": "Record ID: 8011571A"\n}')

    def test_repeated_json_blocks_keep_first_answer(self) -> None:
        text = '{"text":"A"}\n{"text":"A"}\n{"text":"A"}'
        stop = ocr_engine._stream_text_json_repeat_completion(text)
        self.assertIsNotNone(stop)
        self.assertEqual(stop.get("mode"), "repeated_identical_json")
        kept = ocr_engine._completed_json_from_stream(text, stop)
        self.assertEqual(kept, '{"text":"A"}')

    def test_repeated_tail_stops_malformed_loop(self) -> None:
        unit = '```json\n{"text": "still looping"\n'
        text = "prefix " + (unit * 3)
        stop = ocr_engine._stream_repeated_tail_completion(text)
        self.assertIsNotNone(stop)
        self.assertEqual(stop.get("mode"), "repeated_tail")
        kept = ocr_engine._completed_json_from_stream(text, stop)
        self.assertEqual(kept, "prefix " + unit)


if __name__ == "__main__":
    unittest.main()
