"""Tests for OCR engine workflow dispatch."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import ocr_engine
from config import OCR_ENGINE


class OcrEngineWorkflowTests(unittest.TestCase):
    def test_paddle_only_dispatch_uses_paddle_without_vision_fallback(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image_path = Path(tmp.name)

            def fake_paddle_stage(path, *, verbose, force_failure=False):
                return (
                    {
                        "stage_index": 0,
                        "model": OCR_ENGINE.get("paddle_model_name", "paddle-ppocrv5"),
                        "elapsed_sec": 0.01,
                        "raw_response": [],
                        "response_text": "hello",
                        "parsed": {
                            "detected_text": "hello",
                            "confidence_score": 0.99,
                            "confidence_label": "high",
                            "needs_human_review": False,
                        },
                        "accepted": True,
                        "error": None,
                        "image_payload": {},
                    },
                    {
                        "detected_text": "hello",
                        "confidence_score": 0.99,
                        "rec_texts": ["hello"],
                        "rec_scores": [0.99],
                        "raw_response": [],
                    },
                )

            old_workflow = OCR_ENGINE.get("workflow_default")
            OCR_ENGINE["workflow_default"] = "paddle_only"
            try:
                with patch.object(ocr_engine, "_run_paddle_stage", fake_paddle_stage):
                    with patch.object(ocr_engine, "_run_vision_stage") as vision_stage:
                        result = ocr_engine.ocr_raw(image_path)
            finally:
                OCR_ENGINE["workflow_default"] = old_workflow

            self.assertEqual(result["detected_text"], "hello")
            self.assertEqual(result["selected_model"], OCR_ENGINE.get("paddle_model_name", "paddle-ppocrv5"))
            self.assertEqual(len(result["stages"]), 1)
            vision_stage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
