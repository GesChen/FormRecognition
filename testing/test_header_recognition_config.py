"""Tests for HEADER_RECOGNITION OCR workflow override behavior."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import id_form_llm
from config import HEADER_RECOGNITION, OCR_ENGINE


class HeaderRecognitionConfigTests(unittest.TestCase):
    def test_header_ocr_workflow_override_is_temporary(self):
        seen_workflows: list[str] = []
        old_engine_workflow = OCR_ENGINE.get("workflow_default")
        old_header_override = HEADER_RECOGNITION.get("ocr_workflow_override")
        OCR_ENGINE["workflow_default"] = "vision_only"
        HEADER_RECOGNITION["ocr_workflow_override"] = "paddle_vlm_fusion"

        def fake_ocr_raw(path):
            seen_workflows.append(str(OCR_ENGINE.get("workflow_default")))
            return {"detected_text": "header text"}

        try:
            with patch("ocr_engine.ocr_raw", fake_ocr_raw):
                out = id_form_llm._ocr_raw_header("dummy.png")
            workflow_after_call = OCR_ENGINE.get("workflow_default")
        finally:
            OCR_ENGINE["workflow_default"] = old_engine_workflow
            HEADER_RECOGNITION["ocr_workflow_override"] = old_header_override

        self.assertEqual(out["detected_text"], "header text")
        self.assertEqual(seen_workflows, ["paddle_vlm_fusion"])
        self.assertEqual(workflow_after_call, "vision_only")

    def test_header_ocr_uses_engine_default_when_override_empty(self):
        seen_workflows: list[str] = []
        old_engine_workflow = OCR_ENGINE.get("workflow_default")
        old_header_override = HEADER_RECOGNITION.get("ocr_workflow_override")
        OCR_ENGINE["workflow_default"] = "vision_with_paddle_confidence"
        HEADER_RECOGNITION["ocr_workflow_override"] = None

        def fake_ocr_raw(path):
            seen_workflows.append(str(OCR_ENGINE.get("workflow_default")))
            return {"detected_text": "header text"}

        try:
            with patch("ocr_engine.ocr_raw", fake_ocr_raw):
                out = id_form_llm._ocr_raw_header("dummy.png")
        finally:
            OCR_ENGINE["workflow_default"] = old_engine_workflow
            HEADER_RECOGNITION["ocr_workflow_override"] = old_header_override

        self.assertEqual(out["detected_text"], "header text")
        self.assertEqual(seen_workflows, ["vision_with_paddle_confidence"])


if __name__ == "__main__":
    unittest.main()
