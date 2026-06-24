"""Tests for Paddle/VLM fusion inputs in text ROI normalization."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import text_roi_llm


class TextRoiLlmFusionTests(unittest.TestCase):
    def test_postprocess_prompt_includes_paddle_sidecar(self):
        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {
                "text": '{"detected_text":"1234567A"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"id": "12345674"},
                roi_meta_by_name={
                    "id": {
                        "llm_field_data_type": "id",
                        "llm_validation_rules": "Must match ^[0-9]{7}[AB]$ or null",
                        "ocr_paddle_confidence": {
                            "detected_text": "1234567A",
                            "confidence_score": 0.97,
                            "confidence_label": "high",
                            "needs_human_review": False,
                        },
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["id"], "1234567A")
        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertIn("Paddle/VLM fusion inputs:", prompt)
        self.assertIn("VLM extracted text: 12345674", prompt)
        self.assertIn("PaddleOCR extracted text: 1234567A", prompt)
        self.assertTrue(debug["per_roi"][0]["used_paddle_vlm_fusion"])

    def test_postprocess_coerces_string_null_to_empty(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {
                "text": '{"detected_text":"null"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois({"age": "unclear"})

        self.assertEqual(out["age"], "")

    def test_postprocess_uses_roi_name_metadata_and_drops_internal_uid_echo(self):
        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {
                "text": '{"detected_text":"p702:r1:id"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"p702:r1:id": ""},
                roi_meta_by_name={
                    "p702:r1:id": {
                        "roi_name": "id",
                        "llm_prompt_override": "Map the OCR JSON field id into detected_text.",
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["p702:r1:id"], "")
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("ROI name: id", captured_prompts[0])
        self.assertNotIn("ROI name: p702:r1:id", captured_prompts[0])
        self.assertEqual(debug["per_roi"][0]["name"], "p702:r1:id")
        self.assertEqual(debug["per_roi"][0]["roi_name"], "id")

    def test_postprocess_drops_unsupported_numeric_roi_name_echo(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {
                "text": '{"detected_text":"25"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"p11:r0:25": ""},
                roi_meta_by_name={
                    "p11:r0:25": {
                        "roi_name": "25",
                        "llm_field_data_type": "age",
                        "ocr_paddle_confidence": {"detected_text": "4"},
                    }
                },
            )

        self.assertEqual(out["p11:r0:25"], "")

    def test_postprocess_keeps_numeric_value_when_supported_by_ocr(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {
                "text": '{"detected_text":"25"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"p11:r0:25": "Age: 25"},
                roi_meta_by_name={
                    "p11:r0:25": {
                        "roi_name": "25",
                        "llm_field_data_type": "age",
                    }
                },
            )

        self.assertEqual(out["p11:r0:25"], "25")


if __name__ == "__main__":
    unittest.main()
