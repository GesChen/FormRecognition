"""Tests for Paddle/VLM fusion inputs in text ROI normalization."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import text_roi_llm


class TextRoiLlmFusionTests(unittest.TestCase):
    def test_prompt_override_can_embed_ocr_text_with_placeholder(self):
        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {
                "text": '{"detected_text":"trimmed"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            text_roi_llm.postprocess_text_rois(
                {"field": "trimmed"},
                roi_meta_by_name={
                    "field": {
                        "llm_prompt_override": (
                            "Return JSON only.\n"
                            "Input appears here:\n"
                            "<ocr_text>\n"
                            "Remove header text if present."
                        ),
                    }
                },
            )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertIn("Input appears here:\ntrimmed\nRemove header text if present.", prompt)
        self.assertNotIn("OCR extracted text:\ntrimmed\n", prompt)
        self.assertNotIn("You are a strict text normalizer for one OCR ROI field.", prompt)
        self.assertNotIn("ROI name:", prompt)
        self.assertNotIn("Field data type:", prompt)
        self.assertNotIn("Validation rules:", prompt)
        self.assertNotIn("Creator instruction:", prompt)

    def test_prompt_override_without_placeholder_is_used_exactly(self):
        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {
                "text": '{"detected_text":"trimmed"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            text_roi_llm.postprocess_text_rois(
                {"field": "trimmed"},
                roi_meta_by_name={
                    "field": {
                        "llm_prompt_override": "Return JSON only. Remove header text if present.",
                    }
                },
            )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertEqual(prompt, "Return JSON only. Remove header text if present.")
        self.assertNotIn("OCR extracted text:\ntrimmed\n", prompt)
        self.assertNotIn("You are a strict text normalizer for one OCR ROI field.", prompt)
        self.assertNotIn("ROI name:", prompt)
        self.assertNotIn("Field data type:", prompt)
        self.assertNotIn("Validation rules:", prompt)
        self.assertNotIn("Creator instruction:", prompt)

    def test_default_text_roi_prompt_keeps_generic_context(self):
        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {
                "text": '{"detected_text":"trimmed"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            text_roi_llm.postprocess_text_rois(
                {"field": "trimmed"},
                roi_meta_by_name={
                    "field": {
                        "llm_field_data_type": "text",
                        "llm_validation_rules": "Return text or null.",
                    }
                },
            )

        self.assertEqual(len(captured_prompts), 1)
        prompt = captured_prompts[0]
        self.assertIn("You are a strict text normalizer for one OCR ROI field.", prompt)
        self.assertIn("ROI name: field", prompt)
        self.assertIn("Field data type: text", prompt)
        self.assertIn("Validation rules: Return text or null.", prompt)

    def test_postprocess_preserves_paddle_sidecar_in_debug_without_prompting_with_it(self):
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
        self.assertNotIn("Paddle/VLM fusion inputs:", prompt)
        self.assertNotIn("PaddleOCR extracted text: 1234567A", prompt)
        self.assertFalse(debug["per_roi"][0]["used_paddle_vlm_fusion"])
        self.assertTrue(debug["per_roi"][0]["preserved_ocr_paddle_confidence"])

    def test_postprocess_allows_null_after_fusion_has_already_run(self):
        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {"text": '{"detected_text":null}', "elapsed": 0.01}

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"date": "4/13/26"},
                roi_meta_by_name={
                    "date": {
                        "roi_name": "date",
                        "ocr_paddle_confidence": {"detected_text": None},
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["date"], "")
        self.assertEqual(len(captured_prompts), 1)
        self.assertNotIn("PaddleOCR extracted text:", captured_prompts[0])
        self.assertIsNone(debug["per_roi"][0]["attempts"][0]["rejection_reason"])

    def test_postprocess_does_not_use_paddle_sidecar_as_supported_fallback(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {"text": '{"detected_text":null}', "elapsed": 0.01}

        with patch.object(text_roi_llm, "generate", fake_generate):
            with patch.object(text_roi_llm, "_max_reruns", return_value=1):
                out = text_roi_llm.postprocess_text_rois(
                    {"date": ""},
                    roi_meta_by_name={
                        "date": {
                            "ocr_paddle_confidence": {"detected_text": "4/13/26"},
                        }
                    },
                )

        self.assertEqual(out["date"], "")

    def test_fusion_allows_null_when_both_sources_are_empty(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {"text": '{"detected_text":null}', "elapsed": 0.01}

        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"date": ""},
                roi_meta_by_name={
                    "date": {"ocr_paddle_confidence": {"detected_text": None}}
                },
            )

        self.assertEqual(out["date"], "")

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

    def test_postprocess_accepts_single_key_json_with_any_key(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {
                "text": '{"date":"09-27-2016"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois({"date": "Today's Date: 9/27/2016"}, debug_out=debug)

        self.assertEqual(out["date"], "09-27-2016")
        self.assertTrue(debug["per_roi"][0]["attempts"][0]["parsed_ok"])

    def test_postprocess_coerces_single_key_numeric_value(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {
                "text": '{"age":12}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois({"age": "What is your age? 12"})

        self.assertEqual(out["age"], "12")

    def test_postprocess_accepts_messy_single_key_json(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {
                "text": 'answer follows:\n```json\nnoise before {"date":"04-27-2026"} trailing\n```',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois({"date": "Today's Date: 4/27/26"})

        self.assertEqual(out["date"], "04-27-2026")

    def test_global_llm_postprocess_toggle_returns_original_text(self):
        def fail_generate(*args, **kwargs):
            raise AssertionError("generate should not be called")

        debug: dict = {}
        with patch.dict(text_roi_llm.LLM_POSTPROCESS, {"enabled": False}, clear=False):
            with patch.object(text_roi_llm, "generate", fail_generate):
                out = text_roi_llm.postprocess_text_rois({"age": "  14  "}, debug_out=debug)

        self.assertEqual(out, {"age": "14"})
        self.assertFalse(debug["enabled"])
        self.assertEqual(debug["skipped"], "disabled")

    def test_postprocess_uses_roi_name_metadata_without_injecting_it_into_override_prompt(self):
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
        self.assertEqual(
            captured_prompts[0],
            "Map the OCR JSON field id into detected_text.",
        )
        self.assertEqual(debug["per_roi"][0]["name"], "p702:r1:id")
        self.assertEqual(debug["per_roi"][0]["roi_name"], "id")

    def test_postprocess_replaces_unsupported_numeric_echo_with_empty(self):
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
