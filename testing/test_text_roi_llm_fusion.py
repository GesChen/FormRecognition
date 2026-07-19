"""Tests for Paddle/VLM fusion inputs in text ROI normalization."""

import json
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
                            "<input_text>\n"
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

    def test_roi_without_postprocess_passes_or_legacy_override_does_not_call_llm(self):
        def fail_generate(*args, **kwargs):
            raise AssertionError("generate should not be called")

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fail_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"field": "  raw   text  "},
                roi_meta_by_name={
                    "field": {
                        "llm_field_data_type": "text",
                        "llm_validation_rules": "Return text or null.",
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["field"], "raw text")
        self.assertEqual(debug["per_roi"][0]["llm_pass_count"], 0)
        self.assertEqual(debug["per_roi"][0]["llm_prompt_source"], "none")

    def test_postprocess_passes_run_to_completion_before_next_roi(self):
        calls: list[tuple[str, str]] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            if "first raw-a" in prompt:
                calls.append(("a", prompt))
                return {"text": '{"detected_text":"A1"}', "elapsed": 0.01}
            if "second A1" in prompt:
                calls.append(("b", prompt))
                return {"text": '{"detected_text":"A2"}', "elapsed": 0.01}
            if "only raw-b" in prompt:
                calls.append(("c", prompt))
                return {"text": '{"detected_text":"B1"}', "elapsed": 0.01}
            raise AssertionError(f"unexpected prompt: {prompt}")

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"a": "raw-a", "b": "raw-b"},
                roi_meta_by_name={
                    "a": {"postprocess_passes": ["first <input_text>", "second <input_text>"]},
                    "b": {"postprocess_passes": ["only <input_text>"]},
                },
                debug_out=debug,
            )

        self.assertEqual(out, {"a": "A2", "b": "B1"})
        self.assertEqual([name for name, _prompt in calls], ["a", "b", "c"])
        self.assertIn("first raw-a", calls[0][1])
        self.assertIn("second A1", calls[1][1])
        self.assertIn("only raw-b", calls[2][1])
        self.assertEqual(debug["per_roi"][0]["llm_pass_count"], 2)

    def test_llm_pass_model_override_uses_pass_model(self):
        calls: list[tuple[str, str]] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            calls.append((prompt, model or ""))
            if model == "llama3.2:latest":
                return {"text": '{"detected_text":"first"}', "elapsed": 0.01}
            if model == "qwen3.5:9b":
                return {"text": '{"detected_text":"second"}', "elapsed": 0.01}
            raise AssertionError(f"unexpected model: {model}")

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"field": "raw"},
                roi_meta_by_name={
                    "field": {
                        "llm_model": "qwen3.5:9b",
                        "postprocess_passes": [
                            {
                                "type": "llm",
                                "model": "llama3.2:latest",
                                "prompt": "First <input_text>.",
                            },
                            {"type": "llm", "prompt": "Second <input_text>."},
                        ],
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["field"], "second")
        self.assertEqual(
            calls,
            [("First raw.", "llama3.2:latest"), ("Second first.", "qwen3.5:9b")],
        )
        self.assertEqual(debug["per_roi"][0]["passes"][0]["model"], "llama3.2:latest")
        self.assertEqual(debug["per_roi"][0]["passes"][1]["model"], "qwen3.5:9b")
        self.assertEqual(debug["per_roi"][0]["passes"][0]["requested_model"], "llama3.2:latest")
        self.assertEqual(debug["per_roi"][0]["passes"][0]["fallback_model"], "qwen3.5:9b")
        self.assertEqual(debug["per_roi"][0]["passes"][0]["model_source"], "pass")
        self.assertIsNone(debug["per_roi"][0]["passes"][0]["model_fallback_reason"])
        self.assertIsNone(debug["per_roi"][0]["passes"][1]["requested_model"])
        self.assertEqual(debug["per_roi"][0]["passes"][1]["model_source"], "fallback")
        self.assertEqual(debug["per_roi"][0]["passes"][1]["model_fallback_reason"], "no_pass_model")
        self.assertEqual(debug["per_roi"][0]["attempts"][0]["requested_model"], "llama3.2:latest")
        self.assertEqual(debug["per_roi"][0]["attempts"][0]["model_source"], "pass")

    def test_deferred_postprocess_groups_rois_by_first_llm_model(self):
        calls: list[tuple[str, str]] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            calls.append((prompt, model or ""))
            if "raw-a" in prompt:
                return {"text": '{"detected_text":"A"}', "elapsed": 0.01}
            if "raw-b" in prompt:
                return {"text": '{"detected_text":"B"}', "elapsed": 0.01}
            if "raw-c" in prompt:
                return {"text": '{"detected_text":"C"}', "elapsed": 0.01}
            raise AssertionError(f"unexpected prompt: {prompt}")

        debug: dict = {}
        with patch.object(text_roi_llm, "_supported_llm_models", return_value={"model-a", "model-b"}):
            with patch.object(text_roi_llm, "generate", fake_generate):
                out = text_roi_llm.postprocess_text_rois(
                    {"a": "raw-a", "b": "raw-b", "c": "raw-c"},
                    roi_meta_by_name={
                        "a": {
                            "postprocess_passes": [
                                {"type": "llm", "model": "model-a", "prompt": "First <input_text>."},
                            ],
                        },
                        "b": {
                            "postprocess_passes": [
                                {"type": "llm", "model": "model-b", "prompt": "First <input_text>."},
                            ],
                        },
                        "c": {
                            "postprocess_passes": [
                                {"type": "code", "code": "return input_text"},
                                {"type": "llm", "model": "model-a", "prompt": "First <input_text>."},
                            ],
                        },
                    },
                    debug_out=debug,
                )

        self.assertEqual(out, {"a": "A", "b": "B", "c": "C"})
        self.assertEqual(
            calls,
            [
                ("First raw-a.", "model-a"),
                ("First raw-c.", "model-a"),
                ("First raw-b.", "model-b"),
            ],
        )
        self.assertEqual(
            debug["call_batches"],
            [
                {"model": "model-a", "roi_names": ["a", "c"], "roi_count": 2},
                {"model": "model-b", "roi_names": ["b"], "roi_count": 1},
            ],
        )
        self.assertEqual([row["name"] for row in debug["per_roi"]], ["a", "c", "b"])

    def test_unsupported_llm_pass_model_override_falls_back_with_debug(self):
        calls: list[tuple[str, str]] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            calls.append((prompt, model or ""))
            return {"text": '{"detected_text":"fallback"}', "elapsed": 0.01}

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"field": "raw"},
                roi_meta_by_name={
                    "field": {
                        "llm_model": "qwen3.5:9b",
                        "postprocess_passes": [
                            {
                                "type": "llm",
                                "model": "not-installed:latest",
                                "prompt": "Return JSON for <input_text>.",
                            },
                        ],
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["field"], "fallback")
        self.assertEqual(calls, [("Return JSON for raw.", "qwen3.5:9b")])
        pass_debug = debug["per_roi"][0]["passes"][0]
        attempt_debug = debug["per_roi"][0]["attempts"][0]
        self.assertEqual(pass_debug["requested_model"], "not-installed:latest")
        self.assertEqual(pass_debug["fallback_model"], "qwen3.5:9b")
        self.assertEqual(pass_debug["model"], "qwen3.5:9b")
        self.assertEqual(pass_debug["model_source"], "fallback")
        self.assertEqual(pass_debug["model_fallback_reason"], "unsupported_pass_model")
        self.assertEqual(attempt_debug["requested_model"], "not-installed:latest")
        self.assertEqual(attempt_debug["model_fallback_reason"], "unsupported_pass_model")

    def test_code_pass_receives_input_text_and_chains_to_llm(self):
        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {"text": '{"detected_text":"12"}', "elapsed": 0.01}

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"age": "Age: 12"},
                roi_meta_by_name={
                    "age": {
                        "postprocess_passes": [
                            {"type": "code", "code": "return input_text.replace('Age:', '').strip()"},
                            {"type": "llm", "prompt": "Return JSON for <input_text>."},
                        ]
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["age"], "12")
        self.assertEqual(captured_prompts, ["Return JSON for 12."])
        self.assertEqual(debug["per_roi"][0]["postprocess_pass_count"], 2)
        self.assertEqual(debug["per_roi"][0]["passes"][0]["type"], "code")
        self.assertTrue(debug["per_roi"][0]["passes"][0]["ok"])

    def test_code_pass_timeout_keeps_current_value(self):
        debug: dict = {}

        with patch.object(text_roi_llm, "_code_timeout_sec", return_value=0.1):
            out = text_roi_llm.postprocess_text_rois(
                {"field": "raw"},
                roi_meta_by_name={
                    "field": {
                        "postprocess_passes": [
                            {"type": "code", "code": "while True:\n    pass"},
                        ]
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["field"], "raw")
        self.assertFalse(debug["per_roi"][0]["passes"][0]["ok"])
        self.assertEqual(debug["per_roi"][0]["passes"][0]["details"]["error"], "code_pass_timeout")

    def test_empty_postprocess_passes_suppresses_legacy_prompt_override(self):
        def fail_generate(*args, **kwargs):
            raise AssertionError("generate should not be called")

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fail_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"field": "raw"},
                roi_meta_by_name={
                    "field": {
                        "postprocess_passes": [],
                        "llm_prompt_override": "Legacy prompt should not run.",
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["field"], "raw")
        self.assertEqual(debug["per_roi"][0]["llm_prompt_source"], "postprocess_passes")
        self.assertEqual(debug["per_roi"][0]["llm_pass_count"], 0)

    def test_bare_json_null_from_llm_clears_value(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {"text": "null", "elapsed": 0.01}

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"field": "raw text"},
                roi_meta_by_name={"field": {"postprocess_passes": ["Return JSON for <input_text>."]}},
                debug_out=debug,
            )

        self.assertEqual(out["field"], "")
        self.assertTrue(debug["per_roi"][0]["attempts"][0]["parsed_ok"])

    def test_output_regex_mismatch_clears_llm_candidate(self):
        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            return {"text": '{"date":"04-79-2026"}', "elapsed": 0.01}

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"date": "04/79/2026"},
                roi_meta_by_name={
                    "date": {
                        "output_regex": r"^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-(20[0-9]{2})$",
                        "postprocess_passes": ["Return JSON for <input_text>."],
                    }
                },
                debug_out=debug,
            )

        self.assertEqual(out["date"], "")
        self.assertEqual(
            debug["per_roi"][0]["attempts"][0]["rejection_reason"],
            "output_regex_mismatch",
        )

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
                        "postprocess_passes": ["Return JSON only for <input_text>."],
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
                        "postprocess_passes": ["Return JSON only for <input_text>."],
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
                            "postprocess_passes": ["Return JSON only for <input_text>."],
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
                    "date": {
                        "postprocess_passes": ["Return JSON only for <input_text>."],
                        "ocr_paddle_confidence": {"detected_text": None},
                    }
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
            out = text_roi_llm.postprocess_text_rois(
                {"age": "unclear"},
                roi_meta_by_name={"age": {"postprocess_passes": ["Return JSON only for <input_text>."]}},
            )

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
            out = text_roi_llm.postprocess_text_rois(
                {"date": "Today's Date: 9/27/2016"},
                roi_meta_by_name={"date": {"postprocess_passes": ["Return JSON only for <input_text>."]}},
                debug_out=debug,
            )

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
            out = text_roi_llm.postprocess_text_rois(
                {"age": "What is your age? 12"},
                roi_meta_by_name={"age": {"postprocess_passes": ["Return JSON only for <input_text>."]}},
            )

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
            out = text_roi_llm.postprocess_text_rois(
                {"date": "Today's Date: 4/27/26"},
                roi_meta_by_name={"date": {"postprocess_passes": ["Return JSON only for <input_text>."]}},
            )

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
                        "postprocess_passes": ["Return JSON only for <input_text>."],
                    }
                },
            )

        self.assertEqual(out["p11:r0:25"], "25")

    def test_migrated_2026_schema_llm_pass_runs(self):
        schema = json.loads((ROOT / "data" / "roi_schemas" / "2026" / "6post_a.json").read_text())
        roi = next(r for r in schema["rois"] if r.get("name") == "id")
        self.assertNotIn("llm_prompt_override", roi)
        self.assertEqual(len(roi.get("postprocess_passes") or []), 1)
        self.assertEqual(roi["postprocess_passes"][0]["type"], "llm")

        captured_prompts: list[str] = []

        def fake_generate(prompt, *, model=None, timeout=None, extra_params=None):
            captured_prompts.append(prompt)
            return {
                "text": '{"detected_text":"7011606A"}',
                "elapsed": 0.01,
                "eval_count": 1,
                "eval_duration": 1,
            }

        debug: dict = {}
        with patch.object(text_roi_llm, "generate", fake_generate):
            out = text_roi_llm.postprocess_text_rois(
                {"id": "Record ID: 7011606A"},
                roi_meta_by_name={"id": roi},
                debug_out=debug,
            )

        self.assertEqual(out["id"], "7011606A")
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn('OCR text to analyze:\n"Record ID: 7011606A"', captured_prompts[0])
        self.assertEqual(debug["per_roi"][0]["llm_prompt_source"], "postprocess_passes")
        self.assertEqual(debug["per_roi"][0]["llm_pass_count"], 1)


if __name__ == "__main__":
    unittest.main()
