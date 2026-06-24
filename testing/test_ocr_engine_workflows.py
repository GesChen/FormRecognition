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
    def _fake_vision_stage(self, path, *, timeout, min_conf, verbose, stage_index):
        return (
            {
                "stage_index": stage_index,
                "model": OCR_ENGINE.get("model", "qwen2.5vl"),
                "elapsed_sec": 0.02,
                "raw_response": {"response": '{"text":"vision text"}'},
                "response_text": '{"text":"vision text"}',
                "parsed": {
                    "detected_text": "vision text",
                },
                "accepted": True,
                "error": None,
                "image_payload": {"jpeg_bytes": 123},
            },
            {"jpeg_bytes": 123},
        )

    def _fake_paddle_stage(self, path, *, verbose, force_failure=False):
        if force_failure:
            return (
                {
                    "stage_index": 1,
                    "model": OCR_ENGINE.get("paddle_model_name", "paddle-ppocrv5"),
                    "elapsed_sec": 0.01,
                    "raw_response": None,
                    "response_text": "",
                    "parsed": {},
                    "accepted": False,
                    "error": "Forced Paddle failure for testing.",
                    "image_payload": {},
                },
                None,
            )
        raw_response = [{"res": {"rec_texts": ["paddle text"], "rec_scores": [0.72, 0.91]}}]
        return (
            {
                "stage_index": 1,
                "model": OCR_ENGINE.get("paddle_model_name", "paddle-ppocrv5"),
                "elapsed_sec": 0.01,
                "raw_response": raw_response,
                "response_text": "paddle text",
                "parsed": {
                    "detected_text": "paddle text",
                    "confidence_score": 0.72,
                    "confidence_label": "medium",
                    "needs_human_review": True,
                },
                "accepted": False,
                "error": None,
                "image_payload": {},
            },
            {
                "detected_text": "paddle text",
                "confidence_label": "medium",
                "confidence_score": 0.72,
                "needs_human_review": True,
                "self_evaluation": {"engine": "paddle"},
                "rec_texts": ["paddle text"],
                "rec_scores": [0.72, 0.91],
                "raw_response": raw_response,
            },
        )

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

    def test_vision_with_paddle_confidence_uses_vision_text_and_paddle_confidence(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image_path = Path(tmp.name)
            old_workflow = OCR_ENGINE.get("workflow_default")
            old_include_raw = OCR_ENGINE.get("include_paddle_confidence_raw")
            OCR_ENGINE["workflow_default"] = "vision_with_paddle_confidence"
            OCR_ENGINE["include_paddle_confidence_raw"] = False
            try:
                with patch.object(ocr_engine, "_run_vision_stage", self._fake_vision_stage):
                    with patch.object(ocr_engine, "_run_paddle_stage", self._fake_paddle_stage):
                        result = ocr_engine.ocr_raw(image_path)
            finally:
                OCR_ENGINE["workflow_default"] = old_workflow
                OCR_ENGINE["include_paddle_confidence_raw"] = old_include_raw

            self.assertEqual(result["detected_text"], "vision text")
            self.assertEqual(result["selected_model"], OCR_ENGINE.get("model", "qwen2.5vl"))
            self.assertEqual(result["selected_stage_index"], 0)
            self.assertEqual(result["text_source"], "vision")
            self.assertEqual(result["confidence_source"], "paddle")
            self.assertEqual(result["confidence_score"], 0.72)
            self.assertEqual(result["confidence_label"], "medium")
            self.assertEqual(result["rec_scores"], [0.72, 0.91])
            self.assertEqual(result["rec_texts"], ["paddle text"])
            self.assertTrue(result["needs_human_review"])
            self.assertNotIn("raw_response", result["paddle_confidence"])

    def test_vision_with_paddle_confidence_can_include_paddle_raw_debug(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image_path = Path(tmp.name)
            old_workflow = OCR_ENGINE.get("workflow_default")
            old_include_raw = OCR_ENGINE.get("include_paddle_confidence_raw")
            OCR_ENGINE["workflow_default"] = "vlm_paddle_confidence"
            OCR_ENGINE["include_paddle_confidence_raw"] = True
            try:
                with patch.object(ocr_engine, "_run_vision_stage", self._fake_vision_stage):
                    with patch.object(ocr_engine, "_run_paddle_stage", self._fake_paddle_stage):
                        result = ocr_engine.ocr_raw(image_path)
            finally:
                OCR_ENGINE["workflow_default"] = old_workflow
                OCR_ENGINE["include_paddle_confidence_raw"] = old_include_raw

            self.assertEqual(
                result["paddle_confidence"]["raw_response"],
                [{"res": {"rec_texts": ["paddle text"], "rec_scores": [0.72, 0.91]}}],
            )

    def test_vision_with_paddle_confidence_falls_back_to_vlm_confidence_when_paddle_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image_path = Path(tmp.name)
            old_workflow = OCR_ENGINE.get("workflow_default")
            OCR_ENGINE["workflow_default"] = "vision_with_paddle_confidence"
            try:
                with patch.object(ocr_engine, "_run_vision_stage", self._fake_vision_stage):
                    with patch.object(ocr_engine, "_run_paddle_stage", self._fake_paddle_stage):
                        result = ocr_engine.ocr_raw(image_path, force_paddle_failure=True)
            finally:
                OCR_ENGINE["workflow_default"] = old_workflow

            self.assertEqual(result["detected_text"], "vision text")
            self.assertEqual(result["confidence_source"], "fallback_vlm")
            self.assertEqual(result["confidence_score"], 1.0)
            self.assertFalse(result["needs_human_review"])
            self.assertEqual(result["paddle_confidence"]["error"], "Forced Paddle failure for testing.")

    def test_empty_parsed_detected_text_does_not_fall_back_to_json_wrapper(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image_path = Path(tmp.name)

            def fake_empty_vision_stage(path, *, timeout, min_conf, verbose, stage_index):
                return (
                    {
                        "stage_index": stage_index,
                        "model": OCR_ENGINE.get("model", "qwen2.5vl"),
                        "elapsed_sec": 0.02,
                        "raw_response": {"response": '```json { "text": "" }'},
                        "response_text": '```json { "text": "" }',
                        "parsed": {"detected_text": ""},
                        "accepted": True,
                        "error": None,
                        "image_payload": {"jpeg_bytes": 123},
                    },
                    {"jpeg_bytes": 123},
                )

            old_workflow = OCR_ENGINE.get("workflow_default")
            OCR_ENGINE["workflow_default"] = "vision_only"
            try:
                with patch.object(ocr_engine, "_run_vision_stage", fake_empty_vision_stage):
                    result = ocr_engine.ocr_raw(image_path)
            finally:
                OCR_ENGINE["workflow_default"] = old_workflow

            self.assertEqual(result["detected_text"], "")

    def test_paddle_vlm_fusion_dispatch_runs_both_engines(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image_path = Path(tmp.name)
            old_workflow = OCR_ENGINE.get("workflow_default")
            OCR_ENGINE["workflow_default"] = "paddle_vlm_fusion"
            try:
                with patch.object(ocr_engine, "_run_vision_stage", self._fake_vision_stage):
                    with patch.object(ocr_engine, "_run_paddle_stage", self._fake_paddle_stage):
                        result = ocr_engine.ocr_raw(image_path)
            finally:
                OCR_ENGINE["workflow_default"] = old_workflow

            self.assertEqual(result["detected_text"], "vision text")
            self.assertEqual(result["workflow"], "paddle_vlm_fusion")
            self.assertEqual(result["text_source"], "vision")
            self.assertEqual(result["paddle_confidence"]["detected_text"], "paddle text")
            self.assertEqual(
                result["fusion"]["normalizer_inputs"],
                ["vlm_detected_text", "paddle_confidence.detected_text"],
            )

    def test_confidence_stats_preserve_source_and_paddle_sidecar(self):
        raw = {
            "detected_text": "vision text",
            "confidence_score": 0.72,
            "confidence_source": "paddle",
            "text_source": "vision",
            "selected_model": "glm-ocr:latest",
            "selected_stage_index": 0,
            "rec_scores": [0.72, 0.91],
            "rec_texts": ["paddle text"],
            "paddle_confidence": {
                "confidence_score": 0.72,
                "raw_response": [{"res": {"rec_scores": [0.72, 0.91]}}],
            },
        }

        stats = ocr_engine.ocr_confidence_stats(raw)

        self.assertEqual(stats["confidence_score"], 0.72)
        self.assertEqual(stats["confidence_source"], "paddle")
        self.assertEqual(stats["text_source"], "vision")
        self.assertEqual(stats["paddle_confidence"]["raw_response"], [{"res": {"rec_scores": [0.72, 0.91]}}])


if __name__ == "__main__":
    unittest.main()
