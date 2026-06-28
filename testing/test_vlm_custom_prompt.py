import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import ocr_engine
import roi_page_module
from roi_page_module import PageRoiSchema, TextRoi, recognize_text_fields


class VlmCustomPromptTests(unittest.TestCase):
    def test_build_vlm_roi_prompt_uses_query_without_implicit_colon(self):
        self.assertEqual(
            ocr_engine.build_vlm_roi_prompt("Todays Date"),
            '请按下列JSON格式输 出图中信息: {"Todays Date":""}',
        )

    def test_build_vlm_roi_prompt_blank_returns_none(self):
        self.assertIsNone(ocr_engine.build_vlm_roi_prompt(""))
        self.assertIsNone(ocr_engine.build_vlm_roi_prompt("   "))
        self.assertIsNone(ocr_engine.build_vlm_roi_prompt(None))

    def test_roi_output_regex_respects_config_toggle(self):
        meta = {"ocr_output_regex": r"^\d+$"}
        with patch.dict(roi_page_module.ROI_PAGE_RECOGNITION, {"ocr_regex_check_enabled": True}, clear=False):
            self.assertEqual(roi_page_module._roi_output_regex(meta), r"^\d+$")
        with patch.dict(roi_page_module.ROI_PAGE_RECOGNITION, {"ocr_regex_check_enabled": False}, clear=False):
            self.assertIsNone(roi_page_module._roi_output_regex(meta))

    def test_custom_key_json_becomes_detected_text(self):
        parsed = ocr_engine._json_from_text('{"Todays Date":"05/01/2026"}')
        self.assertEqual(parsed["detected_text"], "05/01/2026")

    def test_messy_fenced_custom_key_json_becomes_detected_text(self):
        text = 'Here is the answer:\n```json\n{"Todays Date":"05/01/2026"}\n```\nextra'
        parsed = ocr_engine._json_from_text(text)
        self.assertEqual(parsed["detected_text"], "05/01/2026")

    def test_malformed_custom_key_json_with_missing_colon_becomes_detected_text(self):
        text = '```json\n{"Record ID: "8011576A"}\n```'
        parsed = ocr_engine._json_from_text(text)
        self.assertEqual(parsed["detected_text"], "8011576A")

    def test_malformed_custom_key_json_variants_become_detected_text(self):
        examples = [
            ('{"Record ID" "8011576A"}', "8011576A"),
            ('{"Record ID":8011576A}', "8011576A"),
            ("Record ID: 8011576A", "8011576A"),
            ("```json\n{'Record ID':'8011576A'}\n```", "8011576A"),
        ]
        for text, expected in examples:
            with self.subTest(text=text):
                parsed = ocr_engine._json_from_text(text)
                self.assertEqual(parsed["detected_text"], expected)

    def test_stream_stop_accepts_custom_key_json(self):
        stop = ocr_engine._stream_stop_completion('prefix {"Todays Date":"05/01/2026"} trailing')
        self.assertIsNotNone(stop)
        self.assertEqual(stop["stop_reason"], "json_completion")

    def test_stream_stop_accepts_malformed_custom_key_json(self):
        stop = ocr_engine._stream_stop_completion('prefix {"Record ID: "8011576A"} trailing')
        self.assertIsNotNone(stop)
        self.assertEqual(stop["stop_reason"], "json_completion")
        self.assertEqual(stop["mode"], "malformed_json_object")

    def test_ocr_raw_prompt_override_reaches_vision_call(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            Image.new("L", (8, 8), 255).save(tmp.name)
            with patch.dict(ocr_engine.OCR_ENGINE, {"workflow_default": "vision_only", "stream": False}, clear=False):
                with patch.object(
                    ocr_engine,
                    "_local_vision_call",
                    return_value=('{"Todays Date":"05/01/2026"}', {"response": '{"Todays Date":"05/01/2026"}'}),
                ) as call:
                    result = ocr_engine.ocr_raw(tmp.name, prompt_override="CUSTOM PROMPT")

        self.assertEqual(result["detected_text"], "05/01/2026")
        self.assertEqual(call.call_args.kwargs["prompt"], "CUSTOM PROMPT")

    def test_ocr_raw_without_override_uses_default_vlm_prompt(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            Image.new("L", (8, 8), 255).save(tmp.name)
            with patch.dict(ocr_engine.OCR_ENGINE, {"workflow_default": "vision_only", "stream": False}, clear=False):
                with patch.object(
                    ocr_engine,
                    "_local_vision_call",
                    return_value=('{"text":"abc"}', {"response": '{"text":"abc"}'}),
                ) as call:
                    result = ocr_engine.ocr_raw(tmp.name)

        self.assertEqual(result["detected_text"], "abc")
        self.assertEqual(call.call_args.kwargs["prompt"], ocr_engine._TEMP_TEXT_VLM_PROMPT)

    def test_ocr_raw_empty_json_value_returns_empty_text(self):
        raw_text = '```json\n{"Record ID":""}\n```'
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            Image.new("L", (8, 8), 255).save(tmp.name)
            with patch.dict(ocr_engine.OCR_ENGINE, {"workflow_default": "vision_only", "stream": False}, clear=False):
                with patch.object(
                    ocr_engine,
                    "_local_vision_call",
                    return_value=(raw_text, {"response": raw_text}),
                ):
                    result = ocr_engine.ocr_raw(tmp.name)

        self.assertEqual(result["detected_text"], "")

    def test_roi_vlm_query_passes_prompt_override_only_for_that_roi(self):
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "page.png"
            Image.new("L", (30, 10), 255).save(img_path)
            schema = PageRoiSchema(
                form_type="8pre",
                side="a",
                schema_path=Path(td) / "schema.json",
                text_rois=[
                    TextRoi(name="date", x=0, y=0, w=10, h=10, meta={"vlm_query": "Todays Date"}),
                    TextRoi(name="name", x=10, y=0, w=10, h=10, meta={}),
                ],
                mcq_rois=[],
                raw_schema={"image_width": 30, "image_height": 10},
            )

            calls = []

            def fake_ocr_raw(path, **kwargs):
                calls.append(kwargs)
                return {"detected_text": "x", "confidence_score": 1.0}

            with patch("ocr_engine.ocr_raw", side_effect=fake_ocr_raw):
                results = recognize_text_fields(
                    img_path,
                    schema,
                    debug_collector={},
                    apply_llm_postprocess=False,
                )

        self.assertEqual(results, {"date": "x", "name": "x"})
        self.assertEqual(
            calls[0]["prompt_override"],
            '请按下列JSON格式输 出图中信息: {"Todays Date":""}',
        )
        self.assertEqual(calls[1], {})

    def test_text_roi_expand_px_expands_crop_before_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "page.png"
            Image.new("L", (30, 20), 255).save(img_path)
            schema = PageRoiSchema(
                form_type="8pre",
                side="a",
                schema_path=Path(td) / "schema.json",
                text_rois=[
                    TextRoi(name="date", x=5, y=5, w=10, h=6, meta={}),
                ],
                mcq_rois=[],
                raw_schema={"image_width": 30, "image_height": 20},
            )

            crop_sizes = []

            def fake_ocr_raw(path):
                with Image.open(path) as crop:
                    crop_sizes.append(crop.size)
                return {"detected_text": "x", "confidence_score": 1.0}

            retry_meta = {}
            with patch.dict(roi_page_module.ROI_PAGE_RECOGNITION, {"text_roi_expand_px": 3}, clear=False):
                with patch("ocr_engine.ocr_raw", side_effect=fake_ocr_raw):
                    results = recognize_text_fields(
                        img_path,
                        schema,
                        ocr_retry_meta_out=retry_meta,
                        apply_llm_postprocess=False,
                    )

        self.assertEqual(results, {"date": "x"})
        self.assertEqual(crop_sizes, [(16, 12)])
        self.assertEqual(retry_meta["date"]["bbox_xyxy"], [2, 2, 18, 14])
        self.assertEqual(retry_meta["date"]["text_roi_expand_px"], 3)

    def test_text_roi_horizontal_line_suppression_cleans_crop_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "page.png"
            page = Image.new("L", (120, 30), 255)
            for x in range(5, 115):
                page.putpixel((x, 15), 0)
            page.save(img_path)
            schema = PageRoiSchema(
                form_type="8pre",
                side="a",
                schema_path=Path(td) / "schema.json",
                text_rois=[TextRoi(name="date", x=0, y=0, w=120, h=30, meta={})],
                mcq_rois=[],
                raw_schema={"image_width": 120, "image_height": 30},
            )

            cleaned_rows = []

            def fake_ocr_raw(path):
                with Image.open(path) as crop:
                    cleaned_rows.append(crop.convert("L").crop((5, 15, 115, 16)).getextrema())
                return {"detected_text": "x", "confidence_score": 1.0}

            retry_meta = {}
            debug = {}
            line_cfg = {
                "text_roi_expand_px": 0,
                "text_roi_horizontal_line_suppression_enabled": True,
                "text_roi_horizontal_line_min_len": 80,
                "text_roi_horizontal_line_thickness": 2,
                "text_roi_horizontal_line_inpaint_radius": 3,
            }
            with patch.dict(roi_page_module.ROI_PAGE_RECOGNITION, line_cfg, clear=False):
                with patch("ocr_engine.ocr_raw", side_effect=fake_ocr_raw):
                    recognize_text_fields(
                        img_path,
                        schema,
                        debug_collector=debug,
                        ocr_retry_meta_out=retry_meta,
                        apply_llm_postprocess=False,
                    )

        self.assertGreater(cleaned_rows[0][0], 200)
        cleanup = retry_meta["date"]["horizontal_line_cleanup"]
        self.assertTrue(cleanup["enabled"])
        self.assertGreater(cleanup["removed_pixels"], 0)
        self.assertEqual(debug["text_per_roi"]["date"]["horizontal_line_cleanup"], cleanup)


if __name__ == "__main__":
    unittest.main()
