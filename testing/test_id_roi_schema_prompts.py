"""Schema checks for ID ROI LLM prompt safety."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCHEMA_ROOT = ROOT / "data" / "roi_schemas"


class IdRoiSchemaPromptTests(unittest.TestCase):
    def test_all_id_rois_have_strict_extractor_prompt_and_regex(self):
        id_rois: list[tuple[Path, dict]] = []
        for path in sorted(SCHEMA_ROOT.glob("*/*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rois = data.get("rois") if isinstance(data, dict) else None
            if not isinstance(rois, list):
                continue
            for roi in rois:
                if isinstance(roi, dict) and roi.get("name") == "id":
                    id_rois.append((path, roi))

        self.assertGreater(len(id_rois), 0)
        for path, roi in id_rois:
            with self.subTest(path=str(path)):
                prompt = str(roi.get("llm_prompt_override") or "")
                self.assertEqual(roi.get("llm_field_data_type"), "id")
                self.assertEqual(roi.get("ocr_output_regex"), r"^[0-9]{7}[AB]$")
                self.assertIn("7 digits followed by A or B", str(roi.get("llm_validation_rules") or ""))
                self.assertIn("Record ID", prompt)
                self.assertIn("p702:r1:id", prompt)
                self.assertIn("Do not repair OCR mistakes", prompt)
                self.assertIn("^[0-9]{7}[AB]$", prompt)


class AgeRoiSchemaPromptTests(unittest.TestCase):
    def test_2026_age_rois_have_strict_prompt_and_regex(self):
        age_sources = {
            "6pre_b.json": "21",
            "7pre_b.json": "25",
            "8pre_b.json": "25",
            "6post_b.json": "27",
            "7post_b.json": "31",
            "8post_b.json": "31",
        }

        for filename, source in age_sources.items():
            path = SCHEMA_ROOT / "2026" / filename
            data = json.loads(path.read_text(encoding="utf-8"))
            roi = next(
                (
                    row
                    for row in data.get("rois", [])
                    if isinstance(row, dict) and row.get("name") == source
                ),
                None,
            )
            with self.subTest(path=str(path), source=source):
                self.assertIsNotNone(roi)
                assert roi is not None
                prompt = str(roi.get("llm_prompt_override") or "")
                self.assertEqual(roi.get("llm_field_data_type"), "age")
                self.assertEqual(roi.get("ocr_output_regex"), r"^[0-9]{1,2}$")
                self.assertIn("Do not use the ROI/question number as the age", prompt)
                self.assertIn("one- or two-digit age", prompt)


class DateRoiSchemaPromptTests(unittest.TestCase):
    def test_all_date_rois_have_strict_prompt_and_regex(self):
        date_rois: list[tuple[Path, dict]] = []
        for path in sorted(SCHEMA_ROOT.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rois = data.get("rois") if isinstance(data, dict) else None
            if not isinstance(rois, list):
                continue
            for roi in rois:
                if isinstance(roi, dict) and roi.get("name") == "date":
                    date_rois.append((path, roi))

        self.assertGreater(len(date_rois), 0)
        for path, roi in date_rois:
            with self.subTest(path=str(path)):
                prompt = str(roi.get("llm_prompt_override") or "")
                self.assertEqual(roi.get("llm_field_data_type"), "date")
                self.assertEqual(
                    roi.get("ocr_output_regex"),
                    r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(20[0-9]{2})$",
                )
                self.assertIn("strict date extractor", prompt)
                self.assertIn("MM/DD/YYYY", prompt)
                self.assertIn("must match this regex exactly", prompt)
                self.assertIn("return null", prompt)


if __name__ == "__main__":
    unittest.main()
