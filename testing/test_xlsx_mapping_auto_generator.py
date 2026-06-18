"""Tests for XLSX mapping auto-generation support helpers."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import xlsx_mapping_auto_generator as auto


class XlsxMappingAutoGeneratorTests(unittest.TestCase):
    def test_collect_target_paddle_pair_tolerates_paddle_runtime_error(self):
        def fake_collect(image_path):
            if image_path.name == "6pre_a.png":
                raise RuntimeError("std::exception")
            return {
                "line_count": 1,
                "min_score": 0.99,
                "mean_score": 0.99,
                "lines": [{"text": "ok", "score": 0.99, "x": 1, "y": 2, "w": 3, "h": 4}],
            }

        with patch.object(auto, "_collect_document_structure", fake_collect):
            result = auto.collect_target_paddle_pair("2026", "6pre")

        side_a = result["sides"]["a"]["document_structure_from_paddleocr"]
        side_b = result["sides"]["b"]["document_structure_from_paddleocr"]
        self.assertEqual(side_a["line_count"], 0)
        self.assertEqual(side_a["ocr_error"]["error_type"], "RuntimeError")
        self.assertEqual(side_a["ocr_error"]["error"], "std::exception")
        self.assertEqual(side_b["line_count"], 1)
        self.assertEqual(result["ocr_warnings"][0]["side"], "a")

    def test_workbook_mapping_issues_rejects_id_only_mapping(self):
        target_xlsx = {
            "sheet_name": "Data",
            "suggested_start_row": 4,
            "columns": [
                {"column": "A", "header_values": [{"value": "Record ID"}]},
                {"column": "B", "header_values": [{"value": "Date of Test"}]},
                {"column": "C", "header_values": [{"value": "Q1"}]},
            ],
        }
        mapping = {
            "sheet": "Data",
            "start_row": 4,
            "mappings": [{"type": "direct", "source": "id", "column": "A"}],
        }

        issues = auto._workbook_mapping_issues(mapping, target_xlsx)

        self.assertTrue(any("covers 1 of 3 required workbook columns" in i["message"] for i in issues))
        self.assertTrue(any("B, C" in i["message"] for i in issues))

    def test_workbook_mapping_issues_ignores_optional_if_other_column(self):
        target_xlsx = {
            "sheet_name": "Data",
            "suggested_start_row": 4,
            "columns": [
                {"column": "A", "header_values": [{"value": "Record ID"}]},
                {"column": "B", "header_values": [{"value": "If Other, Please Describe"}]},
                {"column": "C", "header_values": [{"value": "Q1"}]},
            ],
        }
        mapping = {
            "sheet": "Data",
            "start_row": 4,
            "mappings": [
                {"type": "direct", "source": "id", "column": "A"},
                {"type": "direct", "source": "1", "column": "C"},
            ],
        }

        issues = auto._workbook_mapping_issues(mapping, target_xlsx)

        self.assertFalse(any("required workbook columns" in i["message"] for i in issues))


if __name__ == "__main__":
    unittest.main()
