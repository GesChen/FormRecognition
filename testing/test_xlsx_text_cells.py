#!/usr/bin/env python3
"""Tests for explicit text typing in XLSX mapped cells."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from xlsx_data_entry import fill_row  # noqa: E402


class XlsxTextCellTests(unittest.TestCase):
    def test_force_text_cells_converts_values_and_sets_excel_text_type(self) -> None:
        wb = Workbook()
        ws = wb.active
        mapping = {
            "mappings": [
                {"type": "static", "column": "A", "value": 123},
                {"type": "static", "column": "B", "value": "=1+1"},
                {"type": "direct", "column": "C", "source": "score", "transform": "number"},
            ]
        }
        item = {"data": [{"name": "score", "text": "45.5"}]}

        fill_row(ws, 2, mapping, item, cfg={"force_text_cells": True})

        self.assertEqual([ws["A2"].value, ws["B2"].value, ws["C2"].value], ["123", "=1+1", "45.5"])
        for cell in (ws["A2"], ws["B2"], ws["C2"]):
            self.assertEqual(cell.data_type, "s")
            self.assertEqual(cell.number_format, "@")
        wb.close()

    def test_force_text_cells_can_be_disabled(self) -> None:
        wb = Workbook()
        ws = wb.active
        mapping = {"mappings": [{"type": "static", "column": "A", "value": 123}]}

        fill_row(ws, 2, mapping, {}, cfg={"force_text_cells": False})

        self.assertEqual(ws["A2"].value, 123)
        self.assertEqual(ws["A2"].data_type, "n")
        self.assertEqual(ws["A2"].number_format, "General")
        wb.close()


if __name__ == "__main__":
    unittest.main()
