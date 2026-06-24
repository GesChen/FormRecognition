"""Tests for original PDF/page metadata on merged PDF recognition output."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import pdf_recognize


class PdfSourceManifestTests(unittest.TestCase):
    def test_detect_form_types_infers_null_from_last_known_type(self):
        page_paths = [
            ROOT / "output/cache/test/page_0001.png",
            ROOT / "output/cache/test/page_0002.png",
            ROOT / "output/cache/test/page_0003.png",
            ROOT / "output/cache/test/page_0004.png",
            ROOT / "output/cache/test/page_0005.png",
            ROOT / "output/cache/test/page_0006.png",
        ]

        def fake_batch(paths, **kwargs):
            return [
                {"form_type": "8post"},
                {"form_type": None},
                {"form_type": "6pre"},
            ]

        with patch.object(pdf_recognize, "_get_extract_id_form_batch", return_value=fake_batch):
            debug: dict = {}
            page_infos = pdf_recognize._detect_ids_and_form_types(page_paths, verbose=False, debug_out=debug)

        self.assertEqual([page_infos[i]["form_type"] for i in range(0, 6, 2)], ["8post", "8post", "6pre"])
        self.assertFalse(page_infos[0]["form_type_inferred"])
        self.assertTrue(page_infos[2]["form_type_inferred"])
        self.assertEqual(page_infos[2]["form_type_inference_source_pair"], 0)
        self.assertEqual(Path(page_infos[2]["schema_path"]).name, "8post_a.json")
        self.assertEqual(debug["form_type_inference"]["inferred_pair_count"], 1)
        self.assertEqual(debug["form_type_inference"]["unresolved_pair_count"], 0)

    def test_detect_form_types_leaves_leading_null_without_last_known_type(self):
        page_paths = [
            ROOT / "output/cache/test/page_0001.png",
            ROOT / "output/cache/test/page_0002.png",
            ROOT / "output/cache/test/page_0003.png",
            ROOT / "output/cache/test/page_0004.png",
        ]

        def fake_batch(paths, **kwargs):
            return [
                {"form_type": None},
                {"form_type": "7pre"},
            ]

        with patch.object(pdf_recognize, "_get_extract_id_form_batch", return_value=fake_batch):
            debug: dict = {}
            page_infos = pdf_recognize._detect_ids_and_form_types(page_paths, verbose=False, debug_out=debug)

        self.assertIsNone(page_infos[0]["form_type"])
        self.assertFalse(page_infos[0]["form_type_inferred"])
        self.assertEqual(page_infos[2]["form_type"], "7pre")
        self.assertEqual(debug["form_type_inference"]["inferred_pair_count"], 0)
        self.assertEqual(debug["form_type_inference"]["unresolved_pair_count"], 1)

    def test_source_pages_from_manifest_resolves_original_pages(self):
        manifest = {
            "pages": [
                {
                    "merged_page": 1,
                    "source_pdf_path": "output/uploads/original_a.pdf",
                    "source_file_name": "original_a.pdf",
                    "source_page": 3,
                    "source_index": 0,
                },
                {
                    "merged_page": 2,
                    "source_pdf_path": "output/uploads/original_a.pdf",
                    "source_file_name": "original_a.pdf",
                    "source_page": 4,
                    "source_index": 0,
                },
            ]
        }

        pages = pdf_recognize._source_pages_from_manifest(manifest)

        self.assertEqual(pages[1]["source_page"], 3)
        self.assertEqual(pages[2]["source_page"], 4)
        self.assertEqual(Path(pages[1]["source_pdf_path"]).name, "original_a.pdf")

    def test_build_pair_items_uses_original_source_page_numbers(self):
        text_paths = [ROOT / "output/a_page_0001.png", ROOT / "output/a_page_0002.png"]
        mcq_paths = [ROOT / "output/a_page_0001_bin.png", ROOT / "output/a_page_0002_bin.png"]
        page_infos = [
            {
                "form_type": "8pre",
                "source_pdf_path": str(ROOT / "output/uploads/original_a.pdf"),
                "source_file_name": "original_a.pdf",
                "source_page": 7,
                "merged_page": 1,
            },
            {
                "form_type": "8pre",
                "source_pdf_path": str(ROOT / "output/uploads/original_a.pdf"),
                "source_file_name": "original_a.pdf",
                "source_page": 8,
                "merged_page": 2,
            },
        ]
        page_data = [
            [{"name": "id", "kind": "text", "text": "8012345A"}],
            [],
        ]

        items = pdf_recognize._build_pair_items(
            text_paths,
            mcq_paths,
            page_infos,
            page_data,
            verbose=False,
        )

        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["page_odd"], 7)
        self.assertEqual(item["page_even"], 8)
        self.assertEqual(item["merged_page_odd"], 1)
        self.assertEqual(item["merged_page_even"], 2)
        self.assertEqual(item["source_file_name"], "original_a.pdf")
        self.assertEqual(Path(item["pdf_path"]).name, "original_a.pdf")

    def test_deferred_text_llm_clears_final_regex_mismatch_by_default(self):
        page_data = [
            [
                {
                    "name": "date",
                    "kind": "text",
                    "text": "4/20/26",
                    "_llm_prompt_override": "Return a date.",
                    "_ocr_output_regex": r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(20[0-9]{2})$",
                }
            ]
        ]

        def fake_postprocess(text_by_uid, **kwargs):
            return {uid: "04/20/26" for uid in text_by_uid}

        with patch.dict(pdf_recognize.ROI_PAGE_RECOGNITION, {"ocr_regex_retry_steps": []}, clear=False):
            with patch("text_roi_llm.postprocess_text_rois", fake_postprocess):
                debug: dict = {}
                pdf_recognize._run_deferred_text_llm_postprocess(page_data, verbose=False, debug_out=debug)

        self.assertEqual(page_data[0][0]["text"], "")
        retry = debug["text_llm_deferred"]["regex_retry"]
        self.assertEqual(retry["cleared_final_mismatch_count"], 1)
        trace = next(iter(retry["per_roi"].values()))
        self.assertTrue(trace["cleared_final_mismatch"])
        self.assertEqual(trace["final_text_after_retries"], "04/20/26")

    def test_deferred_text_llm_can_keep_final_regex_mismatch_when_configured(self):
        page_data = [
            [
                {
                    "name": "date",
                    "kind": "text",
                    "text": "4/20/26",
                    "_llm_prompt_override": "Return a date.",
                    "_ocr_output_regex": r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(20[0-9]{2})$",
                }
            ]
        ]

        def fake_postprocess(text_by_uid, **kwargs):
            return {uid: "04/20/26" for uid in text_by_uid}

        with patch.dict(
            pdf_recognize.ROI_PAGE_RECOGNITION,
            {
                "ocr_regex_retry_steps": [],
                "ocr_regex_retry_clear_on_final_mismatch": False,
            },
            clear=False,
        ):
            with patch("text_roi_llm.postprocess_text_rois", fake_postprocess):
                debug: dict = {}
                pdf_recognize._run_deferred_text_llm_postprocess(page_data, verbose=False, debug_out=debug)

        self.assertEqual(page_data[0][0]["text"], "04/20/26")
        retry = debug["text_llm_deferred"]["regex_retry"]
        self.assertFalse(retry["clear_on_final_mismatch"])
        self.assertEqual(retry["cleared_final_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
