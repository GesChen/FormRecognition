# Manual review — student ID, text ROIs, and MCQ (schema)

Recognition output JSON may include a top-level **`human_review`** object listing fields that should be checked or corrected by a human (or by a tool that fills in **`corrected_*`** fields). This enables **manual ID correction** and **manual ROI correction** (text and multiple-choice) on the same file.

**Web UI:** `python3 tools/human_review_web.py` — browse `output/recognition/*.json`, use **Review queue** to step through pending fields (full-page preview with ROI outline, predicted values, correction field; **Enter** saves and advances), or edit the table and save (see `docs/tools.md`).

- **Detection** is implemented in `py/ocr_human_review.py` (`build_human_review_block`).
- **Text OCR review** is detected only in `roi_page_module.recognize_text_fields`; entries are appended to a shared queue with a **`review_target_ref`**, then merged into `human_review.pending` by `build_human_review_block` (pipeline passes `text_ocr_queue=...`; legacy callers omit it and fall back to scanning `data[]`).
- Primary decision source is OCR engine `needs_human_review` (from the staged OCR workflow/API result). Score thresholds are retained as fallback for older rows that do not include that flag.
- **Applying edits** without a UI: `apply_manual_corrections(items, human_review)` in the same module (matches `review_target_ref` on `data[]` rows when present).

See also: `docs/modules.md` → `PDF_RECOGNITION["human_review"]`.

---

## Pipeline configuration (`config.py`)

Under `PDF_RECOGNITION["human_review"]`:

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `enabled` | `bool` | `false` | Master switch. If `false`, no `human_review` block is written and text ROIs do not collect OCR confidence stats. |
| `text_review_use_needs_human_review` | `bool` | `true` | **Text** ROIs: use OCR `needs_human_review` as the primary queue signal. |
| `id_review_use_needs_human_review` | `bool` | `true` | **ID** rows: when ID is otherwise valid, use `items[].id_ocr_needs_human_review` as the primary queue signal. |
| `min_text_ocr_confidence` | `float` | `0.85` | **Fallback only** for text rows without `ocr_needs_human_review`: queue when confidence score (`ocr_confidence_score` / legacy `ocr_min_score`) is **&lt;** threshold. |
| `min_id_ocr_confidence` | `float` | `0.85` | **Fallback only** for ID rows without `id_ocr_needs_human_review`: queue when `id_ocr_confidence_score` / legacy `id_ocr_min_score` is **&lt;** threshold (and ID is otherwise valid). |
| `pending_id_review` | `bool` | `true` | If `true`, allow **id** pending rows (all items when `pending_all_ids`, else invalid/missing IDs and/or OCR-driven review signal). |
| `pending_all_ids` | `bool` | `true` | If `true`, add a `kind: "id"` **pending** row for **every** item (full manual ID QC). If `false`, only invalid/missing IDs and low-confidence ID OCR (see `min_id_ocr_confidence`). |
| `pending_mcq_empty_review` | `bool` | `true` | If `true`, add `kind: "mcq"` **pending** when the chosen letter string is empty (unanswered / failed read). |

---

## Recognition output JSON (`output/recognition/<pdf_stem>.json`)

### Item shape (unchanged)

Each element of `items[]` has:

- `id` — student id string (from header OCR + LLM).
- `form_type`, `page_odd`, `page_even`
- `normalized_page_odd`, `normalized_page_even` — project-relative paths to normalized page images (for review UI crops); `even` may be `null` for a single-page pair.
- `id_ocr_confidence_label`, `id_ocr_confidence_score`, `id_ocr_needs_human_review`, `id_ocr_selected_model`, `id_ocr_selected_stage_index` — OCR metadata for the **ID top crop**.
- `id_ocr_min_score`, `id_ocr_mean_score` — legacy confidence aliases (kept for compatibility).
- `data` — list of ROI rows (see below).

### `data[]` rows

| Field | Type | Meaning |
|-------|------|--------|
| `name` | `string` | ROI name. |
| `kind` | `"text"` \| `"mcq"` | Field type. |
| `text` | `string` | Value (text content or chosen letter). |
| `page_in_pair` | `"odd"` \| `"even"` | Which page of the pair. |
| `review_target_ref` | `string` | Optional stable pointer for corrections (see below). Set on each ROI row when the pipeline provides `pdf_stem`. |

When `human_review.enabled` and OCR metadata is collected, **text** rows may include:

- `ocr_confidence_label`, `ocr_confidence_score`, `ocr_needs_human_review`, `ocr_selected_model`, `ocr_selected_stage_index`
- legacy aliases: `ocr_min_score`, `ocr_mean_score`

### `review_target_ref` (stable field pointer)

Format (pipe-separated, `pdf_stem` must not contain `|`):

`evms_review_v1|{pdf_stem}|{pair_index}|{page_in_pair}|{roi_name}|{kind}`

- **`pair_index`**: index into top-level `items[]` for this PDF.
- **`page_in_pair`**: `odd` or `even` (matches `data[].page_in_pair` after merge).
- **`roi_name`**: ROI `name`; for item-level id review the sentinel is `__id__` with `kind` `id` and `page_in_pair` `_` in the ref.

`apply_manual_corrections` prefers matching this string on `data[].review_target_ref`.

### Top-level `human_review` (when `enabled`)

There is **no** `schema_version` field; the format is maintained in lockstep with the code.

```json
{
  "human_review": {
    "text_review_use_needs_human_review": true,
    "id_review_use_needs_human_review": true,
    "min_text_ocr_confidence": 0.85,
    "min_id_ocr_confidence": 0.85,
    "pending": []
  }
}
```

---

## `pending[]` entry kinds

Common columns:

| Field | Type | Meaning |
|-------|------|--------|
| `entry_id` | `string` | Stable 24-hex id (see `make_entry_id` in `ocr_human_review.py`). |
| `review_target_ref` | `string` | Same format as `data[].review_target_ref` where applicable; used to patch the correct row. |
| `item_index` | `integer` | Index into `items[]`. |
| `kind` | `"id"` \| `"text"` \| `"mcq"` | What to correct. |
| `status` | `string` | e.g. `"pending"`. |
| `review_status` | `"pending"` \| `"reviewed"` | Set to **`reviewed`** when a human finishes the field in the Review queue UI (or by hand). |

### `kind: "id"` (manual **student ID**)

| Field | Type | Meaning |
|-------|------|--------|
| `page_in_pair` | `null` | Not used for item-level id. |
| `roi_name` | `"__id__"` | Sentinel. |
| `id_current` | `string` | Current `items[item_index].id`. |
| `corrected_id` | `string` \| `null` | Set to the approved id; then run `apply_manual_corrections`. |

### `kind: "text"` (manual **text ROI**)

| Field | Type | Meaning |
|-------|------|--------|
| `page_in_pair` | `"odd"` \| `"even"` \| `"unknown"` | Must match the `data[]` row. |
| `roi_name` | `string` | ROI `name`. |
| `ocr_confidence_label` | `"low"` \| `"medium"` \| `"high"` \| `null` | OCR confidence label. |
| `ocr_confidence_score` | `number` \| `null` | OCR confidence score (0..1). |
| `ocr_needs_human_review` | `bool` \| `null` | OCR-provided review flag (primary queue signal). |
| `ocr_selected_model` | `string` \| `null` | OCR stage/model selected as best output. |
| `ocr_selected_stage_index` | `integer` \| `null` | Stage index for selected OCR output. |
| `ocr_min_score` | `number` \| `null` | Legacy confidence alias. |
| `ocr_mean_score` | `number` \| `null` | Legacy optional confidence alias. |
| `text_current` | `string` | Current OCR text. |
| `corrected_text` | `string` \| `null` | Approved replacement. |

### `kind: "mcq"` (manual **multiple choice**)

| Field | Type | Meaning |
|-------|------|--------|
| `page_in_pair` | `"odd"` \| `"even"` \| `"unknown"` | Must match the `data[]` row. |
| `roi_name` | `string` | Main MCQ ROI name (e.g. `"5"`). |
| `text_current` | `string` | Usually `""` when auto-flagged. |
| `corrected_text` | `string` \| `null` | Approved letter(s), e.g. `"a"` or `"b"`. |

---

## Applying corrections (no UI)

1. Load the recognition JSON.
2. For each row in `human_review.pending` that you resolve, set **`corrected_id`** or **`corrected_text`**. Use **`null`** (or leave unset) to **keep** the current `items[]` value; the Review queue sets `null` when you submit an **empty** correction.
3. Call:

```python
from ocr_human_review import apply_manual_corrections

items2 = apply_manual_corrections(data["items"], data["human_review"])
data["items"] = items2
```

4. Save JSON.
5. Re-run Excel filling:

```python
from xlsx_data_entry import fill_from_pipeline

fill_from_pipeline(data["items"], pdf_stem=data["pdf_stem"], ...)
```

### Manual entries not in `pending`

You may append additional `pending` objects by hand (same `kind` / `item_index` / `roi_name` / `page_in_pair` shape) and then call `apply_manual_corrections`. **`entry_id`** can be omitted for hand-made rows if your tooling only keys off the structural fields; the reference implementation matches on `item_index`, `kind`, `roi_name`, and `page_in_pair`.
