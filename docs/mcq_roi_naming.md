## MCQ ROI naming convention

This project relies on a consistent naming scheme for multiple‑choice (MCQ) bubbles in ROI schemas so that
`mc_detect` and `pdf_recognize` can automatically discover and group them into questions.

### 1. Option ROIs (current behavior)

**Pattern (required for detection):**

- `^(\d+)([a-h])$`
  - **`\d+`** (one or more digits) = question index (e.g. `1`, `2`, `10`).
  - **`[a-h]`** (single letter a–h, case‑insensitive) = option label.

**Examples:**

- Question 1 options: `1a`, `1b`, `1c`, `1d`
- Question 2 options: `2a`, `2b`, `2c`, `2d`

**How it is used in code:**

- `pdf_recognize._get_mc_roi_names` selects ROI names that match **digits + one of a–h**.
- `pdf_recognize._bool_dict_to_letter_per_question` and
  `_letter_per_question_from_filled_and_darkness`:
  - Parse `(\d+)` as the **question number**.
  - Parse `([a-h])` as the **option letter**.
  - Group all ROIs with the same question number and pick the chosen letter.
- `mc_detect` uses the **last character** of the ROI name as the option label (`a`–`h`) for
  threshold lookup.

> **Important:** Any MCQ option ROI that should participate in detection **must** be named
> according to this pattern. Names that do not match `\d+[a-h]` are ignored by the MCQ
> grouping logic.

### 2. Group / helper ROIs (optional, for authoring)

For tooling (e.g. `roi_editor`) and for more advanced MCQ strategies (like grouping all bubbles
for a question into a larger ROI), you may add **helper ROIs** that are *not* used directly by the
current MCQ code:

- **Question‑level group ROI (optional):**
  - Pattern: `"<question_number>_group"` (e.g. `1_group`, `2_group`)
  - Intended use: a bounding box containing all option bubbles for that question.
  - Current MCQ code ignores these because they do **not** match `\d+[a-h]`.

You are free to define other helper ROIs (for example, regions around instructions, headers,
or scoring guides) as long as their names **do not** match `\d+[a-h]`. They will be ignored by
the existing MCQ detection but will still be available in the schema for future logic.

### 3. Extending to new MCQ types

If you introduce new MCQ‑style question types (e.g. multi‑select, true/false, Likert scales),
you should still:

- Use **digits + letter** for any option that should behave like an MCQ bubble:
  - Example multi‑select: `3a`, `3b`, `3c`, `3d` (logic then decides whether multiple filled
    options are allowed for question 3).
- Use **non‑matching names** (e.g. `3_group`, `3_hint`, `3_text`) for any auxiliary regions.

If/when you add new detection logic for these question types, keep it consistent with this
convention so the same schemas remain reusable across tools.

