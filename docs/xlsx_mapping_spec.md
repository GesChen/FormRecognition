# XLSX Mapping Specification

Human-authored JSON files that define how pipeline recognition output maps onto
Excel template columns. One file per `form_type`, stored as `data/xlsx/mappings/<release>/<form_type>.json`.

---

## Quick example

```json
{
  "form_type": "6post",
  "sheet": "6th Grade Post-Assessment Data",
  "start_row": 4,
  "mappings": [
    { "source": "id",  "column": "A",  "type": "direct" },
    { "source": "1",   "column": "L",  "type": "direct", "transform": "upper" },
    { "source": "7",   "column": "R",  "type": "lookup",
      "map": { "a": "False", "b": "True" } },
    { "source": "25",  "column": "AJ", "type": "direct" },
    { "source": "20",  "column": "AE", "type": "lookup",
      "map": { "a": "Strongly Disagree", "b": "Disagree",
               "c": "Neither Agree nor Disagree", "d": "Agree",
               "e": "Strongly Agree" } },
    { "source": "28",  "type": "multi_column", "mark": "Yes",
      "choices": { "a": "AM", "b": "AN", "c": "AO", "d": "AP",
                   "e": "AQ", "f": "AR", "g": "AS", "h": "AT" } }
  ]
}
```

---

## Top-level keys

| Key          | Type     | Required | Description |
|--------------|----------|----------|-------------|
| `form_type`  | string   | yes      | Pipeline form type this mapping handles (e.g. `"6post"`, `"6pre"`). |
| `sheet`      | string   | yes      | Exact Excel tab name to write into. |
| `start_row`  | integer  | yes      | First data row (rows above are headers). Typically `4`. |
| `mappings`   | array    | yes      | Ordered list of mapping objects (see below). |

---

## Source notation

Every mapping (except `static`) has a `"source"` string that says **where in the
pipeline item to read the value**.

A pipeline item looks like this:

```json
{
  "id": "601034A",
  "form_type": "6post",
  "page_odd": 1,
  "page_even": 2,
  "data": [
    { "name": "1",  "kind": "mcq",  "text": "c" },
    { "name": "13", "kind": "text", "text": "Uterus e" },
    ...
  ]
}
```

| Source value | Reads from | Example result |
|---|---|---|
| `"id"` | `item.id` | `"601034A"` |
| `"form_type"` | `item.form_type` | `"6post"` |
| `"page_odd"` | `item.page_odd` | `1` |
| `"page_even"` | `item.page_even` | `2` |
| `"1"` | `data[]` entry where `name == "1"` → its `text` | `"c"` |
| `"13"` | `data[]` entry where `name == "13"` → its `text` | `"Uterus e"` |
| `"25"` | `data[]` entry where `name == "25"` → its `text` | `"Not doing it"` |

**Rule**: any source that is purely numeric (or matches a `data[].name`) pulls
from `data[]`. The reserved words `id`, `form_type`, `page_odd`, `page_even`
pull from the item's top-level fields.

---

## Mapping types

### Optional cell comments

Any mapping row that writes a cell can include a `comment`. The XLSX filler
renders the comment and attaches it to the cell that was actually written.

```json
{
  "source": "25",
  "column": "AJ",
  "type": "direct",
  "comment": {
    "text": "OCR confidence: {ocr_confidence}\nSource: {file_name}, PDF pages {pdf_pages}"
  }
}
```

`comment` may be a plain string or an object with `text`/`template` and optional
literal replacements:

```json
{
  "source": "25",
  "column": "AJ",
  "type": "direct",
  "comment": {
    "template": "Confidence: {{CONF}}\nFile: {file_name}",
    "replacements": {
      "{{CONF}}": "{ocr_confidence}"
    }
  }
}
```

Built-in drop-ins:

| Drop-in | Description |
|---|---|
| `{value}` | Final value written to the cell after transform/lookup. |
| `{raw}` | Raw source value before transform/lookup. |
| `{source}` | Mapping source name. |
| `{type}` | Mapping type. |
| `{ocr_confidence}` | Text OCR confidence label and score, e.g. `high (0.980)`. Available for text ROI cells and ID cells. |
| `{ocr_confidence_score}` | Text OCR confidence score only. |
| `{ocr_confidence_label}` | Text OCR confidence label only. |
| `{file_name}` | Original PDF filename when available, otherwise the PDF stem. |
| `{pdf_stem}` | Recognition/output PDF stem. |
| `{pdf_pages}` / `{page_numbers}` | Item pages as a single string, e.g. `1-2` or `3`. |
| `{page_odd}` | Odd/side-A PDF page number. |
| `{page_even}` | Even/side-B PDF page number. |

For `multi_column`, the comment is added only to the selected/no-answer cell,
not to blanked unselected cells.

### 1. `direct` — write value as-is

Use for **free-response text**, **MCQ letters**, **numbers**, **dates**, or any
value that should appear in the cell verbatim (or with a simple transform).

```json
{ "source": "25", "column": "AJ", "type": "direct" }
```

| Key         | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `source`    | string | yes      | Where to read (see Source notation). |
| `column`    | string | yes      | Target column letter. |
| `type`      | string | yes      | `"direct"` |
| `transform` | string | no       | Optional transformation (see below). |

**Transforms** (optional):

| Transform   | Effect | Use case |
|-------------|--------|----------|
| `"upper"`   | Uppercase the entire value. | MCQ letters: `"c"` → `"C"` |
| `"lower"`   | Lowercase the entire value. | Normalizing text. |
| `"number"`  | Parse as number (int or float). | Age, scores. |
| `"date"`    | Parse as a date (`YYYY-MM-DD`, `MM/DD/YYYY`). | Test dates. |

Examples:

```json
// Free-response answer → column AJ, written as-is
{ "source": "25", "column": "AJ", "type": "direct" }

// MCQ letter → column L, uppercased
{ "source": "1", "column": "L", "type": "direct", "transform": "upper" }

// Record ID → column A
{ "source": "id", "column": "A", "type": "direct" }

// Age → column AL, as a number
{ "source": "27", "column": "AL", "type": "direct", "transform": "number" }
```

---

### 2. `lookup` — map value through a dictionary

Use when the raw answer is a **letter** (or short code) but the template
expects **specific text** — True/False, Likert scales, grade labels, etc.

```json
{
  "source": "7",
  "column": "R",
  "type": "lookup",
  "map": {
    "a": "False",
    "b": "True"
  }
}
```

| Key       | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `source`  | string | yes      | Where to read. |
| `column`  | string | yes      | Target column letter. |
| `type`    | string | yes      | `"lookup"` |
| `map`     | object | yes      | `{ raw_value: cell_value }` dictionary. Matching is **case-insensitive**. |
| `default` | string | no       | Value to write if raw answer is not in `map`. If omitted, unmatched answers leave the cell blank. |

Examples:

```json
// True/False: "b" → True, "a" → False
{
  "source": "7",
  "column": "R",
  "type": "lookup",
  "map": { "a": "False", "b": "True" }
}

// Likert scale
{
  "source": "20",
  "column": "AE",
  "type": "lookup",
  "map": {
    "a": "Strongly Disagree",
    "b": "Disagree",
    "c": "Neither Agree nor Disagree",
    "d": "Agree",
    "e": "Strongly Agree"
  }
}

// Grade level
{
  "source": "26",
  "column": "AK",
  "type": "lookup",
  "map": { "a": "6th Grade", "b": "7th Grade", "c": "8th Grade" }
}

// Gender
{
  "source": "29",
  "column": "AU",
  "type": "lookup",
  "map": { "a": "Girl", "b": "Boy", "c": "Non-Binary", "d": "Unknown/Not Reported" },
  "default": "Unknown/Not Reported"
}
```

---

### 3. `multi_column` — one answer selects which column gets marked

Use when a **single MCQ letter** determines which of several columns receives
a mark (like `"Yes"`), and the rest stay blank or get a different value.

Typical case: race/ethnicity, where the template has one column per option and
only the selected column gets `"Yes"`.

```json
{
  "source": "28",
  "type": "multi_column",
  "mark": "Yes",
  "no_answer": "AT",
  "choices": {
    "a": "AM",
    "b": "AN",
    "c": "AO",
    "d": "AP",
    "e": "AQ",
    "f": "AR",
    "g": "AS",
    "h": "AT"
  }
}
```

| Key         | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `source`    | string | yes      | Where to read. |
| `type`      | string | yes      | `"multi_column"` |
| `mark`      | string | yes      | Value to write in the **selected** column (e.g. `"Yes"`, `"X"`, `"1"`). |
| `choices`   | object | yes      | `{ raw_value: column_letter }` — maps each possible answer to its target column. |
| `blank`     | string | no       | Value to write in **unselected** columns. Default: leave them empty. Set to `""` or `"No"` if the template requires every column filled. |
| `no_answer` | string | no       | Column letter to receive `mark` when the source has **no answer** (null/empty). Default: leave all columns empty. |

**How it works**:

1. Read the raw answer (e.g. `"c"`).
2. Look up `"c"` in `choices` → `"AO"`.
3. Write `mark` (`"Yes"`) into column `AO`.
4. If `blank` is set, write it into every other column in `choices`
   (`AM`, `AN`, `AP`, `AQ`, `AR`, `AS`, `AT`).

If the raw answer is not in `choices`, nothing is written (all columns stay as-is).

---

### 4. `static` — always write a fixed value

Use for columns that have the same value for every record in a batch (location,
session format, organization, facilitator names, etc.). No `source` needed.

```json
{ "column": "C", "type": "static", "value": "NPS" }
```

| Key     | Type   | Required | Description |
|---------|--------|----------|-------------|
| `column`| string | yes      | Target column letter. |
| `type`  | string | yes      | `"static"` |
| `value` | string | yes      | The fixed value to write. |

Examples:

```json
{ "column": "C",  "type": "static", "value": "NPS" }
{ "column": "D",  "type": "static", "value": "In-Person" }
{ "column": "F",  "type": "static", "value": "EVMS" }
```

---

## Complete example — 6th Grade Post-Assessment

Below is a full mapping for the `6post` form type. Compare this against the
template column table to verify every column is covered.

**Template columns for reference** (Row1 / Row2 / example):

| Col | Row1 header | Row2 header |
|-----|-------------|-------------|
| A   | Record ID | |
| B   | Date of Test | |
| C   | Location | |
| D   | Session Format | |
| E   | If Other, Please Describe | |
| F   | Organization Name | |
| G–K | Facilitator(s) | (per-facilitator sub-columns) |
| L   | Multiple Choice Questions | Q1 |
| M   | | Q2 |
| N   | | Q3 |
| O   | | Q4 |
| P   | | Q5 |
| Q   | | Q6 |
| R   | True or False Questions | Q7 |
| S   | | Q8 |
| T   | | Q9 |
| U   | | Q10 |
| V   | | Q11 |
| W   | | Q12 |
| X   | Matching Questions | Q13 |
| Y   | | Q14 |
| Z   | | Q15 |
| AA  | | Q16 |
| AB  | | Q17 |
| AC  | | Q18 |
| AD  | | Q19 |
| AE  | Experience Questions | Created a Safe Space |
| AF  | | Feel Engaged |
| AG  | | Express Myself & Ask Qs |
| AH  | | Facilitator Appeared Knowledgeable |
| AI  | | Glad Participated |
| AJ  | Written Question | |
| AK  | Grade Level | |
| AL  | Age | |
| AM–AT | Race/Ethnicity | (per-race sub-columns) |
| AU  | Gender | |
| AV  | Sexual Orientation | |

**Pipeline data items** have `name` 1–29 (`kind`: mcq or text).

```json
{
  "form_type": "6post",
  "sheet": "6th Grade Post-Assessment Data",
  "start_row": 4,
  "mappings": [

    { "source": "id", "column": "A", "type": "direct" },

    { "column": "D", "type": "static", "value": "In-Person" },
    { "column": "F", "type": "static", "value": "EVMS" },

    { "source": "1",  "column": "L", "type": "direct", "transform": "upper" },
    { "source": "2",  "column": "M", "type": "direct", "transform": "upper" },
    { "source": "3",  "column": "N", "type": "direct", "transform": "upper" },
    { "source": "4",  "column": "O", "type": "direct", "transform": "upper" },
    { "source": "5",  "column": "P", "type": "direct", "transform": "upper" },
    { "source": "6",  "column": "Q", "type": "direct", "transform": "upper" },

    { "source": "7",  "column": "R", "type": "lookup",
      "map": { "a": "False", "b": "True" } },
    { "source": "8",  "column": "S", "type": "lookup",
      "map": { "a": "False", "b": "True" } },
    { "source": "9",  "column": "T", "type": "lookup",
      "map": { "a": "False", "b": "True" } },
    { "source": "10", "column": "U", "type": "lookup",
      "map": { "a": "False", "b": "True" } },
    { "source": "11", "column": "V", "type": "lookup",
      "map": { "a": "False", "b": "True" } },
    { "source": "12", "column": "W", "type": "lookup",
      "map": { "a": "False", "b": "True" } },

    { "source": "13", "column": "X",  "type": "direct", "transform": "upper" },
    { "source": "14", "column": "Y",  "type": "direct", "transform": "upper" },
    { "source": "15", "column": "Z",  "type": "direct", "transform": "upper" },
    { "source": "16", "column": "AA", "type": "direct", "transform": "upper" },
    { "source": "17", "column": "AB", "type": "direct", "transform": "upper" },
    { "source": "18", "column": "AC", "type": "direct", "transform": "upper" },
    { "source": "19", "column": "AD", "type": "direct", "transform": "upper" },

    { "source": "20", "column": "AE", "type": "lookup",
      "map": { "a": "Strongly Disagree", "b": "Disagree",
               "c": "Neither Agree nor Disagree", "d": "Agree",
               "e": "Strongly Agree" } },
    { "source": "21", "column": "AF", "type": "lookup",
      "map": { "a": "Strongly Disagree", "b": "Disagree",
               "c": "Neither Agree nor Disagree", "d": "Agree",
               "e": "Strongly Agree" } },
    { "source": "22", "column": "AG", "type": "lookup",
      "map": { "a": "Strongly Disagree", "b": "Disagree",
               "c": "Neither Agree nor Disagree", "d": "Agree",
               "e": "Strongly Agree" } },
    { "source": "23", "column": "AH", "type": "lookup",
      "map": { "a": "Strongly Disagree", "b": "Disagree",
               "c": "Neither Agree nor Disagree", "d": "Agree",
               "e": "Strongly Agree" } },
    { "source": "24", "column": "AI", "type": "lookup",
      "map": { "a": "Strongly Disagree", "b": "Disagree",
               "c": "Neither Agree nor Disagree", "d": "Agree",
               "e": "Strongly Agree" } },

    { "source": "25", "column": "AJ", "type": "direct" },

    { "source": "26", "column": "AK", "type": "lookup",
      "map": { "a": "6th Grade" },
      "default": "6th Grade" },

    { "source": "27", "column": "AL", "type": "direct", "transform": "number" },

    { "source": "28", "type": "multi_column", "mark": "Yes",
      "no_answer": "AT",
      "choices": {
        "a": "AM", "b": "AN", "c": "AO", "d": "AP",
        "e": "AQ", "f": "AR", "g": "AS", "h": "AT"
      }
    },

    { "source": "29", "column": "AU", "type": "lookup",
      "map": { "a": "Girl", "b": "Boy", "c": "Non-Binary",
               "d": "Unknown/Not Reported" },
      "default": "Unknown/Not Reported" }
  ]
}
```

---

## How to create a mapping file

1. **Open the blank template** in Excel. Note the sheet name and header rows.
   Write down each column letter and what it expects (from Row 1 / Row 2 headers).

2. **Open a sample pipeline output** JSON. Look at `items[0].data` — list all
   `name` values and their `kind` (mcq / text).

3. **For each template column**, decide:

   | Template expects | Pipeline gives | Use type |
   |---|---|---|
   | Free text verbatim | `kind: "text"` | `direct` |
   | MCQ letter (A, B, C…) | `kind: "mcq"`, text is lowercase letter | `direct` + `transform: "upper"` |
   | Text word from an MCQ (True/False, Agree/Disagree…) | `kind: "mcq"`, text is a letter | `lookup` with a `map` |
   | One of N columns gets a mark | `kind: "mcq"`, text is a letter | `multi_column` with `choices` |
   | Same value every row | (nothing from pipeline) | `static` |
   | Number | `kind: "text"` or `kind: "mcq"` | `direct` + `transform: "number"` |
   | Date | (external / batch-level) | `static` or `direct` + `transform: "date"` |

4. **Write each mapping** in the `mappings` array. Go column by column
   (left to right through the template) so nothing is missed.

5. **Save** the file as `data/xlsx/mappings/<release>/<form_type>.json`
   (e.g. `data/xlsx/mappings/2026/6post.json`).

---

## Repeated lookup maps

When many questions share the same lookup table (e.g. five Likert questions
all use the same Strongly Disagree → Strongly Agree scale), you still write a
separate mapping entry per question/column. Copy-paste the `map` object.

This keeps the file flat and explicit — every column has exactly one mapping
entry, no indirection.

---

## Validation rules

A valid mapping file must satisfy:

- `form_type`, `sheet` are non-empty strings.
- `start_row` is a positive integer.
- `mappings` is a non-empty array.
- Every mapping has `"type"` as one of: `direct`, `lookup`, `multi_column`, `static`.
- `direct` and `lookup` mappings have a `"column"` (single letter or two-letter like `"AE"`).
- `lookup` mappings have a `"map"` object with at least one key.
- `multi_column` mappings have a `"choices"` object with at least one key and a `"mark"` string.
- `static` mappings have a `"value"`.
- No two mappings write to the same column (except `multi_column` entries whose
  `choices` span separate columns — those are fine).
