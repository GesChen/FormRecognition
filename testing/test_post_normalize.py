from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import post_normalize


def _items(form_type: str = "form_a") -> list[dict]:
    return [
        {
            "id": "1",
            "form_type": form_type,
            "data": [
                {"name": "field_alpha", "kind": "text", "text": "Value One"},
                {"name": "field_beta", "kind": "text", "text": "North"},
            ],
        },
        {
            "id": "2",
            "form_type": form_type,
            "data": [
                {"name": "field_alpha", "kind": "text", "text": "Valu One"},
            ],
        },
    ]


def test_normalize_items_skips_without_matching_enabled_fields(monkeypatch):
    called = False

    def fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return {"text": "{}"}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    normalized, original = post_normalize.normalize_items(_items(), {"form_a": {"field_gamma"}})

    assert called is False
    assert normalized == original == _items()


def test_normalize_items_respects_global_llm_postprocess_toggle(monkeypatch):
    called = False

    def fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return {"text": "{}"}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    monkeypatch.setitem(post_normalize.LLM_POSTPROCESS, "enabled", False)
    debug: dict = {}
    normalized, original = post_normalize.normalize_items(
        _items(),
        {"form_a": {"field_alpha"}},
        debug_out=debug,
    )

    assert called is False
    assert normalized == original == _items()
    assert debug["enabled"] is False
    assert debug["skipped_reason"] == "disabled"


def test_normalize_items_applies_mapping_response(monkeypatch):
    def fake_generate(*args, **kwargs):
        return {"text": json.dumps({"Value One": "Value One", "Valu One": "Value One"})}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    normalized, original = post_normalize.normalize_items(_items(), {"form_a": {"field_alpha"}})

    assert original == _items()
    assert [r["text"] for i in normalized for r in i["data"] if r["name"] == "field_alpha"] == [
        "Value One",
        "Value One",
    ]


def test_normalize_items_groups_by_form_and_roi(monkeypatch):
    prompts: list[str] = []
    items = _items("form_a") + _items("form_b")

    def fake_generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        return {"text": json.dumps({"Value One": "Value One", "Valu One": "Valu One"})}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    post_normalize.normalize_items(items, {"form_a": {"field_alpha"}, "form_b": {"field_alpha"}})

    assert len(prompts) == 2
    assert "Form type: form_a" in prompts[0]
    assert "Form type: form_b" in prompts[1]


def test_normalize_items_retries_until_json_mapping_has_expected_keys(monkeypatch):
    calls = iter(
        [
            {"text": "not json"},
            {"text": json.dumps({"unrelated": "only one"})},
            {"text": json.dumps({"Value One": "Value One", "Valu One": "Value One"})},
        ]
    )
    prompts: list[str] = []

    def fake_generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        return next(calls)

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    normalized, original = post_normalize.normalize_items(_items(), {"form_a": {"field_alpha"}})

    assert original == _items()
    assert [r["text"] for i in normalized for r in i["data"] if r["name"] == "field_alpha"] == [
        "Value One",
        "Value One",
    ]
    assert len(prompts) == 3
    assert prompts[0] != prompts[1] != prompts[2]


def test_normalize_items_leaves_values_unchanged_after_retry_exhaustion(monkeypatch):
    def fake_generate(*args, **kwargs):
        return {"text": json.dumps({"unrelated": "only one"})}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    monkeypatch.setattr(post_normalize, "_max_retries", lambda: 1)
    normalized, original = post_normalize.normalize_items(_items(), {"form_a": {"field_alpha"}})

    assert normalized == original == _items()


def test_normalize_items_does_not_apply_placeholder_echo(monkeypatch):
    def fake_generate(*args, **kwargs):
        return {"text": json.dumps({"Value One": "<normalize>", "Valu One": "Value One"})}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    normalized, original = post_normalize.normalize_items(_items(), {"form_a": {"field_alpha"}})

    assert original == _items()
    assert [r["text"] for i in normalized for r in i["data"] if r["name"] == "field_alpha"] == [
        "Value One",
        "Value One",
    ]


def test_normalize_items_prompt_only_uses_observed_values(monkeypatch):
    prompts: list[str] = []

    def fake_generate(prompt, *args, **kwargs):
        prompts.append(prompt)
        return {"text": json.dumps({"Value One": "Value One", "Valu One": "Value One"})}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    normalized, original = post_normalize.normalize_items(_items(), {"form_a": {"field_alpha"}})

    assert original == _items()
    assert [r["text"] for i in normalized for r in i["data"] if r["name"] == "field_alpha"] == [
        "Value One",
        "Value One",
    ]
    assert len(prompts) == 1
    assert "Value One" in prompts[0]
    assert "Valu One" in prompts[0]
    assert "Unobserved Value" not in prompts[0]


def test_placeholder_detection_uses_roi_name_not_specific_field_names(monkeypatch):
    items = [
        {
            "id": "1",
            "form_type": "form_a",
            "data": [{"name": "custom_field", "kind": "text", "text": "Name of custom_field"}],
        }
    ]

    def fake_generate(*args, **kwargs):
        return {"text": json.dumps({"Name of custom_field": None})}

    monkeypatch.setattr(post_normalize, "generate", fake_generate)
    normalized, original = post_normalize.normalize_items(items, {"form_a": {"custom_field"}})

    assert normalized == original == items
