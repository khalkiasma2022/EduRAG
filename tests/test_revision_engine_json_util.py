"""Tests de l'extraction JSON robuste pour Smart Revision."""

from __future__ import annotations

import pytest

from src.revision.engine.json_util import load_json_object
from src.revision.engine.understanding import _parse_window_json


def test_load_json_object_strips_markdown_fence() -> None:
    raw = 'Voici le résultat:\n```json\n{"themes":["A"],"concepts":[]}\n```\nFin.'
    assert load_json_object(raw) == {"themes": ["A"], "concepts": []}


def test_load_json_object_extracts_first_braced_object() -> None:
    raw = 'Explication.\n{"themes": ["Cloud"], "concepts": []}\nMerci.'
    assert load_json_object(raw)["themes"] == ["Cloud"]


def test_load_json_object_repairs_trailing_commas() -> None:
    raw = '{"themes": ["A",], "concepts": [],}'
    assert load_json_object(raw) == {"themes": ["A"], "concepts": []}


def test_parse_window_json_accepts_string_page_numbers() -> None:
    raw = (
        '{"themes":["T"],"concepts":[{"id":"c1","label":"x","kind":"definition",'
        '"pages":["1","2"]}],"relations":[],"contradictions":[]}'
    )
    payload = _parse_window_json(raw)
    assert payload["concepts"][0]["pages"] == ["1", "2"]
    from src.revision.engine.understanding import _concept_from_raw

    concept = _concept_from_raw(payload["concepts"][0])
    assert concept.pages == [1, 2]


def test_parse_window_json_defaults_missing_optional_lists() -> None:
    raw = '{"themes":null,"concepts":[{"id":"c1","label":"x","kind":"other"}]}'
    payload = _parse_window_json(raw)
    assert payload["themes"] == []
    assert payload["relations"] == []
    assert payload["contradictions"] == []
    assert payload["concepts"][0]["pages"] == []


def test_load_json_object_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="JSON invalide"):
        load_json_object("pas du json du tout")


def test_load_json_object_repairs_truncated_concepts_array() -> None:
    raw = (
        '{ "themes": [ "technology" ], "concepts": [ '
        '{ "id": "internet", "label": "Internet", "kind": "other", '
        '"pages": [1], "essential": true, "depends_on": [], "notes": null, "same_as": null }, '
        '{ "id": "box_internet", "label": "Box Internet", "kin'
    )
    payload = load_json_object(raw)
    assert payload["themes"] == ["technology"]
    assert len(payload["concepts"]) == 1
    assert payload["concepts"][0]["id"] == "internet"


def test_load_json_object_repairs_truncated_mid_string_block() -> None:
    raw = (
        '{"index":2,"title":"Copy","concept_ids_covered":[],"blocks":['
        '{"type":"paragraph","text":"Le copier-coller déplace du texte."},'
        '{"type":"definition","text":"Le formatage prépare un disqu'
    )
    payload = load_json_object(raw)
    assert payload["index"] == 2
    assert payload["title"] == "Copy"
    assert len(payload["blocks"]) == 1
    assert payload["blocks"][0]["type"] == "paragraph"
