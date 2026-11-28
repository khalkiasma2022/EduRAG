"""Tests du cache disque de compréhension (CourseModel)."""

from __future__ import annotations

from pathlib import Path

from src.revision.engine.course_cache import CourseUnderstandingCache
from src.revision.engine.models import Concept, CourseModel


def _course() -> CourseModel:
    return CourseModel(
        concepts=[
            Concept(
                id="c_net",
                label="Internet",
                kind="other",
                pages=[1],
                essential=True,
                notes="Réseau mondial.",
            )
        ],
        themes=["Web"],
        essential_concept_ids=["c_net"],
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    cache = CourseUnderstandingCache(tmp_path / "course_cache")
    course = _course()
    cache.save(
        "doc-abc",
        filename="cours.pdf",
        page_start=1,
        page_end=3,
        course=course,
    )
    loaded = cache.load("doc-abc")
    assert loaded is not None
    assert loaded.model_dump() == course.model_dump()
    assert (tmp_path / "course_cache" / "doc-abc.json").is_file()


def test_miss_returns_none(tmp_path: Path) -> None:
    cache = CourseUnderstandingCache(tmp_path / "course_cache")
    assert cache.load("missing-id") is None


def test_corrupt_file_returns_none(tmp_path: Path) -> None:
    cache = CourseUnderstandingCache(tmp_path / "course_cache")
    path = cache.path_for("broken")
    path.write_text("{not-json", encoding="utf-8")
    assert cache.load("broken") is None


def test_invalid_course_payload_returns_none(tmp_path: Path) -> None:
    cache = CourseUnderstandingCache(tmp_path / "course_cache")
    path = cache.path_for("bad-course")
    path.write_text(
        '{"document_id":"bad-course","filename":"x.pdf","page_start":1,'
        '"page_end":1,"course":{"concepts":"oops"}}',
        encoding="utf-8",
    )
    assert cache.load("bad-course") is None
