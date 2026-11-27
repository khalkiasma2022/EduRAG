"""Régressions anti-copie : vrai collage vs jargon / notes de concepts."""

from __future__ import annotations

from src.revision.engine.models import Concept, CourseModel, CoursePage, CourseUnit
from src.revision.engine.near_copy import (
    is_near_copy,
    page_source_corpus,
    whitelist_terms,
)


def _course_with_notes(*, notes: str) -> CourseModel:
    return CourseModel(
        units=[
            CourseUnit(
                text=(
                    "Le cloud computing designe un mode de fourniture de ressources "
                    "informatiques a la demande via un reseau."
                ),
                page_start=1,
                page_end=1,
            )
        ],
        concepts=[
            Concept(
                id="c_cloud",
                label="cloud computing",
                kind="definition",
                pages=[1],
                essential=True,
                notes=notes,
            )
        ],
        themes=["Cloud"],
        essential_concept_ids=["c_cloud"],
    )


def _pages() -> list[CoursePage]:
    return [
        CoursePage(
            page_number=1,
            text=(
                "Le cloud computing designe un mode de fourniture de ressources "
                "informatiques a la demande via un reseau. Les modeles IaaS PaaS "
                "et SaaS organisent les responsabilites entre fournisseur et client."
            ),
            filename="cloud.pdf",
        )
    ]


def test_verbatim_page_paragraph_is_near_copy_even_if_notes_repeat_it() -> None:
    page_sentence = (
        "Le cloud computing designe un mode de fourniture de ressources "
        "informatiques a la demande via un reseau."
    )
    course = _course_with_notes(notes=page_sentence)
    corpus = page_source_corpus(course, _pages())
    assert is_near_copy(page_sentence, corpus, whitelist=whitelist_terms(course))


def test_short_definition_sentence_in_notes_is_not_whitelisted() -> None:
    course = _course_with_notes(notes="Une variable stocke une valeur.")
    terms = whitelist_terms(course)
    assert "Une variable stocke une valeur." not in terms
    assert "cloud computing" in terms


def test_short_jargon_without_period_is_whitelisted() -> None:
    course = _course_with_notes(notes="machine virtuelle\nIaaS")
    terms = whitelist_terms(course)
    assert "machine virtuelle" in terms
    assert "IaaS" in terms


def test_original_rewrite_sharing_labels_is_not_near_copy() -> None:
    course = _course_with_notes(
        notes=(
            "Le cloud computing designe un mode de fourniture de ressources "
            "informatiques a la demande via un reseau."
        )
    )
    corpus = page_source_corpus(course, _pages())
    paragraph = (
        "En revision, le cloud computing s'explique comme un acces flexible a des "
        "capacites distantes : on consomme ce dont on a besoin, quand on en a besoin, "
        "sans gerer soi-meme tout le materiel sous-jacent."
    )
    assert not is_near_copy(paragraph, corpus, whitelist=whitelist_terms(course))


def test_concept_notes_are_excluded_from_page_corpus() -> None:
    unique_note = (
        "Formulation issue de l analyse uniquement presente dans les notes du concept "
        "et absente des pages PDF fournies pour ce test de regression."
    )
    course = _course_with_notes(notes=unique_note)
    corpus = page_source_corpus(course, _pages())
    assert unique_note.casefold() not in corpus.casefold()
    assert not is_near_copy(unique_note, corpus, whitelist=whitelist_terms(course))


def test_long_copied_window_from_pages_is_near_copy() -> None:
    course = _course_with_notes(notes="IaaS")
    corpus = page_source_corpus(course, _pages())
    # Collage d'une longue suite de mots du PDF, reformulée à peine.
    pasted = (
        "Le cloud computing designe un mode de fourniture de ressources "
        "informatiques a la demande via un reseau, ce qui reste central."
    )
    assert is_near_copy(pasted, corpus, whitelist=whitelist_terms(course))
