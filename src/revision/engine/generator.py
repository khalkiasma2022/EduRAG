"""OriginalContentGenerator : rédaction pédagogique chapitre par chapitre."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.revision.engine.llm_guard import (
    TOKENS_CHAPTER_COMPLETE,
    TOKENS_CHAPTER_SHORT,
    LlmBudgetGuard,
    ensure_prompt_fits,
)
from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    Concept,
    ContentBlock,
    CourseModel,
    CoursePage,
    ExampleBlock,
    PedagogicalDocument,
    PedagogicalPlan,
    ParagraphBlock,
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?")
_RESOLVING_RE = re.compile(
    r"\b("
    r"en réalité|la vérité est|on conclut que|donc on ignore|"
    r"in reality|the truth is|we conclude that|"
    r"في الواقع|نستنتج أن"
    r")\b",
    re.IGNORECASE,
)
_LEVEL_HINTS = {
    "beginner": {
        "fr": "Niveau débutant : vocabulaire simple, explications accessibles.",
        "en": "Beginner level: simple vocabulary and accessible explanations.",
        "ar": "مستوى مبتدئ: مفردات بسيطة وشروح ميسّرة.",
    },
    "intermediate": {
        "fr": "Niveau intermédiaire : vocabulaire technique du cours, explications structurées.",
        "en": "Intermediate level: technical course vocabulary and structured explanations.",
        "ar": "مستوى متوسط: مفردات تقنية وشروح منظمة.",
    },
    "advanced": {
        "fr": "Niveau avancé : précision terminologique et explications approfondies, sans invention.",
        "en": "Advanced level: terminological precision and deeper explanations, without invention.",
        "ar": "مستوى متقدم: دقة اصطلاحية وشروح أعمق دون اختراع.",
    },
}
_LANGUAGE_HINTS = {
    "fr": "Rédige entièrement en français.",
    "en": "Write entirely in English.",
    "ar": "اكتب بالكامل باللغة العربية.",
}


class OriginalContentGenerator:
    """Transforme CourseModel + PedagogicalPlan en PedagogicalDocument rédigé."""

    def __init__(self, llm: object, guard: LlmBudgetGuard) -> None:
        if not callable(getattr(llm, "generate", None)):
            raise TypeError("llm doit fournir generate.")
        self.llm = llm
        self.guard = guard

    def generate(
        self,
        course: CourseModel,
        plan: PedagogicalPlan,
        *,
        filename: str,
        page_start: int,
        page_end: int,
        source_pages: list[CoursePage] | None = None,
    ) -> PedagogicalDocument:
        if not isinstance(course, CourseModel):
            raise TypeError("course doit être un CourseModel.")
        if not isinstance(plan, PedagogicalPlan):
            raise TypeError("plan doit être un PedagogicalPlan.")
        if not plan.chapters:
            raise ValueError("Le plan pédagogique est incomplet.")

        pages = list(source_pages or [])
        source_text = _source_corpus(course, pages)
        previous_titles: list[str] = []
        drafts: list[ChapterDraft] = []

        for chapter_plan in plan.chapters:
            draft = self._generate_chapter(
                course=course,
                plan=plan,
                chapter_plan=chapter_plan,
                previous_titles=previous_titles,
                source_pages=pages,
                source_text=source_text,
            )
            drafts.append(draft)
            previous_titles.append(draft.title)

        return PedagogicalDocument(
            filename=filename,
            page_start=page_start,
            page_end=page_end,
            language=plan.language,
            level=plan.level,
            sheet_type=plan.sheet_type,
            chapters=drafts,
        )

    def _generate_chapter(
        self,
        *,
        course: CourseModel,
        plan: PedagogicalPlan,
        chapter_plan: ChapterPlan,
        previous_titles: list[str],
        source_pages: list[CoursePage],
        source_text: str,
    ) -> ChapterDraft:
        tokens = (
            TOKENS_CHAPTER_SHORT
            if plan.sheet_type == "short"
            else TOKENS_CHAPTER_COMPLETE
        )
        prompt = _build_chapter_prompt(
            course=course,
            plan=plan,
            chapter_plan=chapter_plan,
            previous_titles=previous_titles,
            source_pages=source_pages,
        )

        def produce() -> str:
            ensure_prompt_fits(self.llm, prompt, tokens)
            return self.llm.generate(prompt, max_new_tokens=tokens)

        draft = self.guard.run_with_retries(
            produce,
            tokens=tokens,
            parse=lambda raw: _parse_chapter_draft(raw, expected_index=chapter_plan.index),
        )
        _validate_chapter_draft(
            draft,
            chapter_plan=chapter_plan,
            course=course,
            source_text=source_text,
            sheet_type=plan.sheet_type,
        )
        return draft


def _build_chapter_prompt(
    *,
    course: CourseModel,
    plan: PedagogicalPlan,
    chapter_plan: ChapterPlan,
    previous_titles: list[str],
    source_pages: list[CoursePage],
) -> str:
    language = plan.language if plan.language in _LANGUAGE_HINTS else "fr"
    level = plan.level if plan.level in _LEVEL_HINTS else "beginner"
    sheet = plan.sheet_type
    level_line = _LEVEL_HINTS[level].get(language, _LEVEL_HINTS[level]["fr"])
    language_line = _LANGUAGE_HINTS[language]

    concepts = [c for c in course.concepts if c.id in set(chapter_plan.concept_ids)]
    concepts_json = [
        {
            "id": c.id,
            "label": c.label,
            "kind": c.kind,
            "notes": c.notes,
            "pages": c.pages,
            "depends_on": c.depends_on,
            "essential": c.essential,
        }
        for c in concepts
    ]
    relations = [
        {
            "source_id": r.source_id,
            "target_id": r.target_id,
            "relation_type": r.relation_type,
        }
        for r in course.relations
        if r.source_id in chapter_plan.concept_ids or r.target_id in chapter_plan.concept_ids
    ]
    page_excerpts = _page_excerpts(source_pages, chapter_plan.page_refs, concepts)

    if sheet == "short":
        style = (
            "Fiche short : explications essentielles, synthétiques mais compréhensibles ; "
            "exemples indispensables ; synthèse courte ; pas de questions de révision."
        )
    else:
        style = (
            "Fiche complete : développe concepts, relations et mécanismes ; "
            "exemples pertinents ; tableau ou schéma si utile ; questions de révision."
        )

    previous = ", ".join(previous_titles) if previous_titles else "(aucun)"
    plan_titles = " > ".join(ch.title for ch in plan.chapters)

    return (
        "Tu es EduRAG. Rédige UN chapitre de cours pédagogique en JSON uniquement.\n"
        f"{language_line}\n"
        f"{level_line}\n"
        f"{style}\n"
        "Règles strictes :\n"
        "- Reformule de façon originale : n'copie pas le PDF.\n"
        "- N'invente aucune formule, valeur numérique, définition technique ou fait "
        "scientifique absent des pages/concepts fournis.\n"
        "- Distingue origin=from_course et origin=pedagogical_addition pour les exemples.\n"
        "- Si une CONTRADICTION apparaît dans les notes, signale-la sans la résoudre.\n"
        "- Interdit : URL, bibliographie, contact, remerciements, listes de mots-clés vides.\n"
        "Schéma JSON (objet unique, JSON complet uniquement, sans Markdown et sans fence "
        "code) :\n"
        "Structure obligatoire : index (entier), title (texte), concept_ids_covered (liste "
        "d'identifiants), blocks (liste de blocs).\n"
        "Chaque bloc doit avoir un seul champ type exact parmi : paragraph, definition, "
        "example, retain, table, diagram, synthesis, question.\n"
        "Interdit : mettre plusieurs types dans une seule valeur type, ou utiliser le "
        "caractère | pour combiner des types.\n"
        "Champs obligatoires par type :\n"
        "- paragraph, definition, diagram, retain, synthesis, question : type, text\n"
        "- example : type, text, origin "
        "(origin exact : from_course ou pedagogical_addition)\n"
        "- table : type, headers, rows\n"
        "Exemples de blocs valides :\n"
        '{"type":"paragraph","text":"..."}\n'
        '{"type":"definition","text":"..."}\n'
        '{"type":"example","text":"...","origin":"from_course"}\n'
        '{"type":"retain","text":"..."}\n'
        '{"type":"table","headers":["A","B"],"rows":[["1","2"]]}\n'
        '{"type":"diagram","text":"..."}\n'
        '{"type":"synthesis","text":"..."}\n'
        '{"type":"question","text":"..."}\n'
        "Exemple d'objet chapitre : "
        '{"index":0,"title":"...","concept_ids_covered":["c1"],'
        '"blocks":[{"type":"paragraph","text":"..."}]}\n'
        f"Plan global : {plan_titles}\n"
        f"Chapitres déjà rédigés : {previous}\n"
        f"Chapitre à rédiger : index={chapter_plan.index} ; titre={chapter_plan.title}\n"
        f"Objectifs : {json.dumps([o.model_dump() for o in chapter_plan.objectives], ensure_ascii=False)}\n"
        f"needs_example={chapter_plan.needs_example} ; needs_table={chapter_plan.needs_table} ; "
        f"needs_questions={chapter_plan.needs_questions}\n"
        f"Concepts : {json.dumps(concepts_json, ensure_ascii=False)}\n"
        f"Relations : {json.dumps(relations, ensure_ascii=False)}\n"
        f"Extraits de pages :\n{page_excerpts}"
    )


def _page_excerpts(
    source_pages: list[CoursePage],
    page_refs: list[int],
    concepts: list[Concept],
) -> str:
    wanted = set(page_refs)
    for concept in concepts:
        wanted.update(concept.pages)
    selected = [p for p in source_pages if p.page_number in wanted] if wanted else list(source_pages)
    if not selected and source_pages:
        selected = source_pages
    if not selected:
        notes = "\n".join(f"- {c.label}: {c.notes}" for c in concepts if c.notes)
        return notes or "(aucun extrait)"
    blocks = [f"[Page {page.page_number}]\n{page.text}" for page in selected]
    return "\n\n".join(blocks)


def _parse_chapter_draft(raw: str, *, expected_index: int) -> ChapterDraft:
    try:
        payload = _load_json_object(raw)
    except ValueError as error:
        raise ValueError("JSON de chapitre invalide.") from error

    payload["index"] = expected_index
    try:
        if "blocks" in payload and isinstance(payload["blocks"], list):
            normalized_blocks = []
            for item in payload["blocks"]:
                if not isinstance(item, dict):
                    raise ValueError("JSON de chapitre invalide.")
                normalized_blocks.append(ContentBlock.model_validate(item))
            payload["blocks"] = [block.model_dump() for block in normalized_blocks]
        draft = ChapterDraft.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as error:
        raise ValueError("JSON de chapitre invalide.") from error

    if draft.index != expected_index:
        raise ValueError("JSON de chapitre invalide.")
    return draft


def _validate_chapter_draft(
    draft: ChapterDraft,
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
    source_text: str,
    sheet_type: str,
) -> None:
    if not draft.blocks:
        raise ValueError("Le chapitre généré est vide ou incomplet.")

    required = set(chapter_plan.concept_ids)
    covered = set(draft.concept_ids_covered)
    if required and not required.issubset(covered):
        raise ValueError(
            "Le chapitre ne couvre pas les concepts essentiels demandés par le plan."
        )

    if sheet_type == "short":
        if any(block.type == "question" for block in draft.blocks):
            raise ValueError("Une fiche short ne doit pas contenir de questions de révision.")

    concept_notes = " ".join(
        c.notes for c in course.concepts if c.id in required
    )
    has_contradiction = "CONTRADICTION" in concept_notes.upper()

    for block in draft.blocks:
        texts = _block_texts(block)
        for text in texts:
            if _URL_RE.search(text):
                raise ValueError("Contenu interdit : URL inventée ou non autorisée.")
            if isinstance(block, ParagraphBlock) and _is_near_copy(text, source_text):
                raise ValueError(
                    "Le paragraphe est trop proche d'une copie du document ; "
                    "une rédaction originale est exigée."
                )
            if has_contradiction and _RESOLVING_RE.search(text):
                raise ValueError(
                    "Résolution inventée d'une contradiction : le générateur doit "
                    "signaler sans trancher."
                )
            invented = _invented_numbers(text, source_text)
            # Les ajouts pédagogiques peuvent illustrer sans chiffre technique nouveau
            # attribué au cours ; on refuse les chiffres nouveaux hors pedagogical_addition.
            if invented and not (
                isinstance(block, ExampleBlock)
                and block.origin == "pedagogical_addition"
            ):
                raise ValueError(
                    "invention détectée : valeur numérique absente des sources du cours."
                )


def _block_texts(block: object) -> list[str]:
    texts: list[str] = []
    text = getattr(block, "text", None)
    if isinstance(text, str) and text.strip():
        texts.append(text)
    headers = getattr(block, "headers", None)
    if isinstance(headers, list):
        texts.extend(str(item) for item in headers)
    rows = getattr(block, "rows", None)
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, list):
                texts.extend(str(cell) for cell in row)
    return texts


def _invented_numbers(text: str, source_text: str) -> list[str]:
    source_nums = {m.group(0) for m in _NUMBER_RE.finditer(source_text)}
    found = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        if token not in source_nums:
            found.append(token)
    return found


def _is_near_copy(text: str, source_text: str) -> bool:
    normalized = _normalize(text)
    source_norm = _normalize(source_text)
    if len(normalized) < 40:
        return False
    if normalized in source_norm:
        return True
    # Recouvrement fort par fenêtres de mots
    words = normalized.split()
    if len(words) < 8:
        return False
    window = 8
    for start in range(0, len(words) - window + 1):
        chunk = " ".join(words[start : start + window])
        if chunk in source_norm:
            return True
    return False


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _source_corpus(course: CourseModel, pages: list[CoursePage]) -> str:
    parts: list[str] = [page.text for page in pages]
    parts.extend(unit.text for unit in course.units)
    parts.extend(concept.notes for concept in course.concepts if concept.notes)
    parts.extend(concept.label for concept in course.concepts)
    return "\n".join(parts)


def _load_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("JSON invalide.") from error
    if not isinstance(data, dict):
        raise ValueError("JSON invalide.")
    return data
