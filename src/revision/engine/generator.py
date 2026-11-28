"""OriginalContentGenerator : rédaction pédagogique chapitre par chapitre."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.revision.engine.json_util import load_json_object
from src.revision.engine.llm_guard import (
    TOKENS_CHAPTER,
    LlmBudgetGuard,
    ensure_chapter_density,
    ensure_generation_capacity,
    ensure_prompt_fits,
)
from src.revision.engine.models import (
    ChapterDraft,
    ChapterPlan,
    Concept,
    ContentBlock,
    CourseModel,
    CoursePage,
    DefinitionBlock,
    ExampleBlock,
    PedagogicalDocument,
    PedagogicalPlan,
    ParagraphBlock,
    SynthesisBlock,
)
from src.revision.engine.progress import log_progress

_BLOCK_TYPES = frozenset(
    {
        "paragraph",
        "definition",
        "example",
        "retain",
        "table",
        "diagram",
        "synthesis",
    }
)

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
        ensure_generation_capacity(self.guard, chapters=len(plan.chapters))

        pages = list(source_pages or [])
        source_text = _source_corpus(course, pages)
        previous_titles: list[str] = []
        drafts: list[ChapterDraft] = []

        total = len(plan.chapters)
        for chapter_plan in plan.chapters:
            log_progress(
                f"Rédaction chapitre {chapter_plan.index + 1}/{total} "
                f": {chapter_plan.title}"
            )
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
        # Refuse avant appel LLM si la densité dépasse le budget 640 tokens.
        ensure_chapter_density(len(chapter_plan.concept_ids))
        tokens = TOKENS_CHAPTER
        prompt = _build_chapter_prompt(
            course=course,
            plan=plan,
            chapter_plan=chapter_plan,
            previous_titles=previous_titles,
            source_pages=source_pages,
        )
        retry_state: dict[str, str | None] = {"error": None}

        def produce() -> str:
            full_prompt = prompt
            if retry_state["error"]:
                full_prompt = (
                    f"{prompt}\n"
                    "CORRECTION OBLIGATOIRE (essai précédent refusé) :\n"
                    f"- Erreur : {retry_state['error']}\n"
                    "- Produis un objet JSON complet et fermé "
                    '(index, title, concept_ids_covered, blocks).\n'
                    "- Sans Markdown, sans questions, sans URL.\n"
                )
                log_progress(
                    f"Nouvel essai chapitre {chapter_plan.index + 1} "
                    f"après refus…"
                )
            ensure_prompt_fits(self.llm, full_prompt, tokens)
            return self.llm.generate(full_prompt, max_new_tokens=tokens)

        def parse(raw: str) -> ChapterDraft:
            try:
                draft = _parse_chapter_draft(raw, expected_index=chapter_plan.index)
                draft = _enrich_concept_ids_covered(
                    draft,
                    chapter_plan=chapter_plan,
                    course=course,
                )
                _validate_chapter_draft(
                    draft,
                    chapter_plan=chapter_plan,
                    course=course,
                    source_text=source_text,
                )
            except ValueError as error:
                retry_state["error"] = str(error)
                log_progress(
                    f"Chapitre {chapter_plan.index + 1} refusé : {error}"
                )
                raise
            return draft

        try:
            return self.guard.run_with_retries(
                produce,
                tokens=tokens,
                parse=parse,
            )
        except ValueError as error:
            # Qwen tronque souvent le JSON : on termine quand même le chapitre.
            if "JSON" not in str(error):
                raise
            log_progress(
                f"Chapitre {chapter_plan.index + 1} : "
                "JSON inutilisable — fiche de secours sans contrainte de concepts…"
            )
            return _fallback_chapter_draft(
                chapter_plan=chapter_plan,
                course=course,
            )


def _build_chapter_prompt(
    *,
    course: CourseModel,
    plan: PedagogicalPlan,
    chapter_plan: ChapterPlan,
    previous_titles: list[str],
    source_pages: list[CoursePage],
) -> str:
    language = plan.language if plan.language in _LANGUAGE_HINTS else "fr"
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

    style = (
        "Fiche de révision claire et structurée : définitions, explications, "
        "exemples utiles si besoin, synthèse courte. "
        "Interdit : questions de révision, quiz, type question. "
        "Aucune obligation de couvrir une liste de concepts."
    )

    previous = ", ".join(previous_titles) if previous_titles else "(aucun)"
    plan_titles = " > ".join(ch.title for ch in plan.chapters)

    return (
        "Tu es EduRAG. Rédige UN chapitre de cours pédagogique en JSON uniquement.\n"
        f"{language_line}\n"
        f"{style}\n"
        "Règles strictes :\n"
        "- Reformule de façon originale : n'copie pas le PDF.\n"
        "- N'invente aucune formule, valeur numérique, définition technique ou fait "
        "scientifique absent des pages fournies.\n"
        "- Distingue origin=from_course et origin=pedagogical_addition pour les exemples.\n"
        "- Si une CONTRADICTION apparaît dans les notes, signale-la sans la résoudre.\n"
        "- Interdit : URL, bibliographie, contact, remerciements, listes de mots-clés vides.\n"
        "- Interdit : toute question destinée à l'étudiant (type question, quiz, QCM).\n"
        "- JSON compact et COMPLET : ferme toutes les accolades et guillemets.\n"
        "Schéma JSON (objet unique, sans Markdown) :\n"
        "index (entier), title (texte), concept_ids_covered (liste, peut être vide), "
        "blocks (liste).\n"
        "Types de bloc autorisés : paragraph, definition, example, retain, table, "
        "diagram, synthesis.\n"
        "Champs : paragraph/definition/diagram/retain/synthesis → type, text ; "
        "example → type, text, origin ; table → type, headers, rows.\n"
        "Exemple : "
        '{"index":0,"title":"...","concept_ids_covered":[],'
        '"blocks":[{"type":"paragraph","text":"..."},'
        '{"type":"synthesis","text":"..."}]}\n'
        f"Plan global : {plan_titles}\n"
        f"Chapitres déjà rédigés : {previous}\n"
        f"Chapitre à rédiger : index={chapter_plan.index} ; titre={chapter_plan.title}\n"
        f"Objectifs : {json.dumps([o.model_dump() for o in chapter_plan.objectives], ensure_ascii=False)}\n"
        f"needs_example={chapter_plan.needs_example} ; needs_table={chapter_plan.needs_table} ; "
        "needs_questions=false\n"
        f"Matériau optionnel (ne bloque pas si omis) : "
        f"{json.dumps(concepts_json, ensure_ascii=False)}\n"
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
        payload = load_json_object(raw)
    except ValueError as error:
        raise ValueError("JSON de chapitre invalide.") from error

    payload = _coerce_chapter_payload(payload, expected_index=expected_index)
    try:
        draft = ChapterDraft.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as error:
        raise ValueError("JSON de chapitre invalide.") from error

    if draft.index != expected_index:
        raise ValueError("JSON de chapitre invalide.")
    if not draft.blocks:
        raise ValueError("Le chapitre généré est vide ou incomplet.")
    return draft


def _coerce_chapter_payload(
    payload: dict[str, Any],
    *,
    expected_index: int,
) -> dict[str, Any]:
    """Assouplit le JSON LLM : types manquants, exemples sans origin, blocs cassés."""
    coerced: dict[str, Any] = dict(payload)
    coerced["index"] = expected_index
    if not isinstance(coerced.get("title"), str) or not coerced["title"].strip():
        coerced["title"] = f"Chapitre {expected_index + 1}"
    covered = coerced.get("concept_ids_covered")
    if not isinstance(covered, list):
        coerced["concept_ids_covered"] = []
    else:
        coerced["concept_ids_covered"] = [
            cid for cid in covered if isinstance(cid, str) and cid
        ]

    raw_blocks = coerced.get("blocks")
    if not isinstance(raw_blocks, list):
        raw_blocks = []
    normalized: list[dict[str, Any]] = []
    for item in raw_blocks:
        block = _coerce_content_block(item)
        if block is not None:
            normalized.append(block)
    coerced["blocks"] = normalized
    return coerced


def _coerce_content_block(item: object) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    data = dict(item)
    block_type = data.get("type")
    if isinstance(block_type, str):
        block_type = block_type.strip().casefold()
    else:
        block_type = None

    if block_type == "question":
        return None
    if block_type not in _BLOCK_TYPES:
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        data = {"type": "paragraph", "text": text, "source_pages": data.get("source_pages", [])}
        block_type = "paragraph"

    if block_type == "example" and data.get("origin") not in {
        "from_course",
        "pedagogical_addition",
    }:
        data["origin"] = "pedagogical_addition"

    if block_type == "table":
        if not isinstance(data.get("headers"), list):
            data["headers"] = []
        if not isinstance(data.get("rows"), list):
            data["rows"] = []

    if "source_pages" in data and not isinstance(data["source_pages"], list):
        data["source_pages"] = []

    try:
        return ContentBlock.model_validate(data).model_dump()
    except (ValidationError, ValueError, TypeError):
        text = data.get("text")
        if isinstance(text, str) and text.strip():
            try:
                return ContentBlock.model_validate(
                    {"type": "paragraph", "text": text}
                ).model_dump()
            except (ValidationError, ValueError, TypeError):
                return None
        return None


def _fallback_chapter_draft(
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
) -> ChapterDraft:
    """Chapitre minimal à partir du titre / notes — sans contrainte de couverture."""
    concepts = [c for c in course.concepts if c.id in set(chapter_plan.concept_ids)]
    blocks: list[Any] = []
    for concept in concepts:
        notes = " ".join((concept.notes or "").split()).strip()
        if notes and "CONTRADICTION" not in notes.upper():
            text = f"{concept.label} : {notes}"
        else:
            text = (
                f"{concept.label} est une notion présentée dans le cours "
                "à retenir pour la révision."
            )
        pages = [p for p in concept.pages if isinstance(p, int) and p >= 1]
        blocks.append(DefinitionBlock(type="definition", text=text, source_pages=pages))
    if not blocks:
        blocks.append(
            ParagraphBlock(
                type="paragraph",
                text=(
                    "Ce chapitre reprend les points essentiels de "
                    f"« {chapter_plan.title} »."
                ),
                source_pages=list(chapter_plan.page_refs),
            )
        )
    blocks.append(
        SynthesisBlock(
            type="synthesis",
            text=f"Retenir les idées clés de « {chapter_plan.title} ».",
            source_pages=list(chapter_plan.page_refs),
        )
    )
    return ChapterDraft(
        index=chapter_plan.index,
        title=chapter_plan.title,
        blocks=blocks,
        concept_ids_covered=[],
    )


def _enrich_concept_ids_covered(
    draft: ChapterDraft,
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
) -> ChapterDraft:
    """Conserve les ids déclarés ; n'impose plus ceux du plan."""
    del course, chapter_plan
    declared = [cid for cid in draft.concept_ids_covered if isinstance(cid, str) and cid]
    return draft.model_copy(update={"concept_ids_covered": list(dict.fromkeys(declared))})


def _validate_chapter_draft(
    draft: ChapterDraft,
    *,
    chapter_plan: ChapterPlan,
    course: CourseModel,
    source_text: str,
) -> None:
    if not draft.blocks:
        raise ValueError("Le chapitre généré est vide ou incomplet.")

    ensure_chapter_density(len(chapter_plan.concept_ids))

    # Pas de validateur de couverture / concepts : la fiche peut omettre des notions.

    required = set(chapter_plan.concept_ids)
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

