"""CourseUnderstanding : analyse multi-fenêtres et fusion en CourseModel."""

from __future__ import annotations

import json
import re
from typing import Any

from src.revision.engine.llm_guard import (
    TOKENS_UNDERSTANDING,
    LlmBudgetGuard,
    ensure_pipeline_capacity,
    ensure_prompt_fits,
    max_attempts_per_step,
)
from src.revision.engine.models import (
    Concept,
    CourseModel,
    CoursePage,
    CourseRelation,
    CourseUnit,
)

_KIND_VALUES = frozenset(
    {"definition", "formula", "method", "mechanism", "relation", "other"}
)
# Alias fréquents inventés par le LLM pour des notions cloud / outils.
# Ils restent hors du modèle public ConceptKind et sont ramenés à une valeur canonique.
_KIND_ALIASES: dict[str, str] = {
    "service": "other",
    "technology": "other",
    "tool": "other",
    "concept": "other",
    "platform": "other",
    "model": "other",
    "process": "method",
    "equation": "formula",
    "link": "relation",
}
_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


class CourseUnderstanding:
    """Construit un CourseModel structuré à partir des fenêtres du cours."""

    def __init__(self, llm: object, guard: LlmBudgetGuard) -> None:
        if not callable(getattr(llm, "generate", None)):
            raise TypeError("llm doit fournir generate.")
        self.llm = llm
        self.guard = guard

    def analyze(self, windows: list[list[CoursePage]]) -> CourseModel:
        if not isinstance(windows, list) or not windows:
            raise ValueError("Les fenêtres du cours sont invalides.")
        for window in windows:
            if not window:
                raise ValueError("Les fenêtres du cours sont invalides.")
        # Vérifie fenêtres + budget global (retries inclus) avant le premier appel LLM.
        ensure_pipeline_capacity(windows=len(windows))
        attempts = max_attempts_per_step()
        need_calls = attempts * len(windows)
        need_tokens = need_calls * TOKENS_UNDERSTANDING
        if self.guard.remaining_calls < need_calls:
            raise ValueError(
                f"Le plafond d'appels LLM ({self.guard.calls + self.guard.remaining_calls}) "
                "est atteint."
            )
        if self.guard.remaining_tokens < need_tokens:
            raise ValueError(
                f"Le plafond de tokens générés "
                f"({self.guard.reserved_tokens + self.guard.remaining_tokens}) est atteint."
            )

        partials: list[dict[str, Any]] = []
        for window in windows:
            partials.append(self._analyze_window(window))
        return self._merge(partials, windows)

    def _analyze_window(self, window: list[CoursePage]) -> dict[str, Any]:
        prompt = _build_understanding_prompt(window)

        def produce() -> str:
            if callable(getattr(self.llm, "is_loaded", None)) and self.llm.is_loaded():
                ensure_prompt_fits(self.llm, prompt, TOKENS_UNDERSTANDING)
            return self.llm.generate(prompt, max_new_tokens=TOKENS_UNDERSTANDING)

        return self.guard.run_with_retries(
            produce,
            tokens=TOKENS_UNDERSTANDING,
            parse=_parse_window_json,
        )

    def _merge(
        self,
        partials: list[dict[str, Any]],
        windows: list[list[CoursePage]],
    ) -> CourseModel:
        themes: list[str] = []
        concepts_by_id: dict[str, Concept] = {}
        # Correspondance globale alias → id canonique (chaînes same_as successives).
        alias_map: dict[str, str] = {}
        relations: list[CourseRelation] = []
        # Contradictions différées : appliquées après alias_map complet.
        pending_contradictions: list[object] = []
        units = _units_from_windows(windows)

        for partial in partials:
            for theme in partial.get("themes", []):
                if theme not in themes:
                    themes.append(theme)

            for raw in partial.get("concepts", []):
                concept = _concept_from_raw(raw)
                same_as = str(raw.get("same_as") or "").strip()
                if same_as:
                    canonical_id = _resolve_id(same_as, alias_map)
                else:
                    canonical_id = _resolve_id(concept.id, alias_map)

                if canonical_id in concepts_by_id:
                    if concept.id != canonical_id:
                        alias_map[concept.id] = canonical_id
                    _merge_into(concepts_by_id[canonical_id], concept, raw)
                elif concept.id in concepts_by_id:
                    # same_as pointe vers un id encore inconnu : conserver l'entrée existante.
                    _merge_into(concepts_by_id[concept.id], concept, raw)
                else:
                    # same_as vers un id futur → stocker sous l'id canonique demandé.
                    if concept.id != canonical_id:
                        alias_map[concept.id] = canonical_id
                        concept = concept.model_copy(update={"id": canonical_id})
                    concepts_by_id[concept.id] = concept

            for raw_rel in partial.get("relations", []):
                relations.append(_relation_from_raw(raw_rel))

            for contradiction in partial.get("contradictions", []):
                pending_contradictions.append(contradiction)

        concepts_by_id, relations = _finalize_identity_graph(
            concepts_by_id, relations, alias_map
        )
        for contradiction in pending_contradictions:
            _apply_contradiction(concepts_by_id, contradiction, alias_map)
        concepts = list(concepts_by_id.values())
        essential_ids = [c.id for c in concepts if c.essential]
        return CourseModel(
            units=units,
            concepts=concepts,
            relations=relations,
            themes=themes,
            essential_concept_ids=essential_ids,
        )


def _build_understanding_prompt(window: list[CoursePage]) -> str:
    pages_block = "\n\n".join(
        f"[Page {page.page_number}]\n{page.text}" for page in window
    )
    allowed_kinds = ", ".join(sorted(_KIND_VALUES))
    return (
        "Tu es EduRAG. Analyse uniquement les pages fournies.\n"
        "Réponds UNIQUEMENT en JSON valide, sans Markdown.\n"
        "Respecte exactement le schéma JSON demandé.\n"
        "Schéma attendu :\n"
        "{"
        '"themes": ["..."], '
        '"concepts": [{"id":"...", "label":"...", "kind":"definition|formula|method|mechanism|relation|other", '
        '"pages":[1], "essential":true, "depends_on":[], "notes":"...", "same_as":null}], '
        '"relations": [{"source_id":"...", "target_id":"...", "relation_type":"...", "pages":[1]}], '
        '"contradictions": [{"concept_id":"...", "description":"...", "pages":[1]}]'
        "}\n"
        f"Le champ kind doit obligatoirement être l'une de ces six valeurs : {allowed_kinds}.\n"
        "Aucune autre valeur n'est autorisée pour kind.\n"
        "Pour une notion de type service informatique (PaaS, IaaS, SaaS), utilise kind=\"other\".\n"
        "Ne jamais inventer de catégorie comme service, technology, concept ou tool.\n"
        "Extraction pédagogique :\n"
        "- N'extrais une notion que si les pages la définissent ou l'expliquent vraiment "
        "(pas un simple titre, sommaire ou mention isolée).\n"
        "- Dans notes, reporte la définition / mécanisme / propriétés utiles tels que le "
        "cours les expose ; conserve synonymes, acronymes et formulations alternatives "
        "(ex. « aussi appelé … »).\n"
        "- pages doit lister toutes les pages de CETTE fenêtre où la notion est expliquée.\n"
        "- Si la même notion réapparaît sous un autre id déjà vu, renseigne same_as avec "
        "l'id canonique ; n'invente pas de fusion entre notions proches mais distinctes.\n"
        "- contradictions : uniquement des oppositions factuelles explicites du cours "
        "(pas de simples reformulations complémentaires).\n"
        "Règles : n'invente rien ; same_as seulement si la même notion est déjà identifiée ; "
        "ne fusionne pas deux notions proches mais distinctes.\n"
        f"Pages :\n{pages_block}"
    )


def _parse_window_json(raw: str) -> dict[str, Any]:
    try:
        payload = _load_json_object(raw)
    except ValueError as error:
        raise ValueError("JSON de compréhension invalide.") from error

    if not isinstance(payload.get("themes"), list):
        raise ValueError("JSON de compréhension invalide.")
    if not isinstance(payload.get("concepts"), list):
        raise ValueError("JSON de compréhension invalide.")
    if "relations" in payload and not isinstance(payload.get("relations"), list):
        raise ValueError("JSON de compréhension invalide.")
    if "contradictions" in payload and not isinstance(payload.get("contradictions"), list):
        raise ValueError("JSON de compréhension invalide.")

    payload.setdefault("relations", [])
    payload.setdefault("contradictions", [])
    # Validation légère des concepts
    for concept in payload["concepts"]:
        if not isinstance(concept, dict):
            raise ValueError("JSON de compréhension invalide.")
        for key in ("id", "label", "kind", "pages"):
            if key not in concept:
                raise ValueError("JSON de compréhension invalide.")
    return payload


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


def _normalize_kind(raw_kind: object) -> tuple[str | None, str | None]:
    """Retourne (kind_canonique, alias_d_origine_ou_None)."""
    if not isinstance(raw_kind, str):
        return None, None
    cleaned = raw_kind.strip()
    if not cleaned:
        return None, None
    lowered = cleaned.casefold()
    if lowered in _KIND_VALUES:
        return lowered, None
    aliased = _KIND_ALIASES.get(lowered)
    if aliased is not None:
        return aliased, cleaned
    return None, cleaned


def _notes_with_origin(notes: str, origin_kind: str | None) -> str:
    """Conserve l'information du kind LLM d'origine sans écraser les notes."""
    base = notes.strip()
    if not origin_kind:
        return base
    marker = f"Catégorie LLM d'origine : {origin_kind}"
    if marker.casefold() in base.casefold():
        return base
    if not base:
        return marker
    return f"{base} ({marker})"


def _concept_from_raw(raw: dict[str, Any]) -> Concept:
    concept_id = str(raw.get("id") or "").strip() or "(inconnu)"
    kind, origin_kind = _normalize_kind(raw.get("kind"))
    if kind is None or kind not in _KIND_VALUES:
        received = raw.get("kind")
        allowed = ", ".join(sorted(_KIND_VALUES))
        raise ValueError(
            f"kind invalide pour le concept {concept_id!r} : reçu {received!r}. "
            f"Valeurs autorisées : {allowed}."
        )
    pages = _int_list(raw.get("pages"))
    depends_on = raw.get("depends_on") or []
    if not isinstance(depends_on, list):
        raise ValueError("JSON de compréhension invalide.")
    notes = _notes_with_origin(str(raw.get("notes") or ""), origin_kind)
    return Concept(
        id=str(raw["id"]).strip(),
        label=str(raw["label"]).strip(),
        kind=kind,  # type: ignore[arg-type]
        pages=pages,
        essential=bool(raw.get("essential", False)),
        depends_on=[str(item).strip() for item in depends_on if str(item).strip()],
        notes=notes,
    )


def _relation_from_raw(raw: dict[str, Any]) -> CourseRelation:
    if not isinstance(raw, dict):
        raise ValueError("JSON de compréhension invalide.")
    return CourseRelation(
        source_id=str(raw.get("source_id") or "").strip(),
        target_id=str(raw.get("target_id") or "").strip(),
        relation_type=str(raw.get("relation_type") or "").strip() or "related",
        pages=_int_list(raw.get("pages")),
    )


def _int_list(value: object) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("JSON de compréhension invalide.")
    pages: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError("JSON de compréhension invalide.")
        if item not in pages:
            pages.append(item)
    return pages


def _merge_into(existing: Concept, incoming: Concept, raw: dict[str, Any]) -> None:
    existing.pages = _union_sorted(existing.pages, incoming.pages)
    existing.essential = existing.essential or incoming.essential
    existing.depends_on = _union_preserve(existing.depends_on, incoming.depends_on)
    # Preférer un label plus informatif si l'existant est trop court / générique.
    if incoming.label and (
        not existing.label
        or (
            len(incoming.label.strip()) > len(existing.label.strip())
            and _normalize_note(existing.label) in _normalize_note(incoming.label)
        )
    ):
        existing.label = incoming.label
    existing.notes = _merge_notes(existing.notes, incoming.notes)


def _note_content_lines(notes: str) -> list[str]:
    """Découpe les notes en lignes utiles (métadonnées LLM seules exclues du merge)."""
    lines: list[str] = []
    for raw in (notes or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        lines.append(line)
    return lines


def _is_metadata_only_notes(notes: str) -> bool:
    """True si les notes ne portent que le marqueur de catégorie LLM."""
    residual = (notes or "").strip()
    if not residual:
        return True
    residual = re.sub(
        r"\(Catégorie LLM d'origine\s*:[^)]+\)",
        " ",
        residual,
        flags=re.IGNORECASE,
    )
    residual = re.sub(
        r"^Catégorie LLM d'origine\s*:\s*.+$",
        " ",
        residual,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return not residual.strip()


def _merge_notes(existing: str, incoming: str) -> str:
    """Fusionne des notes complémentaires sans inventer de contradiction.

    Les contradictions réelles restent du ressort de `contradictions` / `_apply_contradiction`.
    """
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    # Si l'existant n'est que de la métadonnée, partir du fond pédagogique entrant.
    if _is_metadata_only_notes(existing) and not _is_metadata_only_notes(incoming):
        return _merge_notes(incoming, existing)
    if _normalize_note(existing) == _normalize_note(incoming):
        return existing

    lines = _note_content_lines(existing)
    normalized_existing = [_normalize_note(line) for line in lines]

    for line in _note_content_lines(incoming):
        norm = _normalize_note(line)
        if not norm:
            continue
        if norm in normalized_existing:
            continue
        # Sous-chaîne déjà couverte par une ligne plus riche.
        if any(norm in other and norm != other for other in normalized_existing):
            continue
        # Remplacer une ligne plus pauvre déjà présente.
        replaced = False
        for index, other in enumerate(normalized_existing):
            if (
                other in norm
                and other != norm
                and "CONTRADICTION" not in lines[index].upper()
            ):
                lines[index] = line
                normalized_existing[index] = norm
                replaced = True
                break
        if not replaced:
            lines.append(line)
            normalized_existing.append(norm)

    return "\n".join(lines)


def _resolve_id(concept_id: str, alias_map: dict[str, str]) -> str:
    """Suit la chaîne d'alias jusqu'à l'identifiant canonique."""
    if not concept_id:
        return concept_id
    seen: set[str] = set()
    current = concept_id
    while current in alias_map and current not in seen:
        seen.add(current)
        current = alias_map[current]
    return current


def _apply_contradiction(
    concepts: dict[str, Concept],
    raw: object,
    alias_map: dict[str, str] | None = None,
) -> None:
    if not isinstance(raw, dict):
        return
    concept_id = str(raw.get("concept_id") or "").strip()
    if alias_map is not None:
        concept_id = _resolve_id(concept_id, alias_map)
    description = str(raw.get("description") or "").strip()
    pages = raw.get("pages")
    page_note = ""
    if isinstance(pages, list) and pages:
        page_note = f" (pages {', '.join(str(p) for p in pages)})"
    if concept_id not in concepts or not description:
        return
    concept = concepts[concept_id]
    marker = f"[CONTRADICTION]{page_note} {description}".strip()
    if marker.casefold() in concept.notes.casefold():
        return
    if concept.notes:
        concept.notes = f"{concept.notes}\n{marker}"
    else:
        concept.notes = marker
    extra_pages = []
    if isinstance(pages, list):
        for item in pages:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 1:
                extra_pages.append(item)
    concept.pages = _union_sorted(concept.pages, extra_pages)


def _finalize_identity_graph(
    concepts: dict[str, Concept],
    relations: list[CourseRelation],
    alias_map: dict[str, str],
) -> tuple[dict[str, Concept], list[CourseRelation]]:
    """Remappe depends_on / relations vers les ids canoniques et retire les orphelins."""
    known = set(concepts.keys())

    for concept in concepts.values():
        remapped: list[str] = []
        for dep in concept.depends_on:
            resolved = _resolve_id(dep, alias_map)
            if resolved in known and resolved != concept.id and resolved not in remapped:
                remapped.append(resolved)
        concept.depends_on = remapped

    remapped_relations: list[CourseRelation] = []
    for relation in relations:
        source = _resolve_id(relation.source_id, alias_map)
        target = _resolve_id(relation.target_id, alias_map)
        if source not in known or target not in known:
            continue
        if not source or not target:
            continue
        remapped_relations.append(
            CourseRelation(
                source_id=source,
                target_id=target,
                relation_type=relation.relation_type,
                pages=relation.pages,
            )
        )

    return concepts, _dedupe_relations(remapped_relations)


def _remap_relation(
    relation: CourseRelation,
    concepts: dict[str, Concept],
    partial: dict[str, Any],
    alias_map: dict[str, str] | None = None,
) -> CourseRelation:
    """Remappe une relation (conservé pour compatibilité / usage local)."""
    id_map: dict[str, str] = dict(alias_map or {})
    for raw in partial.get("concepts", []):
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("id") or "").strip()
        same_as = str(raw.get("same_as") or "").strip()
        if cid and same_as:
            id_map[cid] = same_as
    source = _resolve_id(relation.source_id, id_map)
    target = _resolve_id(relation.target_id, id_map)
    if source not in concepts:
        source = relation.source_id
    if target not in concepts:
        target = relation.target_id
    return CourseRelation(
        source_id=source,
        target_id=target,
        relation_type=relation.relation_type,
        pages=relation.pages,
    )


def _dedupe_relations(relations: list[CourseRelation]) -> list[CourseRelation]:
    merged: dict[tuple[str, str, str], CourseRelation] = {}
    for relation in relations:
        if not relation.source_id or not relation.target_id:
            continue
        key = (relation.source_id, relation.target_id, relation.relation_type)
        if key in merged:
            merged[key].pages = _union_sorted(merged[key].pages, relation.pages)
        else:
            merged[key] = relation.model_copy(deep=True)
    return list(merged.values())


def _units_from_windows(windows: list[list[CoursePage]]) -> list[CourseUnit]:
    units: list[CourseUnit] = []
    for window in windows:
        text = "\n\n".join(page.text for page in window).strip()
        if not text:
            continue
        units.append(
            CourseUnit(
                text=text,
                page_start=window[0].page_number,
                page_end=window[-1].page_number,
            )
        )
    return units


def _union_sorted(left: list[int], right: list[int]) -> list[int]:
    return sorted(set(left) | set(right))


def _union_preserve(left: list[str], right: list[str]) -> list[str]:
    result = list(left)
    for item in right:
        if item not in result:
            result.append(item)
    return result


def _normalize_note(text: str) -> str:
    return " ".join(text.casefold().split())
