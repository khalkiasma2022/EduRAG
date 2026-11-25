"""Réponses LLM scriptées compatibles avec le pipeline Smart Revision V2.

Utilisé par les tests applicatifs (assistant / intégration) qui simulent Qwen
sans télécharger le modèle.
"""

from __future__ import annotations

import json
import re


def revision_pipeline_reply(prompt: str) -> str | None:
    """Retourne une réponse JSON V2 si le prompt appartient au pipeline révision.

    Sinon renvoie None (le faux LLM peut alors répondre pour le RAG).
    """
    if "Analyse uniquement les pages fournies" in prompt:
        pages = [int(value) for value in re.findall(r"\[Page (\d+)\]", prompt)]
        page = pages[0] if pages else 1
        return json.dumps(
            {
                "themes": ["Cours"],
                "concepts": [
                    {
                        "id": f"c_page_{page}",
                        "label": f"notion page {page}",
                        "kind": "definition",
                        "pages": [page],
                        "essential": True,
                        "depends_on": [],
                        "notes": "Contenu fidèle extrait du cours fourni.",
                    }
                ],
                "relations": [],
                "contradictions": [],
            },
            ensure_ascii=False,
        )

    if "Construis un plan pédagogique en JSON" in prompt:
        sheet_type = "complete" if "type=complete" in prompt else "short"
        language = "fr"
        if "Langue=en" in prompt:
            language = "en"
        elif "Langue=ar" in prompt:
            language = "ar"
        level = "beginner"
        if "niveau=advanced" in prompt:
            level = "advanced"
        elif "niveau=intermediate" in prompt:
            level = "intermediate"
        concept_ids = re.findall(r'"id":\s*"(c_page_\d+)"', prompt)
        if not concept_ids:
            concept_ids = ["c_page_1"]
        # Un concept par chapitre, borné par short/complete.
        max_chapters = 4 if sheet_type == "short" else 6
        selected = concept_ids[:max_chapters]
        chapters = []
        for index, cid in enumerate(selected):
            page_match = re.search(r"c_page_(\d+)", cid)
            page = int(page_match.group(1)) if page_match else index + 1
            chapters.append(
                {
                    "index": index,
                    "title": f"Chapitre {index + 1}",
                    "objectives": [
                        {"text": f"Comprendre {cid}", "concept_ids": [cid]}
                    ],
                    "concept_ids": [cid],
                    "page_refs": [page],
                    "needs_example": True,
                    "needs_table": sheet_type == "complete",
                    "needs_questions": sheet_type == "complete",
                    "depth": sheet_type,
                }
            )
        return json.dumps(
            {
                "language": language,
                "level": level,
                "sheet_type": sheet_type,
                "chapters": chapters,
            },
            ensure_ascii=False,
        )

    if "Rédige UN chapitre de cours pédagogique" in prompt or "Réécris UNIQUEMENT le chapitre" in prompt:
        index_match = re.search(r"index=(\d+)", prompt)
        title_match = re.search(r"titre=([^\n]+)", prompt)
        index = int(index_match.group(1)) if index_match else 0
        title = title_match.group(1).strip() if title_match else f"Chapitre {index + 1}"
        concept_ids = re.findall(r'"id":\s*"(c_page_\d+)"', prompt)
        concept_id = concept_ids[0] if concept_ids else f"c_page_{index + 1}"
        page_match = re.search(r"c_page_(\d+)", concept_id)
        page = int(page_match.group(1)) if page_match else 1
        needs_questions = "needs_questions=True" in prompt
        # Aucun chiffre dans les textes : le QC refuse les nombres absents du PDF source.
        blocks = [
            {
                "type": "paragraph",
                "text": "Explication originale du chapitre pour la notion étudiée.",
                "source_pages": [page],
            },
            {
                "type": "definition",
                "text": "Contenu fidèle extrait du cours fourni.",
                "source_pages": [page],
            },
            {
                "type": "example",
                "text": "Analogie pédagogique : une boîte clairement étiquetée.",
                "origin": "pedagogical_addition",
                "source_pages": [],
            },
            {
                "type": "synthesis",
                "text": "Retenir l'essentiel du chapitre.",
                "source_pages": [page],
            },
        ]
        if needs_questions:
            blocks.append(
                {
                    "type": "question",
                    "text": "Que retenir de ce chapitre ?",
                    "source_pages": [page],
                }
            )
        return json.dumps(
            {
                "index": index,
                "title": title,
                "concept_ids_covered": [concept_id],
                "blocks": blocks,
            },
            ensure_ascii=False,
        )

    return None
