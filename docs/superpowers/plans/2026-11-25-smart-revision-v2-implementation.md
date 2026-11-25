# Smart Revision V2 — Plan d’implémentation

> **For agentic workers:** THIS IS THE EXECUTION CHECKLIST. Suivre les phases dans l’ordre. TDD. Pas de commit sans autorisation explicite de l’utilisateur. Spec de référence : `docs/superpowers/specs/2026-11-25-smart-revision-v2-design.md`.

**Goal :** Remplacer la génération monolothic Smart Revision par un moteur pédagogique modulaire (approche A, sync), tout en conservant API, React, SQLite `content`, et exports.

**Architecture :** `RevisionEngine` orchestre Reader → Understanding → Planner → Generator → QC → Formatter ; PDF reconstruit depuis Markdown structuré via `ProfessionalPDFBuilder`.

**Tech Stack :** Python 3.12, Pydantic (déjà via FastAPI), Qwen local, fpdf2, pytest, FastAPI existant, React inchangé.

**Contraintes :** pas de jobs async ; pas de colonne `engine_payload` en V2 ; pas de nouvelle dépendance lourde sans justification ; pas de commit/push sans OK ; FR prioritaire.

**Limites numériques V2 (voir spec §8.3) :**
`MAX_ANALYSIS_WINDOWS=5`, `MAX_CHAPTERS_SHORT=4`, `MAX_CHAPTERS_COMPLETE=6`,
`TOKENS_UNDERSTANDING=384`, `TOKENS_PLAN=512`, `TOKENS_CHAPTER_SHORT=384`, `TOKENS_CHAPTER_COMPLETE=640`, `TOKENS_REWRITE=384`,
`MAX_RETRIES_PER_CALL=1`, `MAX_REWRITE_CHAPTERS=2`, `MAX_LLM_CALLS=16`, `MAX_TOTAL_NEW_TOKENS=7500` (maximum, pas une cible),
`MAX_COURSE_TOKENS_PER_WINDOW=1800`.
Totaux théoriques réservés (avant retries) : short **4736** ; complete **7040** (spec §8.3.1).
`llm_guard` fait respecter **simultanément** `MAX_LLM_CALLS` et `MAX_TOTAL_NEW_TOKENS` ; arrêt sans save si plafond atteint.
Trop volumineux → erreur explicite, **jamais** de troncature silencieuse.

---

## Cartographie des fichiers

| Fichier | Action | Responsabilité |
|---|---|---|
| `src/revision/engine/models.py` | Créer | IR Pydantic |
| `src/revision/engine/llm_guard.py` | Créer | Budgets, retries, plafond d’appels, fit contexte |
| `src/revision/engine/reader.py` | Créer | CourseReader |
| `src/revision/engine/understanding.py` | Créer | CourseUnderstanding + fusion |
| `src/revision/engine/planner.py` | Créer | PedagogicalPlanner |
| `src/revision/engine/generator.py` | Créer | OriginalContentGenerator |
| `src/revision/engine/quality.py` | Créer | QualityController |
| `src/revision/engine/formatter.py` | Créer | ContentFormatter (IR ↔ Markdown canonique) |
| `src/revision/engine/pdf_builder.py` | Créer | ProfessionalPDFBuilder |
| `src/revision/engine/facade.py` | Créer | `RevisionEngine.run` |
| `src/revision/engine/__init__.py` | Créer | Exports publics moteur |
| `src/revision/sheet.py` | Modifier | Déléguer `generate` → facade (compat) |
| `src/revision/export.py` | Modifier | PDF via builder ; MD reste metadata + content |
| `src/assistant.py` | Modifier minime | Si besoin pour brancher facade (idéalement via sheet) |
| `src/api/routes/revisions.py` | Modifier minime | Uniquement si mapping d’erreurs à préciser — **éviter** si messages déjà couverts |
| `tests/test_revision_engine_*.py` | Créer | Tests par module |
| `tests/test_revision_sheet.py` | Adapter | Remplacer hypothèses `calls == 1` / prompt monolothic |
| `tests/test_revision_export.py` / `test_revision_pdf_export.py` | Adapter | Nouveau rendu PDF / conventions MD |
| `tests/test_api_revisions.py` | Adapter léger | Succès 201 seulement si pipeline OK ; mocks multi-étapes |
| `tests/fixtures/courses/*.pdf` | Créer | PDF FR de test qualité |
| `frontend/**` | **Ne pas modifier** en V2 moteur (sauf bug bloquant hors scope) | — |
| SQLite schema | **Ne pas migrer** en V2 | `engine_payload` documenté seulement |

---

## Phase 0 — Préparation (documentation & garde-fous)

### Task 0.1 : Checklist de non-régression

**Files :** aucun code

- [ ] Relire la spec `2026-11-25-smart-revision-v2-design.md`
- [ ] Lister les tests actuels qui imposent `llm.calls == 1`
- [ ] Confirmer : pas de migration DB, pas de React, pas de commit

---

## Phase 1 — IR + CourseReader + llm_guard

**Goal :** fondations testables sans changer encore le comportement produit.

### Task 1.1 : Modèles Pydantic

**Files :**
- Create : `src/revision/engine/models.py`
- Test : `tests/test_revision_engine_models.py`

- [ ] Écrire les tests de validation (Concept, ChapterPlan, ContentBlock discriminé, PedagogicalDocument)
- [ ] Implémenter les modèles
- [ ] `pytest tests/test_revision_engine_models.py` → vert

### Task 1.2 : llm_guard

**Files :**
- Create : `src/revision/engine/llm_guard.py`
- Test : `tests/test_revision_engine_llm_guard.py`

- [ ] Encoder les constantes numériques de la spec §8.3 / §8.3.1
- [ ] Tests : budget invalide ; dépassement **isolé** de `MAX_LLM_CALLS` ; dépassement **isolé** de `MAX_TOTAL_NEW_TOKENS`
- [ ] Tests : les **deux** plafonds sont vérifiés avant chaque réservation d’appel
- [ ] Tests calculs short/complete : assert que le plafond théorique short (5×384+512+4×384+2×384 = **4736**) ≤ 7500
- [ ] Tests calculs short/complete : assert que le plafond théorique complete (5×384+512+6×640+2×384 = **7040**) ≤ 7500
- [ ] Tests : une réservation complete maximale (7040) est acceptée ; une réservation qui ferait dépasser 7500 est refusée
- [ ] Tests : JSON invalide → retry (compte dans `MAX_LLM_CALLS`) puis échec après `MAX_RETRIES_PER_CALL`
- [ ] Tests : `ensure_prompt_fits` par appel ; refus sans troncature
- [ ] Implémenter
- [ ] Pytest vert

### Task 1.3 : CourseReader

**Files :**
- Create : `src/revision/engine/reader.py`
- Test : `tests/test_revision_engine_reader.py`

- [ ] Réutiliser logique `read_pages` / `_find_document` / `_select_pages` (extraire depuis `sheet.py` sans dupliquer durablement)
- [ ] Tests : document entier, intervalle, doc absent
- [ ] Tests : fenêtrage respectant `MAX_COURSE_TOKENS_PER_WINDOW`
- [ ] Tests : si > `MAX_ANALYSIS_WINDOWS` nécessaires → `ValueError` explicite « moins de pages » (pas de coupe silencieuse)
- [ ] Pytest vert

**Checkpoint Phase 1 :** Reader + models + guard OK ; API inchangée.

---

## Phase 2 — ContentFormatter + contrat Markdown → PDF

**Goal :** le Markdown sauvegardé est la source de vérité pour reconstruire le PDF.

### Task 2.1 : Formatter IR → Markdown canonique

**Files :**
- Create : `src/revision/engine/formatter.py`
- Test : `tests/test_revision_engine_formatter.py`

- [ ] Tests **un cas par type de bloc** (spec §6.6) : paragraphe, `#`/`##`/`###`, Définition, À retenir, Exemple (cours), Exemple (ajout pédagogique), tableau, Schéma, Synthèse, Questions de révision
- [ ] Implémenter les marqueurs exacts de la spec
- [ ] Pytest vert

### Task 2.2 : Parse Markdown canonique → structure logique (pour PDF)

**Files :**
- Modify : `src/revision/engine/formatter.py` (parse inverse minimal)
- Test : `tests/test_revision_engine_formatter.py` (round-trip)

- [ ] Round-trip : chaque bloc émis est re-détecté après parse
- [ ] **Pas** de dépendance à l’IR en mémoire après coup (entrée = string Markdown seule)

### Task 2.3 : ProfessionalPDFBuilder (squelette)

**Files :**
- Create : `src/revision/engine/pdf_builder.py`
- Test : `tests/test_revision_engine_pdf_builder.py`

- [ ] Couverture, titres, encadrés (définition / retenir / exemples), tableaux, schéma textuel, synthèse, questions, pagination, palette
- [ ] Entrée = Markdown stocké (ou structure parsée depuis ce Markdown)
- [ ] Test : reconstruire PDF après `del` de tout objet IR
- [ ] Conserver police arabe existante pour non-régression export AR

**Checkpoint Phase 2 :** round-trip Markdown → PDF sans IR mémoire.

---

## Phase 3 — Understanding + Planner

### Task 3.1 : CourseUnderstanding

**Files :**
- Create : `src/revision/engine/understanding.py`
- Test : `tests/test_revision_engine_understanding.py`

- [ ] Mock LLM JSON par fenêtre
- [ ] Fusion : **ne pas** fusionner si labels similaires mais notions distinctes
- [ ] Fusion : union des pages conservée quand fusion légitime
- [ ] Contradiction détectée → marquée ; **pas** de résolution inventée
- [ ] Concepts essentiels / relations / dépendances / mécanismes
- [ ] Échec JSON après retries → exception claire

### Task 3.2 : PedagogicalPlanner

**Files :**
- Create : `src/revision/engine/planner.py`
- Test : `tests/test_revision_engine_planner.py`

- [ ] short ≤ `MAX_CHAPTERS_SHORT` ; complete ≤ `MAX_CHAPTERS_COMPLETE`
- [ ] short ≠ complete (profondeur)
- [ ] Respect des dépendances conceptuelles
- [ ] Validation Pydantic du plan
- [ ] Trop de matière pour les plafonds → erreur « moins de pages » (pas de coupe silencieuse)
- [ ] Fallback heuristique documenté ou échec propre

**Checkpoint Phase 3 :** CourseModel + PedagogicalPlan fiables sur fixtures texte.

---

## Phase 4 — Generator + QualityController + Facade

### Task 4.1 : OriginalContentGenerator

**Files :**
- Create : `src/revision/engine/generator.py`
- Test : `tests/test_revision_engine_generator.py`

- [ ] Un appel mocké par chapitre
- [ ] Contexte anti-répétition (titres précédents + plan)
- [ ] Interdiction paraphrase : test heuristique de similarité sur fixture (seuil)
- [ ] Exemples `pedagogical_addition` marqués

### Task 4.2 : QualityController

**Files :**
- Create : `src/revision/engine/quality.py`
- Test : `tests/test_revision_engine_quality.py`

- [ ] URL / biblio / sections interdites → fail
- [ ] Couverture essentiels
- [ ] Similarité : détecte un paragraphe quasi recopié ; **ne pas** flagger seulement jargon/formule/définition courte nécessaire
- [ ] Similarité **seule** ne suffit pas à valider la fidélité scientifique
- [ ] Rewrite ≤ `MAX_REWRITE_CHAPTERS` puis re-évalué
- [ ] Échec final → exception (pas de document « ok »)

### Task 4.3 : RevisionEngine facade

**Files :**
- Create : `src/revision/engine/facade.py`
- Create : `src/revision/engine/__init__.py`
- Test : `tests/test_revision_engine_facade.py`

- [ ] Orchestration complète avec LLM mocké multi-appels
- [ ] **Aucun** save dans la facade (retour dict plat seulement)
- [ ] Exception si QC fail → pas de dict succès
- [ ] Tests robustesse : JSON invalide retries épuisés ; dépassement `MAX_LLM_CALLS` / `MAX_TOTAL_NEW_TOKENS` ; échec d’un chapitre ; QC fail après correction

### Task 4.4 : Brancher `sheet.py` / assistant

**Files :**
- Modify : `src/revision/sheet.py`
- Possibly : `src/assistant.py`

- [ ] `generate` délègue à `RevisionEngine.run`
- [ ] `create_revision` : `save` **uniquement** si `generate` retourne
- [ ] Adapter `tests/test_revision_sheet.py` (plus de `calls == 1` monolothic)
- [ ] Adapter `tests/test_api_revisions.py` (mocks multi-étapes ; 201 seulement si OK)
- [ ] Test : échec pipeline → **aucune** ligne SQLite
- [ ] Test : exception pendant `POST /revisions` → verrou libéré (requête suivante exclusive peut acquérir le lock — pattern déjà couvert / à étendre dans `test_api_revisions`)
- [ ] Non-régression : GET list/get/markdown/pdf, CORS/health inchangés

**Checkpoint Phase 4 :** `POST /revisions` synchrone utilise le nouveau moteur ; atomicité save OK.

---

## Phase 5 — Export PDF professionnel + non-régression

### Task 5.1 : Brancher export

**Files :**
- Modify : `src/revision/export.py`
- Adapt : `tests/test_revision_export.py`, `tests/test_revision_pdf_export.py`

- [ ] `export_pdf` utilise `ProfessionalPDFBuilder` depuis `content` Markdown stocké
- [ ] `export_markdown` inchangé fonctionnellement (metadata + body)
- [ ] Tests Content-Disposition / media types API inchangés

### Task 5.2 : Suite API + smoke

- [ ] `pytest tests/test_api_revisions.py tests/test_revision_store.py`
- [ ] Confirmer routes sans import torch

---

## Phase 6 — Acceptation qualité FR (PDF réel)

### Task 6.1 : Fixture cours FR

**Files :**
- Create : `tests/fixtures/courses/` (petit PDF extractible, notions connues)

### Task 6.2 : Test d’intégration pédagogique

**Files :**
- Create : `tests/test_revision_engine_quality_fr.py` (peut être `@pytest.mark.integration` / skip si pas de GPU)

- [ ] Sur mocks d’abord : structure Markdown, pas d’URL, exemples marqués, couverture
- [ ] Optionnel GPU Colab : run manuel documenté dans la spec/plan (pas obligatoire CI locale)

### Task 6.3 : Vérification manuelle Colab Tesla T4 (checklist)

- [ ] Générer **fiche courte** FR ; noter **durée réelle** (wall-clock) et résultat HTTP
- [ ] Générer **fiche complète** FR ; noter **durée réelle** (wall-clock) et résultat HTTP
- [ ] Comparer aux ordres de grandeur §8.2–8.3 (cible ~quelques minutes ; alerter si ≫ 4–5 min)
- [ ] Vérifier **timeouts** éventuels du tunnel / `proxyPort` / client React (`fetch`) ; documenter le délai observé dans un compte-rendu Phase 6
- [ ] Rouvrir fiche → re-télécharger PDF (reconstruction sans IR mémoire)
- [ ] Verrou : pendant génération, autre `POST` exclusif attend ; après exception, le lock est libre
- [ ] Forcer échec (mock/local) → aucune fiche en historique
- [ ] Non-régression endpoints existants (documents, conversations, health) après les runs

### Task 6.4 : Compte-rendu perf Phase 6

- [ ] Tableau : format × pages sélectionnées × nb appels LLM × durée × succès/timeout
- [ ] Décider si les plafonds §8.3 restent adaptés ou doivent être resserrés (doc only / patch ultérieur)

---

## Mapping erreurs HTTP (rappel exécution)

| Échec | Comportement |
|---|---|
| Avant save | Exception → middleware libère le lock → 400/404/500 selon type → **pas de save** |
| Après QC fail | Idem |
| Save OK | 201 + `RevisionPublic` |

Ne pas annoncer succès côté API si le pipeline n’a pas terminé.

---

## Dépendances entre phases

```text
Phase1 (models, guard, reader)
  → Phase2 (formatter, pdf builder)
  → Phase3 (understanding, planner)
  → Phase4 (generator, quality, facade, wire API)
  → Phase5 (export)
  → Phase6 (qualité FR)
```

---

## Critères de done V2

- [ ] Spec respectée (modules A–G + budgets §8 + fusion + QC similarité + Markdown round-trip)
- [ ] Contrats API/React/SQLite flat `content` intactes
- [ ] PDF régénérable depuis Markdown stocké uniquement
- [ ] Pas de `engine_payload` en V2
- [ ] Pas de jobs async
- [ ] Tests unitaires + API + robustesse (retries, budgets, no-save, lock)
- [ ] Mesures temps short/complete sur T4 + note timeouts tunnel
- [ ] Aucun commit sans autorisation

---

## Hors plan V2 (rappels)

- Colonne SQLite `engine_payload`
- Endpoints async / progress
- Refonte UI Markdown enrichi / spinner multi-étapes
- Optimisation qualité arabe
- Remplacement Qwen
