# Smart Revision V2 — Spécification technique

**Statut :** architecture A validée — amendements budgets / Markdown / fusion / QC / tests (rév. documentaire)
**Date :** 2026-11-25 (rév. budgets & robustesse)
**Périmètre :** moteur pédagogique Smart Revision (backend Python)
**Hors scope V2 :** jobs asynchrones, polling, progression UI, migration SQLite `engine_payload`, OCR, changement de LLM, refonte React, qualité arabe dédiée, commit Git

---

## 1. Objectif produit

Transformer Smart Revision d’un résumé monolothic proche du PDF en un **moteur de création de supports pédagogiques** qui :

1. Lit et comprend le cours PDF.
2. Identifie concepts, définitions, mécanismes, relations et dépendances.
3. Reconstruit un **plan pédagogique nouveau**.
4. Rédige un **cours original** (formulations propres, explications simplifiées, exemples pertinents).
5. Contrôle fidélité scientifique et qualité pédagogique.
6. Produit un **Markdown structuré** sauvegardé et un **PDF professionnel**.

**Règle fondamentale :** le PDF est une source de connaissances, pas un texte à copier ni à paraphraser mécaniquement.

**Style cible :** hybride « cours enseignant + support de révision moderne » (paragraphes courts, titres hiérarchisés, définitions encadrées, blocs « À retenir », exemples, tableaux, synthèses).

---

## 2. Décisions d’architecture figées

| Décision | Choix V2 |
|---|---|
| Intégration | Approche A — façade interne derrière `POST /revisions` **synchrone** |
| Jobs async | Non en V2 ; modules conçus pour migration ultérieure sans réécriture du moteur |
| Contrats API / React / SQLite flat `content` | Conservés |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` inchangé, instance unique partagée |
| Qualité prioritaire | Français |
| IR en mémoire | Oui pendant la génération |
| `engine_payload` SQLite | **Prévu pour plus tard, non implémenté en V2** |
| Reconstruction PDF | **Doit être possible depuis le seul Markdown structuré sauvegardé** |

---

## 3. Surface stable (ne pas casser)

### 3.1 API HTTP (`src/api/routes/revisions.py`)

- `POST /revisions` → 201 + `RevisionPublic`
- `GET /revisions`, `GET /revisions/{id}`
- `GET /revisions/{id}/markdown`, `GET /revisions/{id}/pdf`
- Corps `RevisionCreate` : `document_id`, `language`, `level`, `sheet_type`, `page_start`, `page_end`
- `RevisionPublic` : `id`, `document_id`, `filename`, `page_start`, `page_end`, `language`, `level`, `sheet_type`, **`content: str`**, `created_at`

### 3.2 Assistant

`EduRAGAssistant.create_revision` continue d’appeler une façade de génération puis `RevisionStore.save`.
List / get / export inchangés en signatures.

### 3.3 Frontend React

Formulaire (document, langue, niveau, short/complete, pages), historique, affichage `content`, téléchargements MD/PDF.
Pas de polling ni barre de progression en V2.

### 3.4 Store SQLite actuel

Table `revision_sheets` inchangée en V2 (pas de nouvelle colonne).
**Évolution documentée (post-V2) :** colonne nullable `engine_payload TEXT/JSON` pour IR — hors implémentation V2.

---

## 4. Package cible

```text
src/revision/
  engine/
    __init__.py
    models.py              # IR Pydantic
    reader.py              # CourseReader
    understanding.py       # CourseUnderstanding
    planner.py             # PedagogicalPlanner
    generator.py           # OriginalContentGenerator
    quality.py             # QualityController
    formatter.py           # ContentFormatter
    pdf_builder.py         # ProfessionalPDFBuilder
    llm_guard.py           # budgets, retries, plafonds d’appels
    facade.py              # RevisionEngine.run
  sheet.py                 # adaptateur mince → facade (ou dépréciation progressive)
  store.py                 # inchangé V2
  export.py                # MD + délégation PDF au builder
```

Le moteur **ne** dépend **pas** de FastAPI ni du store. Une future tâche async n’aurait qu’à appeler `RevisionEngine.run` puis `save`.

---

## 5. Modèles IR (Pydantic)

Responsabilité : séparer contenu pédagogique et présentation.

| Modèle | Champs essentiels |
|---|---|
| `CoursePage` | `page_number: int`, `text: str`, `filename: str` |
| `CourseUnit` | texte nettoyé, `page_start`, `page_end`, optionnellement titre détecté |
| `Concept` | `id`, `label`, `kind` (`definition` \| `formula` \| `method` \| `mechanism` \| `relation` \| `other`), `pages: list[int]`, `essential: bool`, `depends_on: list[str]`, `notes: str` |
| `CourseRelation` | `source_id`, `target_id`, `relation_type`, `pages` |
| `CourseModel` | `units`, `concepts`, `relations`, `themes`, `essential_concept_ids` |
| `LearningObjective` | `text`, `concept_ids` |
| `ChapterPlan` | `index`, `title`, `objectives`, `concept_ids`, `page_refs`, `needs_example`, `needs_table`, `needs_questions`, `depth` |
| `PedagogicalPlan` | `language`, `level`, `sheet_type`, `chapters: list[ChapterPlan]` |
| `ContentBlock` | discriminant `type` : `paragraph` \| `definition` \| `example` \| `retain` \| `table` \| `diagram` \| `synthesis` \| `question` ; champs associés ; `source_pages: list[int]` ; pour `example` : `origin` = `from_course` \| `pedagogical_addition` |
| `ChapterDraft` | `index`, `title`, `blocks`, `concept_ids_covered` |
| `PedagogicalDocument` | meta (filename, pages, language, level, sheet_type) + `chapters` |
| `QualityIssue` | `code`, `severity` (`error` \| `warning`), `chapter_index?`, `message` |
| `QualityReport` | `ok: bool`, `issues`, `rewrite_chapter_indexes` |

Validation stricte à chaque frontière LLM → objet Pydantic.

---

## 6. Modules détaillés

### 6.1 CourseReader

**Entrées :** library, `document_id`, intervalle de pages optionnel.
**Sorties :** `list[CoursePage]`, filename, bornes réelles.

**Responsabilités :**
- Résoudre le document (`list_documents` / chemin stocké).
- Extraire via `load_pdf`.
- Nettoyer (pages vides, espaces excessifs).
- Découper en **fenêtres** pour les étapes LLM (taille configurable, ex. N pages ou budget tokens estimé).

**Non-responsabilités :** compréhension sémantique, rédaction, PDF final.

### 6.2 CourseUnderstanding

**Objectif :** véritable modèle structuré du cours, pas un résumé textuel.

**Doit produire :**
- Thèmes / sous-thèmes
- Concepts fondamentaux
- Définitions, formules, méthodes, mécanismes
- Relations et dépendances entre notions
- Distinction essentiel / secondaire

**Processus :**
1. Analyser chaque fenêtre (appel LLM JSON borné).
2. **Fusionner** les résultats selon les règles strictes ci-dessous.
3. Valider `CourseModel` (Pydantic).

### Règles de fusion (obligatoires)

| Règle | Comportement |
|---|---|
| Similarité de labels seule | **Interdit** de fusionner deux concepts uniquement parce que leurs labels se ressemblent |
| Identité | Fusion seulement si même notion au sens du cours (définition / rôle / mécanisme alignés) **et** preuves textuelles dans les pages citées |
| Pages | Toujours **conserver et unionner** les `pages` sources de chaque occurrence |
| Notions proches | Conserver deux `Concept` distincts s’ils diffèrent (ex. « variable » vs « variable globale ») |
| Contradictions | Les détecter et les **marquer** dans le modèle (notes / issue) ; **ne jamais inventer** une résolution scientifique si le cours ne permet pas de trancher |
| Invention | Aucune formule, chiffre ou définition technique ajoutée lors de la fusion |

**Échec :** JSON invalide après retries → erreur pipeline (voir §9). Si le nombre de fenêtres nécessaires dépasse le plafond (§8) → erreur explicite « sélectionner moins de pages » (**pas de troncature silencieuse**).

### 6.3 PedagogicalPlanner

**Entrées :** `CourseModel`, `sheet_type`, `level`, `language`.
**Sortie :** `PedagogicalPlan`.

| Format | Comportement |
|---|---|
| `short` | Peu de chapitres ; notions essentielles ; explications concises ; exemples indispensables ; synthèse finale |
| `complete` | Couverture des essentiels ; développement des complexes ; relations ; exemples détaillés ; tableaux/schémas si utiles ; synthèses par chapitre ; questions de révision |

Le plan est une **nouvelle organisation pédagogique**, pas une copie de l’ordre brut du PDF (tout en respectant les dépendances conceptuelles).

1 appel LLM JSON + validation ; fallback heuristique **uniquement** si parse échoue et qu’un plan minimal fiable peut être dérivé du `CourseModel` ; sinon erreur.

### 6.4 OriginalContentGenerator

**Responsabilité centrale V2 :** rédiger un **nouveau cours**, pas une paraphrase.

Pour chaque `ChapterPlan` (série) :
- Contexte : plan global condensé, titres des chapitres déjà générés, extraits de pages liés aux `page_refs` / concepts.
- Produire `ChapterDraft` avec formulations originales, explications progressives, exemples.
- Marquer clairement les exemples ajoutés (`origin=pedagogical_addition`).
- **Interdit :** copier des paragraphes du PDF ; inventer formules, chiffres, définitions techniques absentes du cours ; URLs / bibliographies / sections génériques.

Budgets **par chapitre** (new tokens), pas un seul budget 2048 pour tout le document.

### 6.5 QualityController

**Contrôles (heuristiques d’abord) :**
- Couverture des `essential_concept_ids` du plan
- Fidélité structurelle : pas d’URL ; pas de sections interdites (biblio, contact, remerciements, annexes vides, glossaire hors cours)
- Qualité : chapitres vides, répétitions grossières, cohérence des titres vs plan
- Exemples `pedagogical_addition` : pertinents, marqués, non contradictoires avec le cours

### Similarité textuelle (anti-copie, anti faux positifs)

| Règle | Comportement |
|---|---|
| Cible | Détecter **passages copiés** ou **excessivement proches** du PDF (n-grammes / fenêtres longues) |
| Exclusions | Ne pas pénaliser seuls : termes techniques, noms de notions, formules, symboles de définitions nécessaires |
| Whitelist | Jargon et symboles normalisées du `CourseModel` / pages sources exclues du score de similarité |
| Usage | Signal **d’originalité rédactionnelle**, pas preuve unique de fidélité scientifique |
| Fidélité scientifique | Évaluée surtout via couverture des concepts, absence d’inventions (URLs, chiffres/formules non ancrés), cohérence QC — **pas** via similarité seule |

**Correction :** rewrite ciblé (plafond §8), pas de régénération totale systématique.

Si `QualityReport.ok` est faux après correction → **échec pipeline**, pas de sauvegarde.

### 6.6 ContentFormatter

**Entrée :** `PedagogicalDocument`
**Sorties :**
1. **Markdown structuré canonique** (celui sauvegardé dans `content`) — suffisamment riche et **déterministe** pour reconstruire le PDF plus tard **sans IR mémoire**.
2. (Optionnel interne) vue objet pour le builder si on parse le Markdown ; le contrat V2 impose que le **Markdown seul** suffise.

**Conventions Markdown V2 (contractuelles pour le PDF) — marqueurs exacts :**

| Bloc | Marqueur Markdown obligatoire |
|---|---|
| Titre document | `# …` |
| Chapitre | `## …` |
| Sous-section | `### …` |
| Paragraphe | texte hors quote / hors tableau |
| Définition | `> **Définition.** …` |
| À retenir | `> **À retenir.** …` |
| Exemple du cours | `> **Exemple (cours).** …` |
| Exemple ajouté | `> **Exemple (ajout pédagogique).** …` |
| Tableau | GFM pipe (`\| … \|` + ligne `\| --- \|`) |
| Schéma textuel | ligne ou bloc commençant par `**Schéma :**` |
| Synthèse (chapitre ou finale) | `## Synthèse` ou `### Synthèse` |
| Questions de révision | `## Questions de révision` puis liste `- ` / `1.` |

```markdown
# Titre du support
## Chapitre …
### Sous-section …
Paragraphe explicatif court.

> **Définition.** …
> **À retenir.** …
> **Exemple (cours).** …
> **Exemple (ajout pédagogique).** …

| Colonne A | Colonne B |
| --- | --- |
| … | … |

**Schéma :** description textuelle structurée…

### Synthèse
- …

## Questions de révision
1. …
```

**Tests round-trip obligatoires :** chaque type de bloc ci-dessus doit survivre à
`IR → Markdown → parse → structure logique → PDF`, **sans IR mémoire**.
Absence d’un marqueur = échec de test Formatter/PDF.

### 6.7 ProfessionalPDFBuilder

**Entrée V2 :** Markdown structuré sauvegardé (et/ou re-parsé).
**Sortie :** PDF bytes / fichier.

Doit inclure : couverture, présentation, TOC si pertinent, hiérarchie titres, typographie lisible, palette cohérente (ardoise/teal EduRAG), marges, encadrés, tableaux, pagination, en-têtes/pieds de page.

Remplace l’implémentation interne de `RevisionExport.export_pdf` ; conserve le nom d’entrypoint assistant/API.

Schémas : rendu textuel structuré en V2 (pas de dépendance Graphviz obligatoire).

---

## 7. Orchestration — `RevisionEngine.run`

```text
1. CourseReader.read(...)
2. CourseUnderstanding.analyze(...)     # multi-fenêtres + fusion
3. PedagogicalPlanner.plan(...)
4. pour chaque chapitre:
     OriginalContentGenerator.write_chapter(...)
5. QualityController.evaluate(+ rewrite ciblé)
6. ContentFormatter.to_document_markdown(...)
7. retourner dict plat compatible store.save
```

**PDF :** généré à l’export (`export_revision_pdf`) à partir du Markdown stocké, pas obligatoirement pendant `POST /revisions` (évite double travail ; garantit reconstruction post-génération).

Alternative acceptable : prévisualisation PDF à la création **uniquement** si le même builder lit le Markdown produit — jamais un IR volatile non persisté.

---

## 8. Robustesse Qwen — audit actuel & limites numériques V2

### 8.1 Audit configuration actuelle (code existant)

| Élément | Valeur constatée |
|---|---|
| Modèle | `Qwen/Qwen2.5-1.5B-Instruct` (`DEFAULT_MODEL_NAME`, surcharge `EDURAG_LLM_MODEL`) |
| Tokenizer | `AutoTokenizer.from_pretrained(self.model_name)` — même checkpoint |
| Chargement | `AutoModelForCausalLM`, `dtype=float16` sur CUDA, `weights_only=True`, `model.eval()` |
| Encodage prompt | `apply_chat_template(..., add_generation_prompt=True)` si `chat_template`, sinon `tokenizer(prompt)` |
| Génération | `model.generate(..., max_new_tokens=…, do_sample=False)` — déterministe |
| Défaut `max_new_tokens` LLM | `DEFAULT_MAX_NEW_TOKENS = 1024` |
| Plafond dur LLM | `MAX_NEW_TOKENS_LIMIT = 2048` |
| Budgets fiche V1 | `_SHEET_TOKEN_BUDGETS = {short: 512, complete: 2048}` — **un seul** appel |
| Contexte | `ensure_prompt_fits` : `prompt_tokens + max_new_tokens ≤ model.config.max_position_embeddings` (typ. **32768** pour Qwen2.5-1.5B) ; refuse sliding window / RoPE non standard ; **ne tronque jamais** — message « Sélectionnez moins de pages » |
| Verrou GPU | `ExclusiveAgentMiddleware` : `async with lock` sur `POST /revisions` (et documents / messages tutor) pendant **toute** la requête ASGI ; libéré en sortie normale ou exception |
| Exclusivité | Un seul travail agent à la fois ; list/get/export fiches non exclusifs |

### 8.2 Contraintes Tesla T4 (Colab)

- VRAM ~16 Go : 1.5B fp16 + KV cache confortable pour contextes **modérés** (quelques k tokens), coûteux si chaque appel pousse vers 8k–32k.
- Débit réaliste ~20–40 tokens/s selon longueur de prompt (ordre de grandeur 1.5B instruct).
- `POST` synchrone + tunnel/proxy Colab : viser quelques minutes au pire raisonnable ; au-delà, risque timeout client/proxy.
- Le verrou bloque aussi upload PDF et chat pendant toute la génération → plafonner les appels.

### 8.3 Limites numériques V2 (justifiées, non arbitraires)

Ces constantes seront centralisées dans `llm_guard` (noms indicatifs).

| Constante | Valeur | Justification |
|---|---|---|
| `MAX_COURSE_TOKENS_PER_WINDOW` | **1 800** | Laisse ~1–2k tokens de consignes + marge ; prompt total ≈ 2.5–3.5k + génération JSON courte → latence/VRAM raisonnables sur T4 |
| `MAX_ANALYSIS_WINDOWS` | **5** | 5 × (analyse) reste dans un budget temps ; au-delà → demander de réduire les pages (**pas de coupe silencieuse**) |
| `MAX_CHAPTERS_SHORT` | **4** | Fiche courte dense sans explosion d’appels |
| `MAX_CHAPTERS_COMPLETE` | **6** | Couverture pédagogique sans 10+ générations séquentielles |
| `TOKENS_UNDERSTANDING` | **384** | JSON structuré (concepts/relations) ; sous le défaut 1024 |
| `TOKENS_PLAN` | **512** | Plan JSON chapitres |
| `TOKENS_CHAPTER_SHORT` | **384** | Chapitre concis |
| `TOKENS_CHAPTER_COMPLETE` | **640** | Développement sans saturer 2048/appel |
| `TOKENS_REWRITE` | **384** | Correction ciblée |
| `MAX_RETRIES_PER_CALL` | **1** | 2 tentatives max (1 + 1 retry) — JSON fragile sur 1.5B, mais retries coûteux sous lock |
| `MAX_REWRITE_CHAPTERS` | **2** | Évite une boucle QC sur tout le document |
| `MAX_LLM_CALLS` | **16** | Pire cas : 5 understanding + 1 plan + 6 chapitres + 2 rewrites + marge retries ; retries compris dans ce plafond |
| `MAX_TOTAL_NEW_TOKENS` | **7 500** | **Maximum**, pas une consommation cible ; doit couvrir le pire cas *complete* (voir §8.3.1) ; ≈ 7.5k / ~25 tok/s ≈ **~5 min** génération pure au plafond |

**Plafond unique V1 (512/2048) :** abandonné comme budget document ; remplacé par la table ci-dessus. Aucun appel ne dépasse `MAX_NEW_TOKENS_LIMIT` (2048).

### 8.3.1 Cohérence short / complete vs plafond global

Budgets par étape **conservés**. Totaux théoriques de tokens *réservés* (avant retries supplémentaires) :

| Parcours | Calcul | Total |
|---|---|---|
| Short (max) | 5×384 + 512 + 4×384 + 2×384 | **4 736** |
| Complete (max) | 5×384 + 512 + 6×640 + 2×384 | **7 040** |

- `MAX_TOTAL_NEW_TOKENS = 7 500` ≥ 7 040 → le parcours complete maximal est faisable.
- Les retries consomment aussi ce compteur et `MAX_LLM_CALLS` ; un retry peut faire échouer le run avant d’atteindre tous les chapitres.
- **Les deux plafonds** (`MAX_LLM_CALLS` et `MAX_TOTAL_NEW_TOKENS`) sont vérifiés **simultanément** à chaque réservation d’appel dans `llm_guard`.
- Si l’un des deux est atteint ou dépassé : **arrêt propre** du pipeline, exception, **aucune** sauvegarde de fiche partielle.
- 7 500 = plafond de sécurité Colab/T4, **pas** un objectif à consommer.

### 8.4 Dépassement de volume — comportement obligatoire

Si, après lecture :

- le texte sélectionné nécessiterait **plus de `MAX_ANALYSIS_WINDOWS`**, **ou**
- un appel ne passe pas `ensure_prompt_fits`, **ou**
- le plan exigerait plus de chapitres que le plafond format,

alors lever une erreur explicite du type :

> « Le cours sélectionné est trop volumineux. Sélectionnez moins de pages. »

**Interdit :** tronquer silencieusement pages, concepts ou chapitres.

### 8.5 Mécanismes `llm_guard`

| Mécanisme | Règle V2 |
|---|---|
| Validation | Sortie structurée → JSON → Pydantic ; échec = retry jusqu’à `MAX_RETRIES_PER_CALL` |
| Double plafond | Avant chaque appel : vérifier **à la fois** `calls < MAX_LLM_CALLS` et `reserved_new_tokens + budget_étape ≤ MAX_TOTAL_NEW_TOKENS` |
| Arrêt | Dépassement d’un plafond → exception claire ; pas de chapitre partiel persisté ; pas de `save` |
| Compteurs | Incrémenter appels (y compris retries) et tokens réservés/consommés |
| Contexte | `ensure_prompt_fits` **par appel** |
| Sampling | `do_sample=False` |
| Couverture | QC post-génération ; rewrite ≤ `MAX_REWRITE_CHAPTERS` puis échec si insuffisant |

---

## 9. Erreurs HTTP, verrou GPU, absence de sauvegarde (précision §5)

### 9.1 Verrou GPU

- `POST /revisions` reste sous `ExclusiveAgentMiddleware`.
- Le verrou est détenu pendant **toute** la requête (pipeline + `save`).
- Libération : automatique à la fin de la requête ASGI (succès **ou** exception) — **aucune** rétention manuelle hors middleware existant.
- `GET` list/get/exports : non exclusifs (inchangé).

### 9.2 Atomicité sauvegarde

- **Aucun** `RevisionStore.save` tant que le pipeline n’a pas produit un Markdown QC-validé.
- Échec à n’importe quelle étape → **zéro** nouvelle ligne dans `revision_sheets`.
- Succès API 201 **uniquement** après `save` réussi.
- Ne jamais renvoyer un 201 avec contenu partiel ou placeholder.

### 9.3 Mapping d’erreurs (contrat)

| Situation | HTTP | `detail` (orientation) |
|---|---|---|
| Document absent | 404 | `Document introuvable dans la bibliothèque.` (existant) |
| Paramètres invalides (langue, niveau, type, pages) | 400 | message métier actuel / étendu |
| Cours trop volumineux / fenêtre impossible | 400 | message explicite (sélectionner moins de pages) |
| Échec parse/validation LLM après retries | 500 | génération échouée (message stable, sans traceback) |
| QC non satisfait | 500 ou 400 | préférer 500 « La génération de la fiche a échoué. » pour alignement route actuelle `RuntimeError` → 500 ; documenter si 400 métier plus clair |
| Exception LLM / RuntimeError | 500 | `str(error)` ou message stabilisé |
| Fiche absente (get/export) | 404 | `Fiche introuvable.` |

Toute exception remonte après libération naturelle du lock ; pas de double save.

---

## 10. Formats short / complete

Même pipeline ; profondeur différente via Planner + Generator + budgets/chapitre.

- Pas de nombre de pages imposé.
- Pas de sections vides pour « faire du volume ».

---

## 11. Enrichissement contrôlé

Autorisé : exemples pédagogiques, analogies, reformulations explicatives, tableaux comparatifs utiles.
Interdit : présenter un ajout comme extrait du PDF ; inventer formules / données / définitions techniques ; sources externes inventées.

Signalement : blocs Markdown `Exemple (ajout pédagogique)`.

---

## 12. Critères d’acceptation

1. Analyse structurée du cours (concepts + relations + essentiels).
2. Plan pédagogique distinct du plan brut PDF.
3. Rédaction originale (pas copie/paraphrase mécanique).
4. Explications simplifiées selon le niveau.
5. Fidélité scientifique aux concepts du PDF.
6. Absence de sections génériques artificielles.
7. Exemples pertinents et marqués si ajoutés.
8. Cohérence inter-chapitres.
9. PDF professionnel reconstruit depuis Markdown stocké.
10. Short et complete opérationnels.
11. API + React inchangés fonctionnellement.
12. Sauvegarde uniquement en succès ; exports MD/PDF OK.
13. Test réel FR sur un PDF de cours.

---

## 13. Risques résiduels

- Latence synchrone élevée sur T4 (N appels sous lock).
- Timeouts proxy Colab / clients HTTP.
- Qwen 1.5B : JSON parfois fragile → retries + validation.
- Qualité PDF limitée pour « schémas » sans moteur graphique.
- Similarité textuelle : faux positifs sur jargon — whitelist termes techniques.

---

## 14. Migration future (hors V2)

1. Persister `engine_payload` (IR JSON) en colonne additive.
2. Envelopper `RevisionEngine.run` dans une tâche async + endpoints status.
3. Progression UI optionnelle.
Le cœur pédagogique (modules A–G) ne doit pas être réécrit pour cela.
