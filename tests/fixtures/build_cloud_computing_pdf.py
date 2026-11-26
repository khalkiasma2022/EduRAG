"""Genere le PDF pedagogique de test Cloud_Computing_EduRAG_Test.pdf.

Usage :
    python tests/fixtures/build_cloud_computing_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUTPUT = Path(__file__).with_name("Cloud_Computing_EduRAG_Test.pdf")


class CoursePDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "EduRAG - Cours de test : Cloud Computing", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def _write(pdf: CoursePDF, text: str) -> None:
    pdf.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _h1(pdf: CoursePDF, text: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)


def _h2(pdf: CoursePDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def _h3(pdf: CoursePDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _p(pdf: CoursePDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 11)
    _write(pdf, text)
    pdf.ln(2)


def _definition(pdf: CoursePDF, term: str, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    _write(pdf, f"Definition - {term}")
    pdf.set_font("Helvetica", "", 11)
    _write(pdf, text)
    pdf.ln(2)


def _example(pdf: CoursePDF, text: str) -> None:
    pdf.set_font("Helvetica", "I", 11)
    _write(pdf, f"Exemple : {text}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)


def _retain(pdf: CoursePDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    _write(pdf, f"A retenir : {text}")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)


def _table(pdf: CoursePDF, headers: list[str], rows: list[list[str]]) -> None:
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = usable / len(headers)
    pdf.set_font("Helvetica", "B", 10)
    for header in headers:
        pdf.cell(col_w, 7, header[:28], border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for row in rows:
        for cell in row:
            pdf.cell(col_w, 7, cell[:28], border=1)
        pdf.ln()
    pdf.ln(3)


def build() -> Path:
    pdf = CoursePDF(format="A4", unit="mm")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(22, 22, 22)

    # --- Page de titre ---
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 22)
    pdf.multi_cell(0, 10, "Cloud Computing", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 14)
    pdf.multi_cell(0, 8, "Cours pedagogique de test pour EduRAG", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Document textuel extractible destine a valider la generation de fiches "
        "de revision. Il couvre les concepts fondamentaux du cloud, les modeles "
        "de service et de deploiement, la virtualisation, la securite et des cas pratiques.",
        align="C",
    )
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 5, "Fichier : Cloud_Computing_EduRAG_Test.pdf", align="C")

    # --- Table des matieres ---
    pdf.add_page()
    _h1(pdf, "Table des matieres")
    toc = [
        "1. Introduction au Cloud Computing",
        "2. Historique et naissance du cloud",
        "3. Concepts fondamentaux",
        "4. Architecture cloud",
        "5. IaaS - Infrastructure as a Service",
        "6. PaaS - Platform as a Service",
        "7. SaaS - Software as a Service",
        "8. Modeles de deploiement",
        "9. Virtualisation et conteneurisation",
        "10. Stockage et reseau",
        "11. Securite dans le cloud",
        "12. Avantages et limites",
        "13. Comparaisons utiles",
        "14. Cas pratiques",
        "15. Conclusion et glossaire",
    ]
    for line in toc:
        _p(pdf, line)

    # 1
    pdf.add_page()
    _h1(pdf, "1. Introduction au Cloud Computing")
    _p(
        pdf,
        "Le cloud computing designe un mode de fourniture de ressources informatiques "
        "a la demande via un reseau, le plus souvent Internet. L'utilisateur consomme "
        "du calcul, du stockage ou des services applicatifs sans gerer directement les "
        "machines physiques sous-jacentes.",
    )
    _definition(
        pdf,
        "Cloud computing",
        "Modele qui permet un acces ubiquitaire, pratique et a la demande a un ensemble "
        "partage de ressources configurables (reseaux, serveurs, stockage, applications).",
    )
    _p(
        pdf,
        "Trois idees structurent le sujet : l'elasticite (ajuster la capacite), "
        "la mutualisation (partager l'infrastructure) et le self-service (provisionner "
        "sans intervention manuelle systematique).",
    )
    _example(
        pdf,
        "Une startup lance une API web. Au lieu d'acheter un serveur, elle reserve "
        "des instances virtuelles et augmente leur nombre lors d'un pic de trafic.",
    )
    _retain(
        pdf,
        "Le cloud transforme l'informatique en service consomme, mesure et facture.",
    )
    _h2(pdf, "1.1 Pourquoi etudier le cloud ?")
    _p(
        pdf,
        "Les organisations migrent pour reduire le delai de mise sur le marche, pour "
        "absorber des pics saisonniers et pour acceder a des services specializes "
        "(bases gerees, analytique, diffusion de contenu) sans construire chaque brique.",
    )
    _p(
        pdf,
        "Pour l'etudiant, le cloud est aussi un vocabulaire : region, zone, instance, "
        "image, volume, bucket, politique IAM, equilibrage de charge. Ces termes "
        "reviendront dans tous les chapitres suivants.",
    )
    _definition(
        pdf,
        "Self-service",
        "Capacite pour un utilisateur autorise de provisionner des ressources via un "
        "portail ou une API, sans ticket manuel a chaque demande.",
    )
    _example(
        pdf,
        "Un enseignant cree un environnement de TP pour 40 etudiants en quelques "
        "minutes, puis le detruit a la fin du semestre.",
    )

    # 2
    pdf.add_page()
    _h1(pdf, "2. Historique et naissance du cloud")
    _p(
        pdf,
        "Les racines du cloud remontent au temps partage des mainframes, puis a la "
        "virtualisation des annees 2000 et a la generalisation de la bande passante.",
    )
    _h2(pdf, "2.1 Etapes marquantes")
    _p(
        pdf,
        "Les fournisseurs ont d'abord propose du stockage et du calcul a la demande, "
        "puis des plateformes de developpement et enfin des suites logicielles completes. "
        "La standardisation des API a accelere l'adoption.",
    )
    _definition(
        pdf,
        "Utility computing",
        "Idee selon laquelle les ressources informatiques se consomment comme une "
        "utilite (eau, electricite), avec une facturation a l'usage.",
    )
    _example(
        pdf,
        "Les offres d'instances a l'heure ont popularise le paiement a l'usage pour "
        "les equipes de developpement et les laboratoires universitaires.",
    )
    _retain(
        pdf,
        "Le cloud nait de la convergence entre virtualisation, reseaux rapides et "
        "modeles economiques a l'usage.",
    )

    # 3
    pdf.add_page()
    _h1(pdf, "3. Concepts fondamentaux")
    _h2(pdf, "3.1 Elasticite et scalabilite")
    _definition(
        pdf,
        "Elasticite",
        "Capacite d'augmenter ou de diminuer rapidement les ressources allouees en "
        "fonction de la charge.",
    )
    _definition(
        pdf,
        "Scalabilite",
        "Aptitude d'un systeme a absorber une hausse de charge en ajoutant des "
        "ressources (horizontale) ou en renforcant une ressource (verticale).",
    )
    _h2(pdf, "3.2 Multitenancy et isolation")
    _definition(
        pdf,
        "Multitenancy",
        "Architecture ou plusieurs clients (tenants) partagent la meme infrastructure "
        "tout en restant logiquement isoles.",
    )
    _p(
        pdf,
        "L'isolation repose sur la virtualisation, les espaces de noms, les politiques "
        "d'acces et le cloisonnement reseau. Une mauvaise isolation expose des risques "
        "de fuite de donnees entre locataires.",
    )
    _example(
        pdf,
        "Deux entreprises utilisent le meme hyperviseur, mais des machines virtuelles "
        "et des comptes distincts avec des cles d'API separees.",
    )
    _h2(pdf, "3.3 Mesure et facturation")
    _definition(
        pdf,
        "Metering",
        "Mesure precise de la consommation (CPU-heure, Go stockes, requetes API) pour "
        "piloter la facturation et l'optimisation des couts.",
    )
    _retain(
        pdf,
        "Elasticite, multitenancy et metering forment le socle conceptuel du cloud.",
    )

    # 4
    pdf.add_page()
    _h1(pdf, "4. Architecture cloud")
    _p(
        pdf,
        "Une architecture cloud combine des couches physiques, de virtualisation, "
        "d'orchestration et de services exposes via des API.",
    )
    _h2(pdf, "4.1 Couches principales")
    _table(
        pdf,
        ["Couche", "Role"],
        [
            ["Physique", "Serveurs, disques, commutateurs"],
            ["Virtualisation", "Hyperviseurs, reseaux virtuels"],
            ["Controle", "Orchestrateur, ordonnanceur, API"],
            ["Services", "IaaS, PaaS, SaaS proposes aux clients"],
        ],
    )
    _definition(
        pdf,
        "Orchestrateur",
        "Composant qui place, demarre, surveille et repare les charges de travail "
        "selon des politiques (capacite, affinite, cout).",
    )
    _definition(
        pdf,
        "Region et zone de disponibilite",
        "Une region regroupe plusieurs zones isolees. Repartir une application sur "
        "plusieurs zones ameliore la resilience.",
    )
    _example(
        pdf,
        "Une API critique s'execute dans deux zones d'une meme region pour survivre "
        "a la panne d'un centre de donnees.",
    )
    _retain(
        pdf,
        "L'architecture cloud se lit comme une pile : physique, virtualisation, "
        "controle, services.",
    )

    # 5
    pdf.add_page()
    _h1(pdf, "5. IaaS - Infrastructure as a Service")
    _definition(
        pdf,
        "IaaS",
        "Modele ou le fournisseur livre des ressources d'infrastructure (machines "
        "virtuelles, reseaux, volumes) et le client gere le systeme d'exploitation "
        "et les applications.",
    )
    _p(
        pdf,
        "L'IaaS offre un controle fin : choix de l'OS, installation de middlewares, "
        "regles reseau. En contrepartie, la responsabilite de correctifs et de "
        "supervision reste largement cote client.",
    )
    _h2(pdf, "5.1 Blocs typiques")
    _p(
        pdf,
        "Instances de calcul, images systeme, groupes de securite, equilibrage de "
        "charge et adresses publiques constituent le vocabulaire courant de l'IaaS.",
    )
    _definition(
        pdf,
        "Machine virtuelle (VM)",
        "Ordinateur logiciel isole qui execute un systeme d'exploitation au-dessus "
        "d'un hyperviseur partage.",
    )
    _example(
        pdf,
        "Un laboratoire cree trois VM Ubuntu pour un cluster Hadoop experimental, "
        "puis les detruit apres l'experience pour arreter la facturation.",
    )
    _retain(
        pdf,
        "En IaaS, vous louez l'infrastructure ; vous administrez encore beaucoup.",
    )

    # 6
    pdf.add_page()
    _h1(pdf, "6. PaaS - Platform as a Service")
    _definition(
        pdf,
        "PaaS",
        "Modele ou le fournisseur livre une plateforme prete a deployer du code "
        "(runtime, bases gerees, outils CI) sans gerer les VM sous-jacentes.",
    )
    _p(
        pdf,
        "Le PaaS accelere le developpement : build, deploiement, scaling et journaux "
        "sont industrialises. La contrepartie est une perte de flexibilite sur "
        "l'environnement d'execution.",
    )
    _definition(
        pdf,
        "Runtime gere",
        "Environnement d'execution (par exemple Node.js ou Python) maintenu par le "
        "fournisseur, avec mises a jour et scaling automatiques.",
    )
    _example(
        pdf,
        "Une equipe pousse une application web vers un service PaaS qui compile, "
        "deploie et ajoute des instances lors des pics du weekend.",
    )
    _retain(
        pdf,
        "Le PaaS echange du controle contre de la vitesse de livraison.",
    )

    # 7
    pdf.add_page()
    _h1(pdf, "7. SaaS - Software as a Service")
    _definition(
        pdf,
        "SaaS",
        "Modele ou l'application complete est fournie via le navigateur ou une API, "
        "avec abonnement. Le client configure l'usage, pas l'infrastructure.",
    )
    _p(
        pdf,
        "Messagerie, bureautique collaborative, CRM et outils de tickets illustrent "
        "le SaaS. La gouvernance porte surtout sur les identites, les donnees et les "
        "integrations.",
    )
    _definition(
        pdf,
        "Abonnement SaaS",
        "Mode de facturation recurrent (par utilisateur, par organisation ou par "
        "volume de transactions) pour un logiciel heberge.",
    )
    _example(
        pdf,
        "Un etablissement scolaire utilise une suite collaborative en SaaS pour les "
        "cours a distance, sans installer de serveur de messagerie local.",
    )
    _table(
        pdf,
        ["Modele", "Ce que gere le client"],
        [
            ["IaaS", "OS, middleware, applications, donnees"],
            ["PaaS", "Applications, donnees, configuration"],
            ["SaaS", "Donnees, utilisateurs, parametrage"],
        ],
    )
    _retain(
        pdf,
        "Du IaaS au SaaS, la responsabilite operationnelle se deplace vers le fournisseur.",
    )

    # 8
    pdf.add_page()
    _h1(pdf, "8. Modeles de deploiement")
    _definition(
        pdf,
        "Cloud public",
        "Infrastructure mutualisee accessible via Internet, operee par un fournisseur "
        "externe.",
    )
    _definition(
        pdf,
        "Cloud prive",
        "Infrastructure dediee a une organisation, hebergee en interne ou chez un "
        "operateur exclusif.",
    )
    _definition(
        pdf,
        "Cloud hybride",
        "Combinaison de cloud public et prive avec portabilite des charges et des "
        "donnees entre les deux.",
    )
    _definition(
        pdf,
        "Cloud communautaire",
        "Infrastructure partagee par plusieurs organisations ayant des besoins "
        "communs (recherche, administration).",
    )
    _example(
        pdf,
        "Une banque conserve les donnees sensibles en cloud prive et utilise le "
        "cloud public pour des analyses non critiques.",
    )
    _retain(
        pdf,
        "Le choix public / prive / hybride depend de la sensibilite des donnees et "
        "des contraintes de conformite.",
    )

    # 9
    pdf.add_page()
    _h1(pdf, "9. Virtualisation et conteneurisation")
    _h2(pdf, "9.1 Virtualisation")
    _definition(
        pdf,
        "Hyperviseur",
        "Logiciel qui abstrait le materiel et heberge plusieurs machines virtuelles "
        "sur un meme hote physique.",
    )
    _p(
        pdf,
        "La virtualisation isole les systemes d'exploitation et facilite la densite "
        "d'hebergement. Elle reste plus lourde qu'un conteneur lorsque l'on veut "
        "deployer rapidement de nombreuses applications.",
    )
    _h2(pdf, "9.2 Conteneurs")
    _definition(
        pdf,
        "Conteneur",
        "Unite legere qui empaquette une application et ses dependances en partageant "
        "le noyau de l'hote.",
    )
    _definition(
        pdf,
        "Orchestration de conteneurs",
        "Automatisation du placement, du scaling et de la reparation des conteneurs "
        "(exemple conceptuel : un cluster d'orchestrateur).",
    )
    _example(
        pdf,
        "Une API est empaquetee en image conteneur, deployee en trois replicas, puis "
        "mise a jour par remplacement progressif des instances.",
    )
    _retain(
        pdf,
        "VM = isolation forte et OS complets ; conteneurs = densite et deploiements rapides.",
    )

    # 10
    pdf.add_page()
    _h1(pdf, "10. Stockage et reseau")
    _h2(pdf, "10.1 Types de stockage")
    _definition(
        pdf,
        "Stockage objet",
        "Stockage de blobs identifies par cle, adapte aux fichiers media, sauvegardes "
        "et jeux de donnees peu structures.",
    )
    _definition(
        pdf,
        "Stockage bloc",
        "Volumes disque attaches a des instances, adaptes aux bases de donnees et "
        "systemes de fichiers.",
    )
    _definition(
        pdf,
        "Stockage fichier",
        "Partages accessibles via des protocoles de fichiers, utiles pour des postes "
        "ou applications qui attendent un systeme de fichiers reseau.",
    )
    _h2(pdf, "10.2 Reseau cloud")
    _definition(
        pdf,
        "VPC (Virtual Private Cloud)",
        "Reseau logique isole dans le cloud, avec sous-reseaux, tables de routage et "
        "passerelles.",
    )
    _definition(
        pdf,
        "Equilibreur de charge",
        "Composant qui repartit le trafic entre plusieurs instances saines pour la "
        "disponibilite et la montee en charge.",
    )
    _example(
        pdf,
        "Des images de cours sont deposees en stockage objet, tandis que la base "
        "relationnelle utilise un volume bloc chiffre.",
    )
    _retain(
        pdf,
        "Choisir le stockage selon la structure des donnees ; isoler le reseau via un VPC.",
    )

    # 11
    pdf.add_page()
    _h1(pdf, "11. Securite dans le cloud")
    _definition(
        pdf,
        "Modele de responsabilite partagee",
        "Repartition des devoirs de securite entre fournisseur (infrastructure) et "
        "client (configuration, donnees, identites), variable selon IaaS/PaaS/SaaS.",
    )
    _definition(
        pdf,
        "IAM (Identity and Access Management)",
        "Ensemble de mecanismes pour authentifier les utilisateurs et applications, "
        "puis leur attribuer le moindre privilege necessaire.",
    )
    _definition(
        pdf,
        "Chiffrement",
        "Protection des donnees en transit et au repos par algorithmes cryptographiques "
        "et gestion des cles.",
    )
    _p(
        pdf,
        "Les erreurs de configuration (seaux publics, cles exposees, droits trop larges) "
        "restent une cause frequente d'incidents. La journalisation et la detection "
        "d'anomalies complementent les controles preventifs.",
    )
    _example(
        pdf,
        "Une politique IAM interdit l'acces anonyme au stockage objet et impose une "
        "authentification a facteurs multiples pour les administrateurs.",
    )
    _retain(
        pdf,
        "Dans le cloud, la securite est un partage de roles : identifier qui fait quoi.",
    )

    # 12
    pdf.add_page()
    _h1(pdf, "12. Avantages et limites")
    _h2(pdf, "12.1 Avantages")
    _p(
        pdf,
        "Reduction du delai de mise a disposition, paiement a l'usage, elasticite, "
        "services manages et couverture geographique sont des atouts majeurs.",
    )
    _h2(pdf, "12.2 Limites")
    _definition(
        pdf,
        "Vendor lock-in",
        "Dependance forte a un fournisseur rendue couteuse par des services proprietaires "
        "ou des formats difficiles a migrer.",
    )
    _p(
        pdf,
        "La latence reseau, la complexite des factures, les contraintes de souverainete "
        "des donnees et la necessite de competences cloud restent des freins.",
    )
    _example(
        pdf,
        "Une application fortement couplee a un service de files d'attente proprietaire "
        "demande un effort important pour changer de fournisseur.",
    )
    _retain(
        pdf,
        "Le cloud apporte de la vitesse, mais impose gouvernance, couts et portabilite.",
    )

    # 13
    pdf.add_page()
    _h1(pdf, "13. Comparaisons utiles")
    _h2(pdf, "13.1 CapEx versus OpEx")
    _definition(
        pdf,
        "CapEx",
        "Depenses d'investissement (achat de serveurs) amorties sur plusieurs annees.",
    )
    _definition(
        pdf,
        "OpEx",
        "Depenses d'exploitation recurrentes, typiques du paiement a l'usage cloud.",
    )
    _h2(pdf, "13.2 Cloud versus datacenter classique")
    _table(
        pdf,
        ["Critere", "Datacenter", "Cloud"],
        [
            ["Provisionnement", "Semaines", "Minutes"],
            ["Capacite", "Planifiee", "Elastique"],
            ["Couts", "CapEx dominant", "OpEx dominant"],
            ["Responsabilite", "Interne", "Partagee"],
        ],
    )
    _example(
        pdf,
        "Un projet de trois mois prefere l'OpEx cloud pour eviter l'achat de materiel "
        "qui resterait inutilise ensuite.",
    )
    _retain(
        pdf,
        "Comparer le cloud au datacenter sur le temps, le cout et la responsabilite.",
    )

    # 14
    pdf.add_page()
    _h1(pdf, "14. Cas pratiques")
    _h2(pdf, "14.1 Site web elastique")
    _p(
        pdf,
        "Un site d'inscription a un evenement utilise un equilibrage de charge, des "
        "instances auto-scale et un stockage objet pour les pieces jointes.",
    )
    _h2(pdf, "14.2 Pipeline de donnees")
    _p(
        pdf,
        "Des journaux applicatifs sont collectes, stockes en objet, puis analyses par "
        "un service de traitement par lots. Le metering permet de suivre le cout par "
        "journee de logs.",
    )
    _h2(pdf, "14.3 Application educative")
    _example(
        pdf,
        "EduRAG heberge des PDF de cours, indexe leur contenu et genere des fiches. "
        "Le stockage objet conserve les documents ; le calcul traite les requetes "
        "d'etudiants pendant les pics de revision.",
    )
    _definition(
        pdf,
        "Auto-scaling",
        "Politique qui ajoute ou retire automatiquement des instances selon des "
        "indicateurs (CPU, files d'attente, latence).",
    )
    _retain(
        pdf,
        "Les cas pratiques relient elasticite, stockage et securite a un besoin metier.",
    )

    # 15
    pdf.add_page()
    _h1(pdf, "15. Conclusion et glossaire")
    _p(
        pdf,
        "Le cloud computing articule modeles de service (IaaS, PaaS, SaaS), modeles "
        "de deploiement, virtualisation, reseaux, stockage et securite partagee. Une "
        "fiche de revision utile doit relier ces concepts sans inventer de faits hors "
        "cours et sans transformer le document en questionnaire.",
    )
    _h2(pdf, "Glossaire rapide")
    glossary = [
        ("API", "Interface programmee pour piloter des services cloud."),
        ("Availability", "Proportion de temps ou un service reste accessible."),
        ("Bucket", "Conteneur logique de stockage objet."),
        ("CDN", "Reseau de diffusion de contenu proche des utilisateurs."),
        ("Failover", "Basculement automatique vers une ressource de secours."),
        ("Latency", "Delai de reponse d'un service ou d'un reseau."),
        ("Snapshot", "Image ponctuelle d'un volume pour sauvegarde ou clonage."),
        ("SLA", "Engagement contractuel de niveau de service."),
        ("Tenant", "Locataire logique dans une architecture multitenant."),
        ("Workload", "Charge de travail applicative a placer sur l'infrastructure."),
    ]
    for term, meaning in glossary:
        _definition(pdf, term, meaning)
    _retain(
        pdf,
        "Relire les definitions IaaS/PaaS/SaaS, VPC, IAM et elasticite avant un examen.",
    )

    # Pages supplementaires de synthese pedagogique (sans questions).
    pdf.add_page()
    _h1(pdf, "Synthese transversale A - Du besoin au service")
    _p(
        pdf,
        "Partir d'un besoin metier (site web, pipeline de logs, plateforme educative) "
        "puis choisir un modele de service. Si l'equipe veut controler l'OS, IaaS est "
        "naturel. Si elle veut deployer du code rapidement, PaaS convient. Si elle "
        "veut un outil pret a l'emploi, SaaS suffit.",
    )
    _p(
        pdf,
        "Ensuite choisir le deploiement : public pour la flexibilite, prive pour le "
        "controle, hybride pour combiner les deux. Enfin dimensionner stockage et "
        "reseau, puis appliquer IAM et chiffrement.",
    )
    _definition(
        pdf,
        "Portabilite",
        "Faculte de deplacer une charge de travail entre environnements avec un effort "
        "raisonnable, souvent aidee par les conteneurs et des API standard.",
    )
    _example(
        pdf,
        "Empaqueter une API en conteneur facilite le passage d'un cloud public a un "
        "cluster interne, a condition d'eviter les services trop proprietaires.",
    )
    _retain(
        pdf,
        "Besoin metier -> modele de service -> deploiement -> stockage/reseau -> securite.",
    )

    pdf.add_page()
    _h1(pdf, "Synthese transversale B - Dependances conceptuelles")
    _p(
        pdf,
        "L'elasticite depend du metering et de l'auto-scaling. L'auto-scaling s'appuie "
        "sur l'equilibreur de charge et des instances saines. Les instances vivent dans "
        "un VPC. Le VPC s'appuie sur des sous-reseaux et des regles de securite. Les "
        "donnees persistent via stockage bloc, fichier ou objet.",
    )
    _p(
        pdf,
        "La securite traverse toutes ces couches via le modele de responsabilite "
        "partagee : le fournisseur securise l'infrastructure de base ; le client "
        "securise comptes, configurations et donnees selon IaaS, PaaS ou SaaS.",
    )
    _definition(
        pdf,
        "Observabilite",
        "Capacite a comprendre l'etat d'un systeme via metriques, journaux et traces, "
        "indispensable pour piloter l'auto-scaling et detecter les incidents.",
    )
    _definition(
        pdf,
        "Haute disponibilite",
        "Conception qui limite les interruptions en eliminant les points uniques de "
        "panne, souvent via plusieurs zones et un failover automatise.",
    )
    _example(
        pdf,
        "Une base primaire dans la zone A et une secondaire dans la zone B illustrent "
        "la haute disponibilite regionale.",
    )
    _retain(
        pdf,
        "Les concepts cloud forment un graphe de dependances, pas une liste isolee.",
    )

    pdf.add_page()
    _h1(pdf, "Synthese transversale C - Pieges frequents")
    _p(
        pdf,
        "Sous-estimer les couts de sortie de donnees, ouvrir un stockage objet au public "
        "par erreur, donner des droits administrateur trop larges, et coupler le code a "
        "des services non portables sont des erreurs classiques.",
    )
    _p(
        pdf,
        "Une bonne fiche de revision doit expliquer ces pieges avec le vocabulaire du "
        "cours : vendor lock-in, IAM, bucket, VPC, metering, OpEx. Elle doit conserver "
        "les relations entre concepts et proposer des exemples marques comme tels.",
    )
    _definition(
        pdf,
        "Cout de sortie (egress)",
        "Frais lies au transfert de donnees hors du cloud du fournisseur, souvent "
        "oublies dans les estimations initiales.",
    )
    _definition(
        pdf,
        "Configuration drift",
        "Ecart progressif entre l'infrastructure reelle et la configuration desiree, "
        "qui augmente le risque d'incidents.",
    )
    _example(
        pdf,
        "Une regle de pare-feu ajoutee manuellement en production et jamais reconduite "
        "dans le modele d'infrastructure provoque un drift difficile a auditer.",
    )
    _retain(
        pdf,
        "Gouverner identites, configurations et couts autant que la capacite technique.",
    )

    pdf.add_page()
    _h1(pdf, "Complement - DevOps et cloud")
    _definition(
        pdf,
        "Infrastructure as Code",
        "Description declarative de l'infrastructure versionnee, permettant de recreer "
        "des environnements de facon reproductible.",
    )
    _definition(
        pdf,
        "CI/CD",
        "Integration et deploiement continus : construire, tester et publier "
        "automatiquement des versions applicatives.",
    )
    _p(
        pdf,
        "Le cloud amplifie DevOps : des environnements ephemeres, des pipelines qui "
        "provisionnent des ressources le temps d'un test, puis les detruisent. La "
        "discipline des etiquettes (tags) et des budgets complete la technique.",
    )
    _example(
        pdf,
        "Un pipeline cree un environnement de preproduction, execute les tests "
        "d'acceptation, publie l'artefact, puis supprime l'environnement temporaire.",
    )
    _retain(
        pdf,
        "Sans automatisation, l'elasticite cloud reste sous-exploitee.",
    )

    pdf.add_page()
    _h1(pdf, "Complement - Gouvernance et conformite")
    _definition(
        pdf,
        "Gouvernance cloud",
        "Ensemble de politiques, roles et controles qui cadrent qui peut creer quelles "
        "ressources, ou et a quel cout.",
    )
    _definition(
        pdf,
        "Residance des donnees",
        "Exigence de conserver certaines donnees dans une region ou un pays donne pour "
        "des raisons legales ou contractuelles.",
    )
    _p(
        pdf,
        "La conformite s'appuie sur des journaux d'audit, le chiffrement, la "
        "classification des donnees et des revues periodiques des droits IAM. Le choix "
        "d'une region n'est pas seulement technique : il engage la souverainete.",
    )
    _example(
        pdf,
        "Une administration publique impose que les dossiers citoyens restent dans une "
        "region nationale, meme si une autre region est moins chere.",
    )
    _retain(
        pdf,
        "Technique, cout et droit doivent etre alignes dans tout projet cloud.",
    )

    pdf.add_page()
    _h1(pdf, "Annexe - Carte mentale textuelle")
    _p(
        pdf,
        "Cloud computing\n"
        "- Modeles de service : IaaS, PaaS, SaaS\n"
        "- Modeles de deploiement : public, prive, hybride, communautaire\n"
        "- Fondation technique : virtualisation, conteneurs, orchestration\n"
        "- Donnees : objet, bloc, fichier\n"
        "- Reseau : VPC, equilibrage, latence\n"
        "- Securite : responsabilite partagee, IAM, chiffrement\n"
        "- Economie : CapEx vs OpEx, metering, vendor lock-in",
    )
    _p(
        pdf,
        "Cette annexe resume environ quarante notions distinctes utilisees dans le "
        "document. Elle sert de checklist de couverture pour EduRAG lors des tests "
        "de generation de fiches completes sans questions de revision.",
    )
    _retain(
        pdf,
        "Fin du cours de test Cloud Computing pour EduRAG.",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT))
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
