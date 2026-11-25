import FeatureCard from "./FeatureCard.jsx";
import "./Home.css";

const FEATURES = [
  {
    id: "documents",
    title: "Documents",
    accent: "documents",
    description:
      "Déposez et organisez vos cours PDF pour constituer la base de connaissances d’EduRAG.",
  },
  {
    id: "assistant",
    title: "Assistant conversationnel",
    accent: "assistant",
    description:
      "Posez vos questions et obtenez des réponses ancrées dans vos documents, pas seulement dans la mémoire du modèle.",
  },
  {
    id: "revision",
    title: "Smart Revision",
    accent: "revision",
    description:
      "Générez des fiches de révision adaptées à votre niveau, votre langue et le passage du cours à retravailler.",
  },
];

function Home() {
  return (
    <section className="home" aria-labelledby="home-title">
      <div className="home__hero">
        <p className="home__eyebrow">Bienvenue</p>
        <h1 id="home-title" className="home__title">
          Bienvenue sur <span>EduRAG</span>
        </h1>
        <p className="home__lead">
          Votre espace d’étude intelligent. Cette première interface pose la structure visuelle
          de l’application : navigation claire, accueil accueillant, et présentation des trois
          fonctionnalités centrales.
        </p>
      </div>

      <div className="home__features">
        <div className="home__features-header">
          <h2>Trois piliers pour apprendre mieux</h2>
          <p>Les écrans métier arriveront ensuite ; pour l’instant, voici le panorama.</p>
        </div>

        <div className="home__grid">
          {FEATURES.map((feature) => (
            <FeatureCard
              key={feature.id}
              title={feature.title}
              description={feature.description}
              accent={feature.accent}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

export default Home;
