import "./FeatureCard.css";

function FeatureCard({ title, description, accent }) {
  return (
    <article className={`feature-card feature-card--${accent}`}>
      <div className="feature-card__accent" aria-hidden="true" />
      <h3 className="feature-card__title">{title}</h3>
      <p className="feature-card__description">{description}</p>
    </article>
  );
}

export default FeatureCard;
