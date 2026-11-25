import "./PlaceholderPage.css";

function PlaceholderPage({ title, description }) {
  return (
    <section className="placeholder" aria-labelledby="placeholder-title">
      <p className="placeholder__eyebrow">À venir</p>
      <h1 id="placeholder-title" className="placeholder__title">
        {title}
      </h1>
      <p className="placeholder__lead">{description}</p>
    </section>
  );
}

export default PlaceholderPage;
