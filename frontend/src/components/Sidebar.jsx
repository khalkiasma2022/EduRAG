import "./Sidebar.css";

function Sidebar({ brand, items, activeId, onNavigate, open, onClose }) {
  return (
    <aside className={`sidebar${open ? " sidebar--open" : ""}`} aria-label="Navigation principale">
      <div className="sidebar__header">
        <div className="sidebar__brand">
          <span className="sidebar__mark" aria-hidden="true">
            E
          </span>
          <div>
            <p className="sidebar__title">{brand}</p>
            <p className="sidebar__subtitle">Assistant d&apos;étude</p>
          </div>
        </div>

        <button
          type="button"
          className="sidebar__close"
          onClick={onClose}
          aria-label="Fermer le menu"
        >
          ×
        </button>
      </div>

      <nav className="sidebar__nav">
        <p className="sidebar__section">Navigation</p>
        <ul className="sidebar__list">
          {items.map((item) => {
            const isActive = item.id === activeId;

            return (
              <li key={item.id}>
                <button
                  type="button"
                  className={`sidebar__link${isActive ? " sidebar__link--active" : ""}`}
                  onClick={() => onNavigate(item.id)}
                  aria-current={isActive ? "page" : undefined}
                >
                  <span className="sidebar__dot" aria-hidden="true" />
                  {item.label}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <footer className="sidebar__footer">
        <p>Interface visuelle initiale</p>
        <p>API à brancher plus tard</p>
      </footer>
    </aside>
  );
}

export default Sidebar;
