import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import Home from "./components/Home.jsx";
import Documents from "./components/Documents.jsx";
import Assistant from "./components/Assistant.jsx";
import Revision from "./components/Revision.jsx";
import "./App.css";

const NAV_ITEMS = [
  { id: "home", label: "Accueil" },
  { id: "documents", label: "Documents" },
  { id: "assistant", label: "Assistant conversationnel" },
  { id: "revision", label: "Smart Revision" },
];

function renderActivePage(activeId) {
  switch (activeId) {
    case "documents":
      return <Documents />;
    case "assistant":
      return <Assistant />;
    case "revision":
      return <Revision />;
    case "home":
    default:
      return <Home />;
  }
}

function App() {
  const [activeId, setActiveId] = useState("home");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  function handleNavigate(id) {
    setActiveId(id);
    setSidebarOpen(false);
  }

  return (
    <div className={`app-shell${sidebarOpen ? " app-shell--sidebar-open" : ""}`}>
      <div
        className="app-shell__backdrop"
        onClick={() => setSidebarOpen(false)}
        aria-hidden={!sidebarOpen}
      />

      <Sidebar
        brand="EduRAG"
        items={NAV_ITEMS}
        activeId={activeId}
        onNavigate={handleNavigate}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="app-shell__main">
        <header className="app-shell__topbar">
          <button
            type="button"
            className="app-shell__menu"
            onClick={() => setSidebarOpen(true)}
            aria-label="Ouvrir le menu"
          >
            <span />
            <span />
            <span />
          </button>
          <div className="app-shell__brand-mobile" aria-hidden="true">
            EduRAG
          </div>
        </header>

        <main className="app-shell__content">{renderActivePage(activeId)}</main>
      </div>
    </div>
  );
}

export default App;
