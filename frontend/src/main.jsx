import { StrictMode, useEffect } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { checkHealth } from "./api/health.js";
import "./index.css";

function Root() {
  useEffect(() => {
    checkHealth()
      .then((result) => {
        console.log("Connexion EduRAG API OK:", result);
      })
      .catch((error) => {
        console.error("Connexion EduRAG API échouée:", error);
      });
  }, []);

  return <App />;
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
