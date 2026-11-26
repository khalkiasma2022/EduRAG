import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  hostnameFromUrl,
  isPdfSource,
  isSafeHttpUrl,
  isWebSource,
  partitionSources,
  shortSnippet,
} from "./messageSources.js";

describe("messageSources", () => {
  it("accepte uniquement http et https", () => {
    assert.equal(isSafeHttpUrl("https://www.python.org/"), true);
    assert.equal(isSafeHttpUrl("http://example.com"), true);
    assert.equal(isSafeHttpUrl("ftp://example.com"), false);
    assert.equal(isSafeHttpUrl("javascript:alert(1)"), false);
    assert.equal(isSafeHttpUrl("not-a-url"), false);
    assert.equal(isSafeHttpUrl(""), false);
  });

  it("extrait le domaine sans www", () => {
    assert.equal(hostnameFromUrl("https://www.python.org/downloads/"), "python.org");
    assert.equal(hostnameFromUrl("https://fastapi.tiangolo.com/"), "fastapi.tiangolo.com");
  });

  it("détecte les sources PDF et Web", () => {
    const pdf = {
      text: "extrait",
      filename: "cours.pdf",
      page_number: 2,
      chunk_id: 1,
    };
    const web = {
      type: "web",
      title: "Python",
      url: "https://www.python.org/",
      snippet: "Official",
    };
    assert.equal(isPdfSource(pdf), true);
    assert.equal(isWebSource(pdf), false);
    assert.equal(isWebSource(web), true);
    assert.equal(isPdfSource(web), false);
  });

  it("partitionne PDF, Web et ignore les URL invalides", () => {
    const { pdf, web } = partitionSources([
      {
        text: "extrait",
        filename: "cours.pdf",
        page_number: 2,
        chunk_id: 1,
      },
      {
        type: "web",
        title: "Python",
        url: "https://www.python.org/",
        snippet: "Official",
      },
      {
        type: "web",
        title: "Bad",
        url: "ftp://evil.example/",
        snippet: "no",
      },
      null,
    ]);
    assert.equal(pdf.length, 1);
    assert.equal(web.length, 1);
    assert.equal(web[0].url, "https://www.python.org/");
  });

  it("gère l'absence de sources", () => {
    assert.deepEqual(partitionSources([]), { pdf: [], web: [] });
    assert.deepEqual(partitionSources(undefined), { pdf: [], web: [] });
  });

  it("tronque les extraits sans HTML", () => {
    assert.equal(shortSnippet("  bonjour   monde  "), "bonjour monde");
    assert.equal(shortSnippet("<b>x</b>"), "<b>x</b>");
    const long = "a".repeat(200);
    assert.equal(shortSnippet(long, 20).endsWith("…"), true);
    assert.ok(shortSnippet(long, 20).length <= 20);
  });
});
