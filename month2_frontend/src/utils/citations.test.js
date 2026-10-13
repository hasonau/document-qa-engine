import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  formatPageLabel,
  normalizeCitations,
  SNIPPET_MAX,
} from "./citations.js";

describe("formatPageLabel", () => {
  it("maps a single 0-based page to 1-based p. N", () => {
    assert.equal(formatPageLabel(0, 0), "p. 1");
    assert.equal(formatPageLabel(1, 1), "p. 2");
  });

  it("maps a 0-based range to pp. A-B", () => {
    assert.equal(formatPageLabel(0, 1), "pp. 1-2");
  });

  it("omits the label when page fields are missing", () => {
    assert.equal(formatPageLabel(undefined, undefined), null);
    assert.equal(formatPageLabel(null, null), null);
    assert.equal(formatPageLabel("0", "0"), null);
  });

  it("uses the one valid page when only one side is present", () => {
    assert.equal(formatPageLabel(0, undefined), "p. 1");
    assert.equal(formatPageLabel(undefined, 2), "p. 3");
  });
});

describe("normalizeCitations", () => {
  it("returns [] for null, undefined, non-arrays", () => {
    assert.deepEqual(normalizeCitations(null), []);
    assert.deepEqual(normalizeCitations(undefined), []);
    assert.deepEqual(normalizeCitations({}), []);
    assert.deepEqual(normalizeCitations("x"), []);
  });

  it("returns [] for an empty array", () => {
    assert.deepEqual(normalizeCitations([]), []);
  });

  it("normalizes a cache-miss citations shape", () => {
    const rows = normalizeCitations([
      {
        sectionNumber: 1,
        chunkNumber: 2,
        heading: "Intro",
        startPage: 0,
        endPage: 0,
        chunk_text: "Hello world from the intro chunk.",
      },
    ]);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].heading, "Intro");
    assert.equal(rows[0].pageLabel, "p. 1");
    assert.equal(rows[0].fullText, "Hello world from the intro chunk.");
    assert.ok(rows[0].snippet.length <= SNIPPET_MAX + 1);
  });

  it("normalizes a cache-hit sources shape the same way", () => {
    const rows = normalizeCitations([
      {
        sectionNumber: 3,
        chunkNumber: 1,
        heading: "Details",
        startPage: 0,
        endPage: 1,
        chunk_text: "Passage A",
        document_id: "ignored-for-ui",
      },
    ]);
    assert.equal(rows[0].pageLabel, "pp. 1-2");
    assert.equal(rows[0].heading, "Details");
  });

  it("uses Untitled section when heading is missing", () => {
    const rows = normalizeCitations([
      {
        sectionNumber: 1,
        chunkNumber: 1,
        startPage: 0,
        endPage: 0,
        chunk_text: "x",
      },
    ]);
    assert.equal(rows[0].heading, "Untitled section");
  });

  it("omits pageLabel when pages are absent", () => {
    const rows = normalizeCitations([
      {
        sectionNumber: 1,
        chunkNumber: 1,
        heading: "H",
        chunk_text: "body",
      },
    ]);
    assert.equal(rows[0].pageLabel, null);
  });

  it("de-duplicates by sectionNumber+chunkNumber and keeps first order", () => {
    const rows = normalizeCitations([
      {
        sectionNumber: 1,
        chunkNumber: 1,
        heading: "First",
        startPage: 0,
        endPage: 0,
        chunk_text: "A",
      },
      {
        sectionNumber: 2,
        chunkNumber: 1,
        heading: "Second",
        startPage: 1,
        endPage: 1,
        chunk_text: "B",
      },
      {
        sectionNumber: 1,
        chunkNumber: 1,
        heading: "Duplicate",
        startPage: 0,
        endPage: 0,
        chunk_text: "A2",
      },
    ]);
    assert.equal(rows.length, 2);
    assert.equal(rows[0].heading, "First");
    assert.equal(rows[1].heading, "Second");
  });

  it("truncates long chunk text into a snippet with an ellipsis", () => {
    const long = "word ".repeat(80);
    const rows = normalizeCitations([
      {
        sectionNumber: 1,
        chunkNumber: 1,
        heading: "Long",
        startPage: 0,
        endPage: 0,
        chunk_text: long,
      },
    ]);
    assert.ok(rows[0].snippet.endsWith("…"));
    assert.ok(rows[0].snippet.length <= SNIPPET_MAX + 1);
    assert.equal(rows[0].fullText, long);
  });
});
