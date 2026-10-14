import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  formatAnswer,
  isDontKnow,
  parseInlineSpans,
} from "./formatAnswer.js";

describe("isDontKnow", () => {
  it("matches exact phrase ignoring case and trailing period", () => {
    assert.equal(isDontKnow("I don't know"), true);
    assert.equal(isDontKnow("i don't know."), true);
    assert.equal(isDontKnow("  I DON'T KNOW  "), true);
  });

  it("matches curly apostrophes", () => {
    assert.equal(isDontKnow("I don\u2019t know"), true);
    assert.equal(isDontKnow("I don\u2018t know."), true);
  });

  it("rejects longer or different answers", () => {
    assert.equal(isDontKnow("I don't know the answer"), false);
    assert.equal(isDontKnow("Unknown"), false);
    assert.equal(isDontKnow(""), false);
    assert.equal(isDontKnow(null), false);
  });
});

describe("parseInlineSpans", () => {
  it("parses paired bold markers", () => {
    assert.deepEqual(parseInlineSpans("a **bold** z"), [
      { type: "text", text: "a " },
      { type: "bold", text: "bold" },
      { type: "text", text: " z" },
    ]);
  });

  it("keeps unbalanced ** as plain text", () => {
    assert.deepEqual(parseInlineSpans("a **oops"), [
      { type: "text", text: "a **oops" },
    ]);
  });

  it("keeps HTML-looking text as plain text spans", () => {
    const spans = parseInlineSpans('<script>alert(1)</script>');
    assert.deepEqual(spans, [
      { type: "text", text: '<script>alert(1)</script>' },
    ]);
  });
});

describe("formatAnswer", () => {
  it("returns [] for empty or null", () => {
    assert.deepEqual(formatAnswer(null), []);
    assert.deepEqual(formatAnswer(""), []);
    assert.deepEqual(formatAnswer(undefined), []);
  });

  it("formats a single paragraph", () => {
    const blocks = formatAnswer("Hello world.");
    assert.equal(blocks.length, 1);
    assert.equal(blocks[0].type, "paragraph");
    assert.deepEqual(blocks[0].spans, [{ type: "text", text: "Hello world." }]);
  });

  it("splits paragraphs on blank lines", () => {
    const blocks = formatAnswer("One.\n\nTwo.");
    assert.equal(blocks.length, 2);
    assert.equal(blocks[0].type, "paragraph");
    assert.equal(blocks[1].type, "paragraph");
  });

  it("formats hyphen and star bullet lists", () => {
    const blocks = formatAnswer("- alpha\n- beta\n* gamma");
    assert.equal(blocks.length, 1);
    assert.equal(blocks[0].type, "ul");
    assert.equal(blocks[0].items.length, 3);
    assert.deepEqual(blocks[0].items[0].spans, [
      { type: "text", text: "alpha" },
    ]);
  });

  it("formats numbered lists", () => {
    const blocks = formatAnswer("1. first\n2. second");
    assert.equal(blocks.length, 1);
    assert.equal(blocks[0].type, "ol");
    assert.equal(blocks[0].items.length, 2);
    assert.deepEqual(blocks[0].items[1].spans, [
      { type: "text", text: "second" },
    ]);
  });

  it("supports mixed paragraph then list then paragraph", () => {
    const blocks = formatAnswer("Intro\n\n- a\n- b\n\nOutro");
    assert.deepEqual(
      blocks.map((b) => b.type),
      ["paragraph", "ul", "paragraph"],
    );
  });

  it("applies bold inside list items", () => {
    const blocks = formatAnswer("- see **this**");
    assert.equal(blocks[0].items[0].spans[1].type, "bold");
    assert.equal(blocks[0].items[0].spans[1].text, "this");
  });

  it("does not interpret HTML as markup", () => {
    const blocks = formatAnswer("Use <b>tags</b> safely");
    const flat = blocks[0].spans.map((s) => s.text).join("");
    assert.equal(flat, "Use <b>tags</b> safely");
    assert.ok(blocks[0].spans.every((s) => s.type === "text" || s.type === "bold"));
  });

  it("handles very long text without dropping content", () => {
    const long = "word ".repeat(500).trim();
    const blocks = formatAnswer(long);
    assert.equal(blocks.length, 1);
    const joined = blocks[0].spans.map((s) => s.text).join("");
    assert.equal(joined, long);
  });
});
