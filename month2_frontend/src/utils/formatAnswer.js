/**
 * Detect the model's exact "I don't know" refusal (case-insensitive),
 * allowing a trailing period and curly apostrophes.
 */
export function isDontKnow(text) {
  if (text == null) return false;
  if (typeof text !== "string") return false;
  const normalized = text
    .trim()
    .toLowerCase()
    .replace(/\u2018|\u2019|\u02bc/g, "'")
    .replace(/\.$/, "");
  return normalized === "i don't know";
}

/**
 * Split a line into plain and bold spans. Unbalanced ** markers stay plain.
 * Never interprets HTML — callers must render spans as text nodes only.
 */
export function parseInlineSpans(line) {
  const input = line == null ? "" : String(line);
  const spans = [];
  let i = 0;
  let buffer = "";

  while (i < input.length) {
    if (input[i] === "*" && input[i + 1] === "*") {
      const close = input.indexOf("**", i + 2);
      if (close === -1) {
        buffer += input.slice(i);
        break;
      }
      if (buffer) {
        spans.push({ type: "text", text: buffer });
        buffer = "";
      }
      spans.push({ type: "bold", text: input.slice(i + 2, close) });
      i = close + 2;
      continue;
    }
    buffer += input[i];
    i += 1;
  }

  if (buffer) {
    spans.push({ type: "text", text: buffer });
  }

  if (spans.length === 0) {
    return [{ type: "text", text: "" }];
  }
  return spans;
}

function bulletMatch(line) {
  const m = /^(?:[-*]|\u2022)\s+(.*)$/.exec(line);
  return m ? m[1] : null;
}

function numberedMatch(line) {
  const m = /^(\d+)\.\s+(.*)$/.exec(line);
  return m ? { n: Number(m[1]), text: m[2] } : null;
}

/**
 * Turn plain answer text into a simple block structure for React rendering.
 * Supports paragraphs, "- "/"* " bullets, "1." numbered lines, and **bold**.
 *
 * @param {unknown} text
 * @returns {Array<{
 *   type: 'paragraph' | 'ul' | 'ol',
 *   spans?: Array<{type: 'text'|'bold', text: string}>,
 *   items?: Array<{spans: Array<{type: 'text'|'bold', text: string}>}>
 * }>}
 */
export function formatAnswer(text) {
  if (text == null) return [];
  const raw = typeof text === "string" ? text : String(text);
  if (raw === "") return [];

  const lines = raw.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
  const blocks = [];
  let i = 0;

  function flushParagraph(paraLines) {
    if (paraLines.length === 0) return;
    const joined = paraLines.join(" ").replace(/\s+/g, " ").trim();
    if (!joined) return;
    blocks.push({ type: "paragraph", spans: parseInlineSpans(joined) });
  }

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    const bullet = bulletMatch(trimmed);
    if (bullet != null) {
      const items = [];
      while (i < lines.length) {
        const t = lines[i].trim();
        if (!t) break;
        const b = bulletMatch(t);
        if (b == null) break;
        items.push({ spans: parseInlineSpans(b) });
        i += 1;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    const numbered = numberedMatch(trimmed);
    if (numbered != null) {
      const items = [];
      while (i < lines.length) {
        const t = lines[i].trim();
        if (!t) break;
        const n = numberedMatch(t);
        if (n == null) break;
        items.push({ spans: parseInlineSpans(n.text) });
        i += 1;
      }
      blocks.push({ type: "ol", items });
      continue;
    }

    const paraLines = [];
    while (i < lines.length) {
      const t = lines[i].trim();
      if (!t) break;
      if (bulletMatch(t) != null || numberedMatch(t) != null) break;
      paraLines.push(t);
      i += 1;
    }
    flushParagraph(paraLines);
  }

  return blocks;
}
