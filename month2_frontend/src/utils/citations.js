/** Max characters for the collapsed snippet. */
export const SNIPPET_MAX = 160;

/**
 * Build a 1-based page label from 0-based start/end, or null if pages are missing.
 * Single page: "p. 1". Range: "pp. 1-2". Never returns "page 0" / "undefined".
 */
export function formatPageLabel(startPage, endPage) {
  const startOk = typeof startPage === "number" && Number.isFinite(startPage);
  const endOk = typeof endPage === "number" && Number.isFinite(endPage);

  if (!startOk && !endOk) return null;

  const start = startOk ? startPage : endPage;
  const end = endOk ? endPage : startPage;
  const displayStart = start + 1;
  const displayEnd = end + 1;

  if (displayStart === displayEnd) {
    return `p. ${displayStart}`;
  }
  return `pp. ${displayStart}-${displayEnd}`;
}

function asText(value) {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return String(value);
}

function snippetOf(text) {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length <= SNIPPET_MAX) return cleaned;
  return `${cleaned.slice(0, SNIPPET_MAX).trimEnd()}…`;
}

/**
 * Normalize cache-miss `citations` or cache-hit SSE `sources` into UI rows.
 * Both are arrays of chunk-like objects; meaning may differ (cited-only vs all
 * fused chunks) but the shape is the same.
 *
 * @param {unknown} input - citations array, sources array, null, or undefined
 * @returns {Array<{
 *   key: string,
 *   heading: string,
 *   pageLabel: string | null,
 *   snippet: string,
 *   fullText: string,
 *   sectionNumber: unknown,
 *   chunkNumber: unknown,
 * }>}
 */
export function normalizeCitations(input) {
  if (input == null) return [];
  if (!Array.isArray(input)) return [];

  const seen = new Set();
  const out = [];

  for (const raw of input) {
    if (!raw || typeof raw !== "object") continue;

    const sectionNumber = raw.sectionNumber;
    const chunkNumber = raw.chunkNumber;
    const dedupeKey = `${sectionNumber ?? ""}:${chunkNumber ?? ""}`;
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);

    const headingRaw = asText(raw.heading).trim();
    const fullText = asText(raw.chunk_text);
    const pageLabel = formatPageLabel(raw.startPage, raw.endPage);

    out.push({
      key: dedupeKey || `row-${out.length}`,
      heading: headingRaw || "Untitled section",
      pageLabel,
      snippet: snippetOf(fullText),
      fullText,
      sectionNumber,
      chunkNumber,
    });
  }

  return out;
}
