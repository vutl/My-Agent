import type { RetrievedDocument } from "./types";

const CAPTION_START = /^(fig\.?|figure|hình)\s*(\d+)\s*[.:–-]/i;
const FIGURE_NUMBER_START = /^(fig\.?|figure|hình)\s*(\d+)\b/i;
const HYPHEN_BREAK = /-\s*\n+\s*/g;

export function normalizeCaptionText(text: string): string {
  return text
    .replace(HYPHEN_BREAK, "")
    .replace(/-\s+(?=[a-z])/g, "")
    .replace(/\s*\n+\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function dedupeRepeatedFigurePrefix(text: string): string {
  const normalized = normalizeCaptionText(text);
  if (!normalized) return normalized;

  const match = normalized.match(CAPTION_START);
  if (!match || match.index === undefined) return normalized;

  const figureNumber = match[2];
  const splitter = new RegExp(
    `(?=(?:fig\\.?|figure|hình)\\s*${figureNumber}\\s*[.:–-])`,
    "i"
  );
  const parts = normalized
    .split(splitter)
    .map((part) => part.trim())
    .filter(Boolean);
  if (parts.length <= 1) return normalized;
  return parts.reduce((longest, part) => (part.length > longest.length ? part : longest), "");
}

export function extractFigureTitleSentence(text: string, maxChars = 180): string {
  const normalized = dedupeRepeatedFigurePrefix(text);
  if (!normalized) return normalized;

  const match = normalized.match(CAPTION_START);
  if (!match || match.index === undefined) {
    return normalized.length > maxChars ? `${normalized.slice(0, maxChars - 1)}…` : normalized;
  }

  const body = normalized.slice(match.index);
  const period = body.indexOf(". ");
  if (period > 20 && period < maxChars) {
    return body.slice(0, period + 1).trim();
  }
  if (body.length <= maxChars) return body.trim();
  return `${body.slice(0, maxChars - 1).trim()}…`;
}

function extractFigureCaptionFromContent(
  content: string,
  figureNumber?: number | null
): string | null {
  const paragraphs = content.replace(HYPHEN_BREAK, "").split(/\n\s*\n/);

  for (const paragraph of paragraphs) {
    const stripped = paragraph.replace(/\s+/g, " ").trim();
    if (!stripped) continue;
    const match = stripped.match(CAPTION_START);
    if (!match) continue;
    const number = Number(match[2]);
    if (typeof figureNumber === "number" && number !== figureNumber) continue;
    return extractFigureTitleSentence(stripped);
  }

  return null;
}

export function figureDisplayCaption(figure: RetrievedDocument): string {
  const candidates: string[] = [];

  if (figure.caption) {
    candidates.push(extractFigureTitleSentence(figure.caption));
  }

  if (figure.content) {
    const extracted = extractFigureCaptionFromContent(
      figure.content,
      labeledFigureNumber(figure)
    );
    if (extracted) candidates.push(extracted);
  }

  if (candidates.length === 0) {
    if (figure.figure_label) return figure.figure_label;
    return "Figure";
  }

  candidates.sort((left, right) => right.length - left.length);
  return candidates[0];
}

export function isLowSignalFigure(figure: RetrievedDocument): boolean {
  const assetKind = (figure.asset_kind || "").trim().toLowerCase();
  if (figure.quality_status === "rejected" || figure.is_content === false) return true;
  if (["branding", "logo", "decorative", "publisher_mark"].includes(assetKind)) return true;
  const caption = (figure.caption || "").trim().toLowerCase();
  const content = (figure.content || "").trim().toLowerCase();
  return (
    caption.startsWith("figure extracted from page") ||
    content.startsWith("figure extracted from page") ||
    caption.includes("visual fallback") ||
    content.includes("visual fallback")
  );
}

export function figureRelevanceScore(
  figure: RetrievedDocument,
  preferredFigureNumber?: number | null
): number {
  let score = figure.score || 0;
  const caption = (figure.caption || "").toLowerCase();
  const content = (figure.content || "").toLowerCase();

  if (isLowSignalFigure(figure)) return score - 200;

  if (preferredFigureNumber) {
    if (labeledFigureNumber(figure) === preferredFigureNumber) score += 80;
  }

  for (const hint of ["architecture", "overview", "structure", "proposed model", "pipeline"]) {
    if (caption.includes(hint)) score += 25;
    if (content.includes(hint)) score += 10;
  }

  return score;
}

function labeledFigureNumber(figure: RetrievedDocument): number | null {
  if (typeof figure.figure_number === "number") return figure.figure_number;
  for (const value of [figure.figure_label, figure.caption]) {
    const match = (value || "").match(FIGURE_NUMBER_START);
    if (match) return Number(match[2]);
  }
  return null;
}

export function requestedFigureNumber(sources: RetrievedDocument[]): number | null {
  for (const source of sources) {
    if (source.figure_id) continue;
    const text = (source.content || "").toLowerCase();
    const match =
      text.match(/\bfig(?:ure)?\.?\s*(\d+)\b/) || text.match(/\bhình\s*(\d+)\b/);
    if (match) return Number(match[1]);
  }
  return null;
}
