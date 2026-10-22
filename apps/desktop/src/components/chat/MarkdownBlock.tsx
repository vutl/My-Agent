import { Fragment, ReactNode, useMemo } from "react";
import katex from "katex";

import type { Citation, RetrievedDocument } from "../../lib/types";

interface MarkdownBlockProps {
  content: string;
  sources?: RetrievedDocument[];
  citations?: Citation[];
  isStreaming?: boolean;
}

interface SourceEntry {
  filename?: string | null;
  source_path?: string | null;
  page_number?: number | null;
  chip?: string;
  note?: string;
  url?: string | null;
}
type SourceMap = Map<string, SourceEntry>;

type Block =
  | { type: "heading"; level: number; content: string }
  | { type: "ul"; items: string[] }
  | { type: "ol"; items: string[] }
  | { type: "table"; headers: string[]; rows: string[][] }
  | { type: "code"; lang: string; content: string }
  | { type: "math"; content: string }
  | { type: "paragraph"; content: string };

export function MarkdownBlock({ content, sources = [], citations = [], isStreaming = false }: MarkdownBlockProps) {
  const blocks = useMemo(
    () => parseBlocks(content, { isStreaming }),
    [content, isStreaming]
  );
  if (blocks.length === 0) {
    return isStreaming ? <div className="markdown-block streaming" aria-busy="true" /> : null;
  }
  const sourceMap = buildSourceMap(sources, citations);

  return (
    <div className={`markdown-block${isStreaming ? " streaming" : ""}`} aria-busy={isStreaming}>
      {blocks.map((block, index) => renderBlock(block, index, sourceMap))}
    </div>
  );
}

function parseBlocks(content: string, options: { isStreaming?: boolean } = {}): Block[] {
  const lines = content.split("\n");
  const blocks: Block[] = [];
  let index = 0;

  while (index < lines.length) {
    while (index < lines.length && !lines[index].trim()) {
      index += 1;
    }
    if (index >= lines.length) {
      break;
    }

    const line = lines[index];
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim().toLowerCase();
      index += 1;
      const codeLines: string[] = [];
      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      blocks.push({ type: "code", lang, content: codeLines.join("\n").trimEnd() });
      continue;
    }

    // Display math: $$ ... $$ or \[ ... \], on one line or spanning lines.
    const mathOpen = line.trim().match(/^(\$\$|\\\[)/);
    if (mathOpen) {
      const close = mathOpen[1] === "$$" ? "$$" : "\\]";
      const first = line.trim().slice(mathOpen[1].length);
      if (first.trimEnd().endsWith(close) && first.trim().length > close.length) {
        blocks.push({ type: "math", content: first.trimEnd().slice(0, -close.length) });
        index += 1;
        continue;
      }
      const mathLines: string[] = first.trim() ? [first] : [];
      let cursor = index + 1;
      let closed = false;
      while (cursor < lines.length) {
        const current = lines[cursor];
        if (current.trimEnd().endsWith(close)) {
          mathLines.push(current.trimEnd().slice(0, -close.length));
          closed = true;
          cursor += 1;
          break;
        }
        mathLines.push(current);
        cursor += 1;
      }
      if (closed || options.isStreaming) {
        if (closed) {
          blocks.push({ type: "math", content: mathLines.join("\n") });
          index = cursor;
          continue;
        }
        break; // unfinished formula while streaming: wait for the closing delimiter
      }
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, content: heading[2].trim() });
      index += 1;
      continue;
    }

    if (isTableStart(lines, index)) {
      const table = parseTable(lines, index);
      blocks.push(table.block);
      index = table.nextIndex;
      continue;
    }

    if (/^\s*(-|\*)\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*(-|\*)\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*(-|\*)\s+/, "").trim());
        index += 1;
      }
      blocks.push({ type: "ul", items });
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+\.\s+/, "").trim());
        index += 1;
      }
      blocks.push({ type: "ol", items });
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length) {
      const current = lines[index];
      if (!current.trim()) {
        break;
      }
      if (
        current.startsWith("```") ||
        /^(\$\$|\\\[)/.test(current.trim()) ||
        /^(#{1,4})\s+/.test(current) ||
        isTableStart(lines, index) ||
        /^\s*(-|\*)\s+/.test(current) ||
        /^\s*\d+\.\s+/.test(current)
      ) {
        break;
      }
      paragraphLines.push(current);
      index += 1;
    }
    if (paragraphLines.length > 0) {
      blocks.push({ type: "paragraph", content: paragraphLines.join("\n") });
    }
  }

  if (options.isStreaming) {
    const tail = flushStreamingTail(lines, index);
    if (tail) {
      blocks.push(tail);
    }
  }

  return blocks;
}

function flushStreamingTail(lines: string[], start: number): Block | null {
  if (start >= lines.length) {
    return null;
  }

  const remaining = lines.slice(start).join("\n").trimEnd();
  if (!remaining) {
    return null;
  }

  if (remaining.startsWith("```")) {
    const firstLine = lines[start];
    const lang = firstLine.slice(3).trim().toLowerCase();
    const codeLines = lines.slice(start + 1);
    return { type: "code", lang, content: codeLines.join("\n") };
  }

  return { type: "paragraph", content: remaining };
}

function isTableStart(lines: string[], index: number): boolean {
  if (!lines[index]?.includes("|")) {
    return false;
  }
  const separator = lines[index + 1] ?? "";
  return /^\s*\|?[\s|:-]+\|[\s|:-]*$/.test(separator);
}

function parseTable(lines: string[], start: number): { block: Block; nextIndex: number } {
  const headerCells = splitTableRow(lines[start]);
  let index = start + 2;
  const rows: string[][] = [];
  while (index < lines.length && lines[index].includes("|")) {
    rows.push(splitTableRow(lines[index]));
    index += 1;
  }
  return {
    block: { type: "table", headers: headerCells, rows },
    nextIndex: index,
  };
}

function splitTableRow(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return trimmed.split("|").map((cell) => cell.trim());
}

function renderBlock(block: Block, index: number, sourceMap: SourceMap): ReactNode {
  switch (block.type) {
    case "heading":
      return renderHeading(block.level, block.content, index, sourceMap);
    case "ul":
      return (
        <ul key={index}>
          {block.items.map((item, itemIndex) => (
            <li key={itemIndex}>{renderInline(item, sourceMap)}</li>
          ))}
        </ul>
      );
    case "ol":
      return (
        <ol key={index}>
          {block.items.map((item, itemIndex) => (
            <li key={itemIndex}>{renderInline(item, sourceMap)}</li>
          ))}
        </ol>
      );
    case "table":
      return (
        <div className="markdown-table-wrap" key={index}>
          <table className="markdown-table">
            <thead>
              <tr>
                {block.headers.map((header, headerIndex) => (
                  <th key={headerIndex}>{renderInline(header, sourceMap)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex}>{renderInline(cell, sourceMap)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "code": {
      const isDiagram =
        ["text", "ascii", "diagram", "architecture", "pipeline"].includes(block.lang) ||
        /[│▼►┌┐└┘─→]/.test(block.content);
      return (
        <pre key={index} className={isDiagram ? "diagram-block" : undefined}>
          <code>{block.content}</code>
        </pre>
      );
    }
    case "math":
      return (
        <div
          className="math-block"
          key={index}
          dangerouslySetInnerHTML={{ __html: renderLatex(block.content, true) }}
        />
      );
    case "paragraph":
      return (
        <p key={index}>
          {block.content.split("\n").map((line, lineIndex) => (
            <Fragment key={lineIndex}>
              {lineIndex > 0 ? <br /> : null}
              {renderInline(line, sourceMap)}
            </Fragment>
          ))}
        </p>
      );
    default:
      return null;
  }
}

function renderHeading(
  level: number,
  text: string,
  key: string | number,
  sourceMap: SourceMap
): ReactNode {
  const content = renderInline(text, sourceMap);
  if (level === 1) return <h1 key={key}>{content}</h1>;
  if (level === 2) return <h2 key={key}>{content}</h2>;
  if (level === 3) return <h3 key={key}>{content}</h3>;
  return <h4 key={key}>{content}</h4>;
}

function renderInline(text: string, sourceMap: SourceMap): ReactNode[] {
  const parts = text
    .split(/(\\\(.+?\\\)|\$\$[^$\n]+\$\$|\$[^$\n]+\$|\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\[[^\]]+\])/g)
    .filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith("\\(") && part.endsWith("\\)")) {
      return (
        <span key={index} dangerouslySetInnerHTML={{ __html: renderLatex(part.slice(2, -2), false) }} />
      );
    }
    if (part.startsWith("$$") && part.endsWith("$$") && part.length > 4) {
      return <span key={index} dangerouslySetInnerHTML={{ __html: renderLatex(part.slice(2, -2), false) }} />;
    }
    if (part.startsWith("$") && part.endsWith("$") && part.length > 2) {
      const body = part.slice(1, -1);
      // "$5 and $10" is money, not math: require a LaTeX-looking body.
      if (looksLikeLatex(body)) {
        return <span key={index} dangerouslySetInnerHTML={{ __html: renderLatex(body, false) }} />;
      }
      return <Fragment key={index}>{renderPlainText(part, index)}</Fragment>;
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("[") && part.endsWith("]")) {
      const label = part.slice(1, -1);
      const source = sourceMap.get(label);
      if (source) {
        return (
          <button
            className="source-chip"
            key={index}
            onClick={() => copySource(source)}
            title={sourceTitle(source)}
            type="button"
          >
            {source.chip ?? label}
          </button>
        );
      }
    }
    if (/^\[SOURCE \d+\]$/.test(part)) {
      return (
        <span className="source-chip" key={index}>
          {part}
        </span>
      );
    }
    return renderPlainText(part, index);
  });
}

function renderPlainText(text: string, key: string | number): ReactNode {
  const cleaned = cleanupOrphanMarkdown(text);
  const label = cleaned.match(/^([^:\n]{2,64}:)(\s+.+)$/);
  if (label && !/[.!?]$/.test(label[1])) {
    return (
      <Fragment key={key}>
        <strong>{label[1]}</strong>
        {label[2]}
      </Fragment>
    );
  }
  return <Fragment key={key}>{cleaned}</Fragment>;
}

function cleanupOrphanMarkdown(text: string): string {
  return text
    .replace(/:\*(?=\s|$)/g, ":")
    .replace(/([\p{L}\p{N})\]])\*(?=([.,;:!?)]|\s|$))/gu, "$1")
    .replace(/^\*\s+(?=\S)/, "");
}

function looksLikeLatex(body: string): boolean {
  const trimmed = body.trim();
  if (!trimmed || /^\s|\s$/.test(body)) return false;
  return /[\\^_{}=]/.test(trimmed) || /^[A-Za-z]$/.test(trimmed);
}

function renderLatex(source: string, displayMode: boolean): string {
  try {
    return katex.renderToString(source, { displayMode, throwOnError: false, strict: "ignore" });
  } catch {
    return escapeHtml(renderMathInline(source));
  }
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char] ?? char);
}

function renderMathInline(raw: string): string {
  const normalized = raw.trim();
  const replacements: Record<string, string> = {
    "\\rightarrow": "→",
    "rightarrow": "→",
    "\\to": "→",
    "to": "→",
    "\\leftarrow": "←",
    "leftarrow": "←",
    "\\leftrightarrow": "↔",
    "leftrightarrow": "↔",
    "\\Rightarrow": "⇒",
    "Rightarrow": "⇒",
    "\\times": "×",
    "times": "×",
    "\\cdot": "·",
    "cdot": "·",
    "\\alpha": "α",
    "alpha": "α",
    "\\beta": "β",
    "beta": "β",
    "\\gamma": "γ",
    "gamma": "γ",
    "\\sigma": "σ",
    "sigma": "σ",
  };
  return replacements[normalized] ?? normalized.replace(/\\/g, "");
}

function buildSourceMap(sources: RetrievedDocument[], citations: Citation[]): SourceMap {
  const sourceMap: SourceMap = new Map();
  sources.forEach((source, index) => {
    if (source.citation_label) {
      sourceMap.set(source.citation_label, source);
    }
    sourceMap.set(`SOURCE ${index + 1}`, source);
  });
  for (const citation of citations) {
    if (!citation.id) continue;
    const page = typeof citation.page_number === "number" ? ` p.${citation.page_number}` : "";
    const whole = citation.whole_document ? " (whole document)" : "";
    sourceMap.set(citation.id, {
      filename: citation.filename,
      source_path: citation.source_path,
      page_number: citation.page_number,
      chip: citation.id,
      url: citation.url,
      note: citation.url
        ? `${citation.title || citation.url}\n${citation.url}`
        : `${citation.filename || citation.title || citation.id}${page}${whole}`,
    });
  }
  return sourceMap;
}

function sourceTitle(source: SourceEntry): string {
  if (source.note) return source.note;
  const page = typeof source.page_number === "number" ? ` page ${source.page_number}` : "";
  return `${source.filename}${page}\n${source.source_path}\nClick to copy source path`;
}

function copySource(source: SourceEntry) {
  if (source.url) {
    // Web sources open in the browser; the link is copied as well in case the
    // desktop shell blocks new windows.
    void navigator.clipboard?.writeText(source.url);
    window.open(source.url, "_blank", "noopener,noreferrer");
    return;
  }
  if (!source.source_path) return;
  const page = typeof source.page_number === "number" ? `#page=${source.page_number}` : "";
  void navigator.clipboard?.writeText(`${source.source_path}${page}`);
}
