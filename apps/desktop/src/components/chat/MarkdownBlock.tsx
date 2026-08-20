import { Fragment, ReactNode, useMemo } from "react";

import type { RetrievedDocument } from "../../lib/types";

interface MarkdownBlockProps {
  content: string;
  sources?: RetrievedDocument[];
  isStreaming?: boolean;
}

type Block =
  | { type: "heading"; level: number; content: string }
  | { type: "ul"; items: string[] }
  | { type: "ol"; items: string[] }
  | { type: "table"; headers: string[]; rows: string[][] }
  | { type: "code"; lang: string; content: string }
  | { type: "paragraph"; content: string };

export function MarkdownBlock({ content, sources = [], isStreaming = false }: MarkdownBlockProps) {
  const blocks = useMemo(
    () => parseBlocks(content, { isStreaming }),
    [content, isStreaming]
  );
  if (blocks.length === 0) {
    return isStreaming ? <div className="markdown-block streaming" aria-busy="true" /> : null;
  }
  const sourceMap = buildSourceMap(sources);

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

function renderBlock(block: Block, index: number, sourceMap: Map<string, RetrievedDocument>): ReactNode {
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
  sourceMap: Map<string, RetrievedDocument>
): ReactNode {
  const content = renderInline(text, sourceMap);
  if (level === 1) return <h1 key={key}>{content}</h1>;
  if (level === 2) return <h2 key={key}>{content}</h2>;
  if (level === 3) return <h3 key={key}>{content}</h3>;
  return <h4 key={key}>{content}</h4>;
}

function renderInline(text: string, sourceMap: Map<string, RetrievedDocument>): ReactNode[] {
  const parts = text
    .split(/(\$\s*\\?[A-Za-z]+(?:\{[^}]+\})?\s*\$|\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|\[[^\]]+\])/g)
    .filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith("$") && part.endsWith("$")) {
      return <span key={index}>{renderMathInline(part.slice(1, -1))}</span>;
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
            {label}
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

function buildSourceMap(sources: RetrievedDocument[]): Map<string, RetrievedDocument> {
  const sourceMap = new Map<string, RetrievedDocument>();
  sources.forEach((source, index) => {
    if (source.citation_label) {
      sourceMap.set(source.citation_label, source);
    }
    sourceMap.set(`SOURCE ${index + 1}`, source);
  });
  return sourceMap;
}

function sourceTitle(source: RetrievedDocument): string {
  const page = typeof source.page_number === "number" ? ` page ${source.page_number}` : "";
  return `${source.filename}${page}\n${source.source_path}\nClick to copy source path`;
}

function copySource(source: RetrievedDocument) {
  const page = typeof source.page_number === "number" ? `#page=${source.page_number}` : "";
  void navigator.clipboard?.writeText(`${source.source_path}${page}`);
}
