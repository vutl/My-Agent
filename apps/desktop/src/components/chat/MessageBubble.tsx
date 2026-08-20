import { memo, useState } from "react";
import type { ChatMessage, RetrievedDocument } from "../../lib/types";
import { API_BASE_URL } from "../../lib/api";
import {
  figureDisplayCaption,
  isLowSignalFigure,
} from "../../lib/figureCaption";
import { MarkdownBlock } from "./MarkdownBlock";
import { FigureLightbox, type FigurePreview } from "./FigureLightbox";

interface MessageBubbleProps {
  message: ChatMessage;
}

function modelShortLabel(model: string): string {
  if (model.startsWith("cc/")) return model.replace("cc/", "");
  if (model.startsWith("cx/")) return model.replace("cx/", "");
  return model;
}

export const MessageBubble = memo(function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <article className={`message-row ${message.role}`}>
      <div className="message-meta">
        {isUser ? (
          <span>Bạn</span>
        ) : (
          <>
            <span>Aya</span>
            {message.model && (
              <span className="model-badge">{modelShortLabel(message.model)}</span>
            )}
          </>
        )}
      </div>

      <div className={`message-bubble${message.pending ? " pending" : ""}`}>
        {message.content || message.pending ? (
          <MarkdownBlock
            content={message.content}
            sources={message.sources}
            isStreaming={Boolean(message.pending)}
          />
        ) : (
          <span className="message-placeholder">…</span>
        )}
        {message.role === "assistant" && !message.pending && <FigureAttachments message={message} />}
      </div>
    </article>
  );
});

function FigureAttachments({ message }: { message: ChatMessage }) {
  const figures = selectFigureAttachments(message.sources || []);
  const [preview, setPreview] = useState<FigurePreview | null>(null);

  if (figures.length === 0) return null;

  const compareLayout = figures.length === 2;

  return (
    <>
      <div className={`figure-attachments${compareLayout ? " figure-attachments--compare" : ""}`}>
        {figures.map((figure) => {
          const imageUrl = absoluteImageUrl(figure.image_url || "");
          const caption = figureDisplayCaption(figure);
          return (
            <figure className="figure-attachment" key={figure.figure_id || figure.image_url}>
              <button
                type="button"
                className="figure-attachment-open"
                onClick={() =>
                  setPreview({
                    imageUrl,
                    caption,
                    filename: figure.filename,
                    pageNumber: figure.page_number,
                  })
                }
                aria-label={`Xem lớn: ${caption}`}
              >
                <img alt={caption} src={imageUrl} />
                <span className="figure-attachment-zoom" aria-hidden="true">
                  Phóng to
                </span>
              </button>
              <figcaption>
                <span>{caption}</span>
                <small>
                  {figure.filename}
                  {typeof figure.page_number === "number" ? ` · page ${figure.page_number}` : ""}
                </small>
              </figcaption>
            </figure>
          );
        })}
      </div>
      <FigureLightbox figure={preview} onClose={() => setPreview(null)} />
    </>
  );
}

function absoluteImageUrl(imageUrl: string): string {
  if (imageUrl.startsWith("http://") || imageUrl.startsWith("https://")) return imageUrl;
  return `${API_BASE_URL}${imageUrl.startsWith("/") ? "" : "/"}${imageUrl}`;
}

function selectFigureAttachments(sources: RetrievedDocument[]): RetrievedDocument[] {
  const withFigure = sources
    .filter((source) => source.image_url && source.figure_id)
    .filter((figure) => !isLowSignalFigure(figure));

  if (withFigure.length === 0) return [];

  // Backend has the original user query and performs scoped, evidence-aware
  // figure curation. Preserve that order; inferring "Figure N" from retrieved
  // excerpt text can boost a figure the user never requested.
  const unique = withFigure.filter(
    (figure, index, all) => all.findIndex((item) => item.figure_id === figure.figure_id) === index
  );

  const documentIds = new Set(
    sources.map((source) => source.document_id).filter((documentId): documentId is string => Boolean(documentId))
  );
  const figureDocIds = new Set(
    unique.map((figure) => figure.document_id).filter((documentId): documentId is string => Boolean(documentId))
  );
  const multiDocument = documentIds.size >= 2 || figureDocIds.size >= 2;

  if (multiDocument) {
    const byDocument = new Map<string, RetrievedDocument[]>();
    for (const figure of unique) {
      if (!figure.document_id) continue;
      const bucket = byDocument.get(figure.document_id) || [];
      bucket.push(figure);
      byDocument.set(figure.document_id, bucket);
    }

    const orderedDocIds = [...documentIds].filter((documentId) => byDocument.has(documentId));
    if (orderedDocIds.length < byDocument.size) {
      for (const documentId of byDocument.keys()) {
        if (!orderedDocIds.includes(documentId)) orderedDocIds.push(documentId);
      }
    }

    const selected: RetrievedDocument[] = [];
    for (const documentId of orderedDocIds) {
      const best = byDocument.get(documentId)?.[0];
      if (best) selected.push(best);
      if (selected.length >= 4) break;
    }
    return selected;
  }

  const textSources = sources.filter((source) => !source.figure_id);
  const docCounts = new Map<string, number>();
  for (const source of textSources) {
    if (!source.document_id) continue;
    docCounts.set(source.document_id, (docCounts.get(source.document_id) || 0) + 1);
  }

  let primaryDoc: string | undefined;
  if (docCounts.size === 1) {
    primaryDoc = [...docCounts.keys()][0];
  } else if (docCounts.size > 1) {
    primaryDoc = [...docCounts.entries()].sort((left, right) => right[1] - left[1])[0][0];
  }

  return unique
    .filter((figure) => !primaryDoc || figure.document_id === primaryDoc)
    .slice(0, 3);
}
