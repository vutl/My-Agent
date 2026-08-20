import { useEffect } from "react";

export interface FigurePreview {
  imageUrl: string;
  caption: string;
  filename?: string;
  pageNumber?: number | null;
}

interface FigureLightboxProps {
  figure: FigurePreview | null;
  onClose: () => void;
}

export function FigureLightbox({ figure, onClose }: FigureLightboxProps) {
  useEffect(() => {
    if (!figure) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [figure, onClose]);

  if (!figure) return null;

  return (
    <div className="figure-lightbox-backdrop" onClick={onClose} role="presentation">
      <div
        className="figure-lightbox-panel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Figure preview"
      >
        <button type="button" className="figure-lightbox-close" onClick={onClose} aria-label="Đóng">
          ×
        </button>
        <img src={figure.imageUrl} alt={figure.caption} className="figure-lightbox-image" />
        <div className="figure-lightbox-caption">
          <p>{figure.caption}</p>
          {(figure.filename || typeof figure.pageNumber === "number") && (
            <small>
              {figure.filename}
              {typeof figure.pageNumber === "number" ? ` · page ${figure.pageNumber}` : ""}
            </small>
          )}
        </div>
      </div>
    </div>
  );
}
