import type { AskUserCard } from "../../lib/types";

interface AskCardProps {
  ask: AskUserCard;
  /** Sends an option as the next user message; absent while a run is busy. */
  onAnswer?: (answer: string) => void;
}

/** A question Aya asked while working. Options send themselves as the next
 * message; a free-text answer goes through the composer. */
export function AskCard({ ask, onAnswer }: AskCardProps) {
  const open = !ask.answered;
  return (
    <div className={`ask-card${open ? "" : " ask-card--answered"}`}>
      <div className="ask-card__question">{ask.question}</div>
      {ask.reason && open ? <div className="ask-card__reason">{ask.reason}</div> : null}
      {open && ask.options.length > 0 ? (
        <div className="ask-card__options">
          {ask.options.map((option) => (
            <button key={option} type="button" disabled={!onAnswer} onClick={() => onAnswer?.(option)}>
              {option}
            </button>
          ))}
        </div>
      ) : null}
      {open && ask.options.length === 0 ? (
        <div className="ask-card__hint">Cậu trả lời ở khung chat bên dưới nhé.</div>
      ) : null}
      {ask.answered ? <div className="ask-card__answer">Cậu đã trả lời: {ask.answered}</div> : null}
    </div>
  );
}
