import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../lib/types";
import { MessageBubble } from "./MessageBubble";

interface ChatWindowProps {
  messages: ChatMessage[];
  isStreaming?: boolean;
}

export function ChatWindow({ messages, isStreaming = false }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastScrollAtRef = useRef(0);

  useEffect(() => {
    const now = Date.now();
    if (isStreaming && now - lastScrollAtRef.current < 120) {
      return;
    }
    lastScrollAtRef.current = now;
    bottomRef.current?.scrollIntoView({ behavior: isStreaming ? "auto" : "smooth", block: "end" });
  }, [messages, isStreaming]);

  return (
    <section className="chat-window" aria-live="polite">
      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
              <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="1.5" />
              <path d="M8 10.5c0-.83.67-1.5 1.5-1.5h3c.83 0 1.5.67 1.5 1.5v.5c0 .83-.67 1.5-1.5 1.5H11l-2 2v-2H9.5C8.67 13 8 12.33 8 11.5v-1Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
            </svg>
          </div>
          <p>
            <strong>Chào cậu~</strong>
            Hỏi gì cũng được — chat thường, tài liệu local, hoặc paper trong Library.
          </p>
        </div>
      ) : (
        <div className="chat-thread">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </section>
  );
}
