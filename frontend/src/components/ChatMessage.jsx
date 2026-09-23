import { useState } from "react";

export default function ChatMessage({ role, text, sources, isError }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const isUser = role === "user";

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--assistant"}`}>
      <div className="msg-speaker">{isUser ? "you" : "assistant"}</div>
      <div className={`msg-body ${isError ? "msg-body--error" : ""}`}>
        {text}

        {sources && sources.length > 0 && (
          <div className="sources-block">
            <button
              className="sources-toggle"
              onClick={() => setSourcesOpen((v) => !v)}
              aria-expanded={sourcesOpen}
            >
              {sourcesOpen ? "Hide sources" : `Show ${sources.length} source${sources.length > 1 ? "s" : ""}`}
            </button>

            {sourcesOpen && (
              <div className="ticket-stack">
                {sources.map((s) => (
                  <div className="ticket-stub" key={s.id}>
                    <div className="ticket-stub__meta">
                      <span>{s.id}</span>
                      <span>match {(s.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="ticket-stub__text">{s.text}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
