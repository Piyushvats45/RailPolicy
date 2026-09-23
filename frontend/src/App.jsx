import { useState, useRef, useEffect } from "react";
import ChatMessage from "./components/ChatMessage.jsx";
import ChatInput from "./components/ChatInput.jsx";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const STARTER_QUESTIONS = [
  "How much does cancelling 2 days before departure cost?",
  "Do I get a refund on a confirmed Tatkal ticket?",
  "What happens if my waitlisted ticket never confirms?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const [statusPulse, setStatusPulse] = useState(false);
  const [apiOnline, setApiOnline] = useState(null);
  const logEndRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((d) => setApiOnline(Boolean(d.pipeline_loaded)))
      .catch(() => setApiOnline(false));
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  async function sendQuestion(question) {
    if (!question.trim() || isThinking) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setIsThinking(true);

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: 3 }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Request failed.");
      }

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, sources: data.sources },
      ]);
      setStatusPulse(true);
      setTimeout(() => setStatusPulse(false), 900);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Couldn't reach the assistant. ${e.message}`,
          isError: true,
        },
      ]);
    } finally {
      setIsThinking(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="header-inner">
          <span
            className={`status-dot ${apiOnline ? "status-dot--on" : "status-dot--off"} ${
              statusPulse ? "status-dot--pulse" : ""
            }`}
            aria-hidden="true"
          />
          <h1 className="header-title">IRCTC Policy Assistant</h1>
          <span className="header-sub">cancellations &amp; refunds, sourced &amp; cited</span>
        </div>
      </header>

      <main className="log" role="log" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty-state">
            <p className="empty-state__lead">
              Ask about ticket cancellation charges, refund timelines, Tatkal rules,
              or TDR claims. Every answer is grounded in the actual policy text below it.
            </p>
            <div className="starter-row">
              {STARTER_QUESTIONS.map((q) => (
                <button
                  key={q}
                  className="starter-chip"
                  onClick={() => sendQuestion(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <ChatMessage key={i} role={m.role} text={m.text} sources={m.sources} isError={m.isError} />
        ))}

        {isThinking && (
          <div className="thinking-row" aria-label="Assistant is thinking">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
          </div>
        )}

        <div ref={logEndRef} />
      </main>

      <ChatInput onSend={sendQuestion} disabled={isThinking} />
    </div>
  );
}
