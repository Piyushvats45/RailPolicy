import { useState } from "react";

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (!value.trim()) return;
    onSend(value);
    setValue("");
  }

  return (
    <form className="input-bar" onSubmit={handleSubmit}>
      <input
        className="input-field"
        type="text"
        placeholder="Ask about cancellations, refunds, TDR claims..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={disabled}
        aria-label="Ask a question"
      />
      <button className="send-btn" type="submit" disabled={disabled || !value.trim()}>
        Send
      </button>
    </form>
  );
}
