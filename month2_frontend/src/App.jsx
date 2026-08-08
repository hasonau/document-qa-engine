import { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  function handleSend(e) {
    e.preventDefault();

    const text = input.trim();
    if (!text) return;

    const userMessage = { role: "user", text };
    // Placeholder reply — replace with FastAPI /query call later
    const assistantMessage = {
      role: "assistant",
      text: "Got your question. (Mock reply — wire to /query next.)",
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Document Assistant</h1>
        <p>Ask questions about your uploaded documents</p>
      </header>

      <main className="messages">
        {messages.length === 0 ? (
          <p className="empty">No messages yet. Ask a question below.</p>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
              <span className="label">{msg.role}</span>
              <p>{msg.text}</p>
            </div>
          ))
        )}
      </main>

      <form className="composer" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          aria-label="Message"
        />
        <button type="submit" disabled={!input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
