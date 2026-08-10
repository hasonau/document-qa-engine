import { useState } from "react";
import "./App.css";

function updateLastAssistant(prev, patch) {
  const next = [...prev];
  const last = next[next.length - 1];
  if (!last || last.role !== "assistant") return prev;
  next[next.length - 1] = { ...last, ...patch(last) };
  return next;
}

/** Parse SSE frames from a POST /query ReadableStream. */
async function readQuerySse(res, { onAnswerToken, onSources, onNotFound }) {
  if (!res.body) {
    throw new Error("No response body");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName = "message";
  let dataLines = [];

  function dispatch() {
    if (dataLines.length === 0) {
      eventName = "message";
      return;
    }

    const data = dataLines.join("\n");
    dataLines = [];
    const currentEvent = eventName;
    eventName = "message";

    if (currentEvent === "answer") {
      onAnswerToken(data);
    } else if (currentEvent === "sources") {
      try {
        onSources(JSON.parse(data));
      } catch {
        // ignore malformed sources payload
      }
    } else if (currentEvent === "not_found") {
      onNotFound(data || "Not in Documents");
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line === "") {
        dispatch();
        continue;
      }
      if (line.startsWith(":")) continue; // SSE comment / keepalive
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
        continue;
      }
      if (line.startsWith("data:")) {
        // Spec: optional single space after "data:"
        dataLines.push(line.slice(5).startsWith(" ") ? line.slice(6) : line.slice(5));
      }
    }
  }

  // Flush a trailing event if the stream ended without a blank line
  if (buffer.trim()) {
    const line = buffer;
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).startsWith(" ") ? line.slice(6) : line.slice(5));
    }
  }
  dispatch();
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const [documentId, setDocumentId] = useState(null);
  const [streaming, setStreaming] = useState(false);

  async function handleSend(e) {
    e.preventDefault();

    const text = input.trim();
    if (!text || streaming) return;

    const userMessage = { role: "user", text };
    const assistantMessage = { role: "assistant", text: "", sources: null };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setStreaming(true);

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          query: text,
          document_id: documentId ?? "",
        }),
      });

      if (!res.ok) {
        setMessages((prev) =>
          updateLastAssistant(prev, () => ({
            text: "Something went wrong, try again",
            sources: null,
          })),
        );
        return;
      }

      await readQuerySse(res, {
        onAnswerToken(token) {
          setMessages((prev) =>
            updateLastAssistant(prev, (last) => ({
              text: `${last.text ?? ""}${token}`,
            })),
          );
        },
        onSources(sources) {
          setMessages((prev) =>
            updateLastAssistant(prev, () => ({
              sources: Array.isArray(sources) ? sources : null,
            })),
          );
        },
        onNotFound(message) {
          setMessages((prev) =>
            updateLastAssistant(prev, () => ({
              text: message,
              sources: null,
            })),
          );
        },
      });

      // If stream closed with no tokens and no not_found text, show a fallback
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && !last.text) {
          return updateLastAssistant(prev, () => ({
            text: "Something went wrong, try again",
          }));
        }
        return prev;
      });
    } catch {
      setMessages((prev) =>
        updateLastAssistant(prev, (last) => ({
          text: last.text
            ? `${last.text}\n\n(Connection interrupted)`
            : "Something went wrong, try again",
        })),
      );
    } finally {
      setStreaming(false);
    }
  }

  async function handleUpload() {
    if (!selectedFile || uploading) return;

    setUploading(true);
    setUploadStatus("Uploading...");

    const formData = new FormData();
    // Backend expects field name "document" (UploadFile param in /upload-document)
    formData.append("document", selectedFile);

    try {
      const res = await fetch("http://localhost:8000/upload-document", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        setUploadStatus("Upload failed");
        return;
      }

      const data = await res.json();
      if (data.document_id) setDocumentId(data.document_id);
      setUploadStatus(`Uploaded: ${selectedFile.name}`);
    } catch {
      setUploadStatus("Upload failed");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Document Assistant</h1>
        <p>Ask questions about your uploaded documents</p>
      </header>

      <section className="upload">
        <div className="upload-row">
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
            aria-label="PDF file"
          />
          <button
            type="button"
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
          >
            Upload
          </button>
        </div>
        {uploadStatus ? <p className="upload-status">{uploadStatus}</p> : null}
      </section>

      <main className="messages">
        {messages.length === 0 ? (
          <p className="empty">No messages yet. Ask a question below.</p>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
              <span className="label">{msg.role}</span>
              <p>
                {msg.text ||
                  (msg.role === "assistant" && streaming && i === messages.length - 1
                    ? "…"
                    : "")}
              </p>
              {msg.role === "assistant" &&
              Array.isArray(msg.sources) &&
              msg.sources.length > 0 ? (
                <ul className="sources">
                  {msg.sources.map((src, j) => {
                    const page =
                      src.startPage === src.endPage
                        ? `p. ${src.startPage}`
                        : `p. ${src.startPage}–${src.endPage}`;
                    return (
                      <li key={`${src.document_id ?? "doc"}-${src.chunkNumber ?? j}`}>
                        Chunk {src.chunkNumber} · {page}
                      </li>
                    );
                  })}
                </ul>
              ) : null}
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
          disabled={streaming}
        />
        <button type="submit" disabled={!input.trim() || streaming}>
          {streaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}

export default App;
