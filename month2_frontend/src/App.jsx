import { useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

function updateLastAssistant(prev, patch) {
  const next = [...prev];
  const last = next[next.length - 1];
  if (!last || last.role !== "assistant") return prev;
  next[next.length - 1] = { ...last, ...patch(last) };
  return next;
}

async function readErrorDetail(res, fallback) {
  try {
    const data = await res.json();
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail;
    }
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      return data.detail
        .map((item) => item.msg ?? JSON.stringify(item))
        .join("; ");
    }
  } catch {
    // non-JSON body
  }
  return fallback;
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
  const [uploadError, setUploadError] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documentId, setDocumentId] = useState(null);
  const [streaming, setStreaming] = useState(false);

  async function runQuery(question) {
    const text = question.trim();
    if (!text || streaming) return;

    setStreaming(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        credentials: "include",
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
        const detail = await readErrorDetail(
          res,
          `Request failed (${res.status}). Try again.`,
        );
        setMessages((prev) =>
          updateLastAssistant(prev, () => ({
            text: detail,
            sources: null,
            error: true,
            retryQuestion: text,
          })),
        );
        return;
      }

      await readQuerySse(res, {
        onAnswerToken(token) {
          setMessages((prev) =>
            updateLastAssistant(prev, (last) => ({
              text: `${last.text ?? ""}${token}`,
              error: false,
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
              error: false,
            })),
          );
        },
      });

      // If stream closed with no tokens and no not_found text, show a fallback
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && !last.text) {
          return updateLastAssistant(prev, () => ({
            text: "No answer came back. Try again.",
            error: true,
            retryQuestion: text,
          }));
        }
        return prev;
      });
    } catch {
      setMessages((prev) =>
        updateLastAssistant(prev, (last) => ({
          text: last.text
            ? `${last.text}\n\nConnection interrupted. You can retry.`
            : "Could not reach the server. Is the API running?",
          error: true,
          retryQuestion: text,
        })),
      );
    } finally {
      setStreaming(false);
    }
  }

  async function handleSend(e) {
    e.preventDefault();

    const text = input.trim();
    if (!text || streaming) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text },
      { role: "assistant", text: "", sources: null, error: false },
    ]);
    setInput("");
    await runQuery(text);
  }

  async function handleRetryMessage(assistantIndex) {
    if (streaming) return;
    const assistant = messages[assistantIndex];
    const question = assistant?.retryQuestion?.trim();
    if (!question) return;

    setMessages((prev) => {
      const next = [...prev];
      next[assistantIndex] = {
        role: "assistant",
        text: "",
        sources: null,
        error: false,
      };
      return next;
    });
    await runQuery(question);
  }

  async function handleUpload() {
    if (!selectedFile || uploading) return;

    setUploading(true);
    setUploadError(false);
    setUploadStatus("Uploading…");

    const formData = new FormData();
    // Backend expects field name "document" (UploadFile param in /upload-document)
    formData.append("document", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/upload-document`, {
        method: "POST",
        credentials: "include",
        body: formData,
      });

      if (!res.ok) {
        const detail = await readErrorDetail(res, `Upload failed (${res.status})`);
        setUploadError(true);
        setUploadStatus(detail);
        return;
      }

      const data = await res.json();
      if (data.document_id) setDocumentId(data.document_id);
      setUploadError(false);
      setUploadStatus(`Uploaded: ${selectedFile.name}`);
    } catch {
      setUploadError(true);
      setUploadStatus("Could not reach the server. Is the API running?");
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
            onChange={(e) => {
              setSelectedFile(e.target.files?.[0] ?? null);
              setUploadError(false);
              setUploadStatus("");
            }}
            aria-label="PDF file"
            disabled={uploading}
          />
          <button
            type="button"
            onClick={handleUpload}
            disabled={!selectedFile || uploading || streaming}
          >
            {uploading ? "Uploading…" : "Upload"}
          </button>
        </div>
        {uploadStatus ? (
          <div className={`upload-status-row${uploadError ? " is-error" : ""}${uploading ? " is-loading" : ""}`}>
            <p className="upload-status">{uploadStatus}</p>
            {uploadError && selectedFile ? (
              <button
                type="button"
                className="retry-btn"
                onClick={handleUpload}
                disabled={uploading || streaming}
              >
                Retry upload
              </button>
            ) : null}
          </div>
        ) : null}
      </section>

      <main className="messages">
        {messages.length === 0 ? (
          <p className="empty">No messages yet. Ask a question below.</p>
        ) : (
          messages.map((msg, i) => (
            <div
              key={i}
              className={`bubble ${msg.role}${msg.error ? " error" : ""}`}
            >
              <span className="label">{msg.role}</span>
              <p>
                {msg.text ||
                  (msg.role === "assistant" && streaming && i === messages.length - 1
                    ? "Thinking…"
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
              {msg.role === "assistant" && msg.error && msg.retryQuestion ? (
                <button
                  type="button"
                  className="retry-btn"
                  onClick={() => handleRetryMessage(i)}
                  disabled={streaming || uploading}
                >
                  Retry
                </button>
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
          placeholder={streaming ? "Waiting for answer…" : "Ask a question..."}
          aria-label="Message"
          disabled={streaming || uploading}
        />
        <button type="submit" disabled={!input.trim() || streaming || uploading}>
          {streaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}

export default App;
