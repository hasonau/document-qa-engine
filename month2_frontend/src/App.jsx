import { useCallback, useEffect, useRef, useState } from "react";
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
    if (data.detail && typeof data.detail === "object") {
      const d = data.detail;
      if (typeof d.message === "string") {
        const failed = d.failed ? ` (failed at ${d.failed})` : "";
        return `${d.message}${failed}`;
      }
      return JSON.stringify(d);
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

function formatUploadedAt(iso) {
  if (!iso) return "Unknown upload time";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
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

  const [documents, setDocuments] = useState([]);
  const [kbLoading, setKbLoading] = useState(false);
  const [kbError, setKbError] = useState("");
  const [kbBusyId, setKbBusyId] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  const reuploadInputRef = useRef(null);
  const reuploadTargetIdRef = useRef(null);

  const busy = uploading || streaming || kbBusyId !== null;

  const refreshDocuments = useCallback(async () => {
    setKbLoading(true);
    setKbError("");
    try {
      const res = await fetch(`${API_BASE}/documents`, {
        method: "GET",
        credentials: "include",
      });

      if (res.status === 401) {
        setDocuments([]);
        setKbError("No session yet. Upload a document to start.");
        return;
      }

      if (!res.ok) {
        const detail = await readErrorDetail(
          res,
          `Could not load documents (${res.status})`,
        );
        setKbError(detail);
        return;
      }

      const data = await res.json();
      setDocuments(Array.isArray(data.documents) ? data.documents : []);
    } catch {
      setKbError("Could not reach the server. Is the API running?");
    } finally {
      setKbLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

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

  async function uploadFile(file) {
    const formData = new FormData();
    // Backend expects field name "document" (UploadFile param in /upload-document)
    formData.append("document", file);

    const res = await fetch(`${API_BASE}/upload-document`, {
      method: "POST",
      credentials: "include",
      body: formData,
    });

    if (!res.ok) {
      const detail = await readErrorDetail(res, `Upload failed (${res.status})`);
      throw new Error(detail);
    }

    const data = await res.json();
    return data;
  }

  async function handleUpload() {
    if (!selectedFile || uploading) return;

    setUploading(true);
    setUploadError(false);
    setUploadStatus("Uploading…");

    try {
      const data = await uploadFile(selectedFile);
      if (data.document_id) setDocumentId(data.document_id);
      setUploadError(false);
      setUploadStatus(`Uploaded: ${selectedFile.name}`);
      await refreshDocuments();
    } catch (err) {
      setUploadError(true);
      setUploadStatus(
        err instanceof Error
          ? err.message
          : "Could not reach the server. Is the API running?",
      );
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(targetId) {
    const res = await fetch(
      `${API_BASE}/documents/${encodeURIComponent(targetId)}`,
      {
        method: "DELETE",
        credentials: "include",
      },
    );

    if (res.status === 404) {
      // Already gone — refresh list; clear active if it was this id
      if (documentId === targetId) setDocumentId(null);
      await refreshDocuments();
      return { alreadyGone: true };
    }

    if (res.status === 401) {
      throw new Error("No session. Upload a document to start.");
    }

    if (!res.ok) {
      const detail = await readErrorDetail(
        res,
        `Delete failed (${res.status})`,
      );
      throw new Error(detail);
    }

    if (documentId === targetId) setDocumentId(null);
    await refreshDocuments();
    return { alreadyGone: false };
  }

  async function handleConfirmDelete() {
    const targetId = confirmDeleteId;
    if (!targetId || busy) return;

    setConfirmDeleteId(null);
    setKbBusyId(targetId);
    setKbError("");

    try {
      await deleteDocument(targetId);
    } catch (err) {
      setKbError(
        err instanceof Error ? err.message : "Delete failed. Try again.",
      );
      await refreshDocuments();
    } finally {
      setKbBusyId(null);
    }
  }

  function handleReuploadClick(targetId) {
    if (busy) return;
    reuploadTargetIdRef.current = targetId;
    if (reuploadInputRef.current) {
      reuploadInputRef.current.value = "";
      reuploadInputRef.current.click();
    }
  }

  async function handleReuploadFileChange(e) {
    const file = e.target.files?.[0] ?? null;
    const targetId = reuploadTargetIdRef.current;
    e.target.value = "";
    reuploadTargetIdRef.current = null;
    if (!file || !targetId || busy) return;

    setKbBusyId(targetId);
    setKbError("");
    setUploadError(false);
    setUploadStatus(`Re-uploading over ${file.name}…`);

    try {
      await deleteDocument(targetId);
      const data = await uploadFile(file);
      if (data.document_id) setDocumentId(data.document_id);
      setUploadStatus(`Re-uploaded: ${file.name}`);
      await refreshDocuments();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Re-upload failed. Try again.";
      setKbError(message);
      setUploadError(true);
      setUploadStatus(message);
      await refreshDocuments();
    } finally {
      setKbBusyId(null);
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
            disabled={busy}
          />
          <button
            type="button"
            onClick={handleUpload}
            disabled={!selectedFile || busy}
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
                disabled={busy}
              >
                Retry upload
              </button>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="knowledge-base" aria-label="Knowledge base">
        <div className="kb-header">
          <h2>Knowledge base</h2>
          <button
            type="button"
            className="retry-btn"
            onClick={refreshDocuments}
            disabled={busy || kbLoading}
          >
            {kbLoading ? "Refreshing…" : "Refresh"}
          </button>
        </div>

        <input
          ref={reuploadInputRef}
          type="file"
          accept=".pdf"
          className="kb-hidden-input"
          aria-hidden="true"
          tabIndex={-1}
          onChange={handleReuploadFileChange}
        />

        {kbError ? <p className="kb-error">{kbError}</p> : null}

        {kbLoading && documents.length === 0 ? (
          <p className="kb-empty">Loading documents…</p>
        ) : null}

        {!kbLoading && documents.length === 0 && !kbError ? (
          <p className="kb-empty">No documents yet. Upload a PDF above.</p>
        ) : null}

        {documents.length > 0 ? (
          <ul className="kb-list">
            {documents.map((doc) => {
              const isActive = documentId === doc.document_id;
              const rowBusy = kbBusyId === doc.document_id;
              const confirming = confirmDeleteId === doc.document_id;
              return (
                <li
                  key={doc.document_id}
                  className={`kb-row${isActive ? " is-active" : ""}${rowBusy ? " is-busy" : ""}`}
                >
                  <button
                    type="button"
                    className="kb-select"
                    onClick={() => setDocumentId(doc.document_id)}
                    disabled={busy}
                    aria-pressed={isActive}
                  >
                    <span className="kb-filename">
                      {doc.filename || "Unnamed document"}
                    </span>
                    <span className="kb-meta">
                      {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
                      {" · "}
                      {formatUploadedAt(doc.uploaded_at)}
                      {isActive ? " · Active" : ""}
                    </span>
                  </button>

                  <div className="kb-actions">
                    {confirming ? (
                      <>
                        <button
                          type="button"
                          className="kb-danger"
                          onClick={handleConfirmDelete}
                          disabled={busy}
                        >
                          Confirm delete
                        </button>
                        <button
                          type="button"
                          className="retry-btn"
                          onClick={() => setConfirmDeleteId(null)}
                          disabled={busy}
                        >
                          Cancel
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          type="button"
                          className="retry-btn"
                          onClick={() => setConfirmDeleteId(doc.document_id)}
                          disabled={busy}
                        >
                          {rowBusy ? "Working…" : "Delete"}
                        </button>
                        <button
                          type="button"
                          className="retry-btn"
                          onClick={() => handleReuploadClick(doc.document_id)}
                          disabled={busy}
                        >
                          Re-upload
                        </button>
                      </>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
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
                  disabled={busy}
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
          placeholder={
            !documentId
              ? "Select a document above, then ask…"
              : streaming
                ? "Waiting for answer…"
                : "Ask a question..."
          }
          aria-label="Message"
          disabled={busy}
        />
        <button type="submit" disabled={!input.trim() || busy || !documentId}>
          {streaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}

export default App;
