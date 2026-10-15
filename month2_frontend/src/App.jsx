import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { normalizeCitations } from "./utils/citations.js";
import { formatAnswer, isDontKnow } from "./utils/formatAnswer.js";

const API_BASE = "http://localhost:8000";
const IDK_HINT =
  "Try asking about a specific topic or detail in the document.";

/** Suggested questions for the bundled sample PDF (and useful after any load). */
const SAMPLE_QUESTIONS = [
  "What is the embedding store demo codeword in the indexing pipeline?",
  "How many candidate passages does this guide keep after hybrid merge before reranking?",
  "What temperature does the grounded answer step use in this guide?",
];

function updateLastAssistant(prev, patch) {
  const next = [...prev];
  const last = next[next.length - 1];
  if (!last || last.role !== "assistant") return prev;
  next[next.length - 1] = { ...last, ...patch(last) };
  return next;
}

/** Map common HTTP statuses to short UI copy when the body is empty/unhelpful. */
function friendlyStatusMessage(status, context) {
  const forUpload = context === "upload";
  switch (status) {
    case 413:
      return forUpload
        ? "That file is too large. Please upload a PDF under 10 MB."
        : "That request is too large.";
    case 415:
      return "Only PDF files are supported.";
    case 422:
      return forUpload
        ? "This PDF cannot be processed (no extractable text, too many pages, or document limit reached)."
        : "That question could not be processed. Check length and try again.";
    case 429:
      return "Too many requests. Please wait a moment and try again.";
    case 503:
      return "The language model is temporarily unavailable. Please try again.";
    default:
      return null;
  }
}

async function readErrorDetail(res, fallback, context) {
  const statusFriendly = friendlyStatusMessage(res.status, context);
  try {
    const data = await res.json();
    if (typeof data.detail === "string" && data.detail.trim()) {
      // Prefer server message (includes 10-doc limit, scanned PDF, etc.)
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
  return statusFriendly || fallback;
}

function formatUploadedAt(iso) {
  if (!iso) return "Unknown upload time";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

/** Parse SSE frames from a POST /query ReadableStream (cache-hit path). */
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

function renderSpans(spans) {
  return spans.map((span, i) =>
    span.type === "bold" ? (
      <strong key={i}>{span.text}</strong>
    ) : (
      <span key={i}>{span.text}</span>
    ),
  );
}

function FormattedAnswer({ text }) {
  const blocks = formatAnswer(text);
  if (blocks.length === 0) return null;

  return (
    <div className="answer-body">
      {blocks.map((block, i) => {
        if (block.type === "ul") {
          return (
            <ul key={i} className="answer-list">
              {block.items.map((item, j) => (
                <li key={j}>{renderSpans(item.spans)}</li>
              ))}
            </ul>
          );
        }
        if (block.type === "ol") {
          return (
            <ol key={i} className="answer-list answer-list-numbered">
              {block.items.map((item, j) => (
                <li key={j}>{renderSpans(item.spans)}</li>
              ))}
            </ol>
          );
        }
        return (
          <p key={i} className="answer-paragraph">
            {renderSpans(block.spans)}
          </p>
        );
      })}
    </div>
  );
}

function SourceItem({ source }) {
  const [expanded, setExpanded] = useState(false);
  const panelId = `source-passage-${source.key}`;

  return (
    <li className="source-item">
      <div className="source-head">
        <span className="source-heading">{source.heading}</span>
        {source.pageLabel ? (
          <span className="source-page">{source.pageLabel}</span>
        ) : null}
      </div>
      <p className="source-snippet">{source.snippet || "(empty chunk)"}</p>
      {source.fullText ? (
        <>
          <button
            type="button"
            className="source-toggle"
            aria-expanded={expanded}
            aria-controls={panelId}
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? "Hide passage" : "Show passage"}
          </button>
          {expanded ? (
            <pre id={panelId} className="source-passage">
              {source.fullText}
            </pre>
          ) : null}
        </>
      ) : null}
    </li>
  );
}

function SourcesBlock({ sources }) {
  if (!Array.isArray(sources)) return null;
  if (sources.length === 0) {
    return <p className="sources-empty">No sources cited</p>;
  }
  return (
    <div className="sources-block">
      <h3 className="sources-title">Sources</h3>
      <ul className="sources">
        {sources.map((src) => (
          <SourceItem key={src.key} source={src} />
        ))}
      </ul>
    </div>
  );
}

function CopyAnswerButton({ text, disabled }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    if (!text || disabled) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <button
      type="button"
      className="copy-btn"
      onClick={handleCopy}
      disabled={disabled || !text}
    >
      {copied ? "Copied" : "Copy answer"}
    </button>
  );
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
  const messagesEndRef = useRef(null);

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
        setDocumentId(null);
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
      const docs = Array.isArray(data.documents) ? data.documents : [];
      setDocuments(docs);
      // Clear active selection if that document is no longer in the list.
      setDocumentId((current) => {
        if (!current) return null;
        return docs.some((d) => d.document_id === current) ? current : null;
      });
    } catch {
      setKbError("Could not reach the server. Is the API running?");
    } finally {
      setKbLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, streaming]);

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
          Accept: "text/event-stream, application/json",
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
          "query",
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

      const contentType = (res.headers.get("content-type") || "").toLowerCase();

      // Cache miss: plain JSON { answer, citations }
      if (contentType.includes("application/json")) {
        const data = await res.json();
        const answer =
          typeof data.answer === "string" ? data.answer : "";
        const normalized = normalizeCitations(data.citations);
        setMessages((prev) =>
          updateLastAssistant(prev, () => ({
            text: answer || "No answer came back. Try again.",
            sources: normalized,
            error: !answer,
            retryQuestion: answer ? undefined : text,
          })),
        );
        return;
      }

      // Cache hit: SSE answer + sources events
      let sawSourcesEvent = false;
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
          sawSourcesEvent = true;
          setMessages((prev) =>
            updateLastAssistant(prev, () => ({
              sources: normalizeCitations(sources),
            })),
          );
        },
        onNotFound(message) {
          setMessages((prev) =>
            updateLastAssistant(prev, () => ({
              text: message,
              sources: [],
              error: false,
            })),
          );
        },
      });

      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && !last.text) {
          return updateLastAssistant(prev, () => ({
            text: "No answer came back. Try again.",
            error: true,
            retryQuestion: text,
            sources: null,
          }));
        }
        // Successful SSE with no sources event → treat as no citations
        if (
          last?.role === "assistant" &&
          !last.error &&
          !sawSourcesEvent &&
          last.sources == null
        ) {
          return updateLastAssistant(prev, () => ({
            sources: [],
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
    if (!text || streaming || !documentId) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text },
      { role: "assistant", text: "", sources: null, error: false },
    ]);
    setInput("");
    await runQuery(text);
  }

  function handleComposerKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!busy && documentId && input.trim()) {
        handleSend(e);
      }
    }
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
      const detail = await readErrorDetail(
        res,
        `Upload failed (${res.status})`,
        "upload",
      );
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

  async function handleLoadSample() {
    if (uploading || busy) return;

    setUploading(true);
    setUploadError(false);
    setUploadStatus("Loading sample document…");

    try {
      const res = await fetch(`${API_BASE}/load-sample`, {
        method: "POST",
        credentials: "include",
      });

      if (!res.ok) {
        const detail = await readErrorDetail(
          res,
          `Could not load sample (${res.status})`,
          "upload",
        );
        throw new Error(detail);
      }

      const data = await res.json();
      if (data.document_id) setDocumentId(data.document_id);
      setUploadError(false);
      const chunks =
        typeof data.chunks_count === "number"
          ? ` (${data.chunks_count} chunks)`
          : "";
      setUploadStatus(
        `Sample loaded: ${data.filename || "rag_how_it_works.pdf"}${chunks}`,
      );
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

  async function handleSuggestedQuestion(question) {
    if (!documentId || streaming || busy) return;
    const text = question.trim();
    if (!text) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text },
      { role: "assistant", text: "", sources: null, error: false },
    ]);
    setInput("");
    await runQuery(text);
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

  function emptyStateMessage() {
    if (!documentId) {
      return "Upload a PDF, load the sample document, or select one in the Knowledge Base to start";
    }
    return "Document selected. Ask a question below about this PDF.";
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
          <button
            type="button"
            className="sample-btn"
            onClick={handleLoadSample}
            disabled={busy}
          >
            Load sample document
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

      <main className="messages" aria-live="polite">
        {messages.length === 0 ? (
          <p className="empty">{emptyStateMessage()}</p>
        ) : (
          messages.map((msg, i) => {
            const isLast = i === messages.length - 1;
            const isLoadingAssistant =
              msg.role === "assistant" &&
              streaming &&
              isLast &&
              !msg.text &&
              !msg.error;
            const dontKnow =
              msg.role === "assistant" &&
              !msg.error &&
              !streaming &&
              isDontKnow(msg.text);

            return (
              <div
                key={i}
                className={`bubble ${msg.role}${msg.error ? " error" : ""}${dontKnow ? " idk" : ""}${isLoadingAssistant ? " is-loading" : ""}`}
              >
                <span className="label">{msg.role}</span>

                {isLoadingAssistant ? (
                  <p className="loading-state" role="status">
                    Generating answer…
                  </p>
                ) : msg.role === "assistant" && !msg.error ? (
                  dontKnow ? (
                    <div className="answer-idk">
                      <p className="answer-idk-text">{msg.text}</p>
                      <p className="answer-hint">{IDK_HINT}</p>
                    </div>
                  ) : (
                    <FormattedAnswer text={msg.text} />
                  )
                ) : (
                  <p className="answer-plain">{msg.text}</p>
                )}

                {msg.role === "assistant" &&
                !msg.error &&
                msg.text &&
                !(streaming && isLast) ? (
                  <div className="answer-actions">
                    <CopyAnswerButton text={msg.text} disabled={busy} />
                  </div>
                ) : null}

                {msg.role === "assistant" && !msg.error ? (
                  <SourcesBlock sources={msg.sources} />
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
            );
          })
        )}
        <div ref={messagesEndRef} />
      </main>

      {documentId ? (
        <section className="suggestions" aria-label="Suggested questions">
          <p className="suggestions-label">Suggested questions</p>
          <ul className="suggestions-list">
            {SAMPLE_QUESTIONS.map((q) => (
              <li key={q}>
                <button
                  type="button"
                  className="suggestion-btn"
                  onClick={() => handleSuggestedQuestion(q)}
                  disabled={busy}
                >
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <form className="composer" onSubmit={handleSend}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleComposerKeyDown}
          rows={2}
          placeholder={
            !documentId
              ? "Upload, load the sample, or select a document to start…"
              : streaming
                ? "Waiting for answer…"
                : "Ask a question… (Enter to send, Shift+Enter for newline)"
          }
          aria-label="Message"
          disabled={busy || !documentId}
        />
        <button type="submit" disabled={!input.trim() || busy || !documentId}>
          {streaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}

export default App;
