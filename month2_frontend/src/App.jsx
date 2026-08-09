import { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  const [documentId, setDocumentId] = useState(null);

  async function handleSend(e) {
    e.preventDefault();

    const text = input.trim();
    if (!text) return;

    const userMessage = { role: "user", text };
    const loadingMessage = { role: "assistant", text: "..." };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setInput("");

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          document_id: documentId ?? "",
        }),
      });

      if (!res.ok) {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            text: "Something went wrong, try again",
          };
          return next;
        });
        return;
      }

      const data = await res.json();
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "assistant",
          text: data.answer ?? "Something went wrong, try again",
        };
        return next;
      });
    } catch {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = {
          role: "assistant",
          text: "Something went wrong, try again",
        };
        return next;
      });
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
