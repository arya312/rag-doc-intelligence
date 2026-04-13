import { useState } from "react";

const API = "";

type Message = {
  role: "user" | "assistant";
  text: string;
  pages?: number[];
};

export default function App() {
  const [collections, setCollections] = useState<string[]>([]);
  const [selectedCollection, setSelectedCollection] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  const loadCollections = async () => {
    try {
      const res = await fetch(`${API}/collections`);
      const data = await res.json();
      setCollections(Array.isArray(data.collections) ? data.collections : []);
    } catch {
      setCollections([]);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(`Ingesting ${file.name}...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.collection_name) {
        setUploadStatus(`Done! "${data.collection_name}" is ready.`);
        setSelectedCollection(data.collection_name);
        await loadCollections();
      } else {
        setUploadStatus(`Error: ${data.detail || "Unknown error"}`);
      }
    } catch {
      setUploadStatus("Upload failed. Is the backend running?");
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || !selectedCollection) return;

    const userMessage: Message = { role: "user", text: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setAsking(true);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage.text,
          collection_name: selectedCollection,
        }),
      });
      const data = await res.json();
      const assistantMessage: Message = {
        role: "assistant",
        text: data.answer,
        pages: data.pages,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error: could not reach the backend." },
      ]);
    } finally {
      setAsking(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: "2rem 1rem", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 4 }}>RAG Doc Intelligence</h1>
      <p style={{ color: "#666", marginBottom: 32 }}>Upload a PDF and ask it anything.</p>

      <div style={{ background: "#f8f8f8", borderRadius: 12, padding: "1.5rem", marginBottom: 24 }}>
        <p style={{ fontWeight: 500, marginBottom: 12 }}>1. Upload a PDF</p>
        <input type="file" accept=".pdf" onChange={handleUpload} disabled={uploading} style={{ marginBottom: 8 }} />
        {uploadStatus && (
          <p style={{ fontSize: 13, color: uploading ? "#888" : uploadStatus.startsWith("Error") ? "#cc0000" : "#2a7a2a", marginTop: 8 }}>
            {uploadStatus}
          </p>
        )}
      </div>

      <div style={{ background: "#f8f8f8", borderRadius: 12, padding: "1.5rem", marginBottom: 24 }}>
        <p style={{ fontWeight: 500, marginBottom: 12 }}>2. Select a document to query</p>
        <div style={{ display: "flex", gap: 8 }}>
          <select
            value={selectedCollection}
            onChange={(e) => setSelectedCollection(e.target.value)}
            style={{ flex: 1, padding: "8px 12px", borderRadius: 8, border: "1px solid #ddd", fontSize: 14 }}
          >
            <option value="">-- choose a document --</option>
            {collections.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <button
            onClick={loadCollections}
            style={{ padding: "8px 16px", borderRadius: 8, border: "1px solid #ddd", background: "white", cursor: "pointer", fontSize: 13 }}
          >
            Refresh
          </button>
        </div>
      </div>

      <div style={{ background: "#f8f8f8", borderRadius: 12, padding: "1.5rem", marginBottom: 16 }}>
        <p style={{ fontWeight: 500, marginBottom: 16 }}>3. Ask a question</p>
        <div style={{ minHeight: 200, marginBottom: 16 }}>
          {messages.length === 0 && (
            <p style={{ color: "#aaa", fontSize: 14 }}>Your conversation will appear here...</p>
          )}
          {messages.map((msg, i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <div style={{
                display: "inline-block",
                background: msg.role === "user" ? "#0066cc" : "white",
                color: msg.role === "user" ? "white" : "#333",
                padding: "10px 14px",
                borderRadius: msg.role === "user" ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
                maxWidth: "85%",
                fontSize: 14,
                lineHeight: 1.6,
                border: msg.role === "assistant" ? "1px solid #eee" : "none",
              }}>
                {msg.text}
              </div>
              {msg.pages && msg.pages.length > 0 && (
                <p style={{ fontSize: 11, color: "#999", marginTop: 4, marginLeft: 4 }}>
                  Retrieved from pages: {Array.from(new Set(msg.pages)).join(", ")}
                </p>
              )}
            </div>
          ))}
          {asking && <div style={{ color: "#888", fontSize: 13 }}>Claude is thinking...</div>}
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            placeholder={selectedCollection ? "Ask a question..." : "Select a document first"}
            disabled={!selectedCollection || asking}
            style={{ flex: 1, padding: "10px 14px", borderRadius: 8, border: "1px solid #ddd", fontSize: 14, outline: "none" }}
          />
          <button
            onClick={handleAsk}
            disabled={!selectedCollection || !question.trim() || asking}
            style={{
              padding: "10px 20px", borderRadius: 8, border: "none",
              background: "#0066cc", color: "white", cursor: "pointer",
              fontSize: 14, fontWeight: 500,
              opacity: (!selectedCollection || !question.trim() || asking) ? 0.5 : 1
            }}
          >
            Ask
          </button>
        </div>
      </div>

      <p style={{ fontSize: 12, color: "#bbb", textAlign: "center" }}>
        Built with FastAPI + ChromaDB + Claude — github.com/arya312/rag-doc-intelligence
      </p>
    </div>
  );
}
