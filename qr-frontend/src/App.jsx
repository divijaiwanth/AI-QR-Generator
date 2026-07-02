import { useState } from "react";

const API_URL = "http://localhost:8000";

export default function App() {
  const [url, setUrl] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function generate() {
    if (!url || !prompt) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          prompt,
          controlnet_conditioning_scale: 1.3,
          num_inference_steps: 30,
        }),
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>AI QR Generator</h1>
      <p style={styles.sub}>Stable Diffusion + ControlNet</p>

      <div style={styles.card}>
        <label style={styles.label}>URL to encode</label>
        <input
          style={styles.input}
          placeholder="https://example.com"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />

        <label style={styles.label}>Style prompt</label>
        <input
          style={styles.input}
          placeholder="japanese garden, koi pond, cherry blossoms..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <button
          style={{ ...styles.button, opacity: loading ? 0.6 : 1 }}
          onClick={generate}
          disabled={loading}
        >
          {loading ? "Generating (~30s)..." : "Generate"}
        </button>
      </div>

      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.card}>
          <img
            src={result.image_url}
            alt="Generated QR"
            style={styles.image}
          />
          <div style={styles.meta}>
            <span>⏱ {result.elapsed_seconds}s</span>
            {result.peak_vram_mb && (
              <span>🖥 {result.peak_vram_mb} MB VRAM</span>
            )}
            <a href={result.image_url} target="_blank" rel="noreferrer">
              Open full size ↗
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    background: "#0f0f0f",
    color: "#fff",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "48px 16px",
    fontFamily: "system-ui, sans-serif",
  },
  title: {
    fontSize: "2rem",
    fontWeight: 700,
    margin: 0,
  },
  sub: {
    color: "#888",
    marginTop: 8,
    marginBottom: 32,
  },
  card: {
    background: "#1a1a1a",
    border: "1px solid #2a2a2a",
    borderRadius: 12,
    padding: 24,
    width: "100%",
    maxWidth: 480,
    marginBottom: 24,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  label: {
    fontSize: 13,
    color: "#aaa",
    marginBottom: -4,
  },
  input: {
    background: "#111",
    border: "1px solid #333",
    borderRadius: 8,
    padding: "10px 14px",
    color: "#fff",
    fontSize: 14,
    outline: "none",
  },
  button: {
    background: "#6366f1",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    padding: "12px 0",
    fontSize: 15,
    fontWeight: 600,
    cursor: "pointer",
    marginTop: 4,
  },
  error: {
    color: "#f87171",
    fontSize: 14,
  },
  image: {
    width: "100%",
    borderRadius: 8,
  },
  meta: {
    display: "flex",
    gap: 16,
    fontSize: 13,
    color: "#aaa",
    flexWrap: "wrap",
  },
};