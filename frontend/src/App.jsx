import { useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const ACCEPTED_TYPES = ["image/jpeg", "image/png"];

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);

  function chooseFile(nextFile) {
    if (!nextFile) return;
    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setError("Please choose a JPG, JPEG, or PNG image.");
      return;
    }
    setError("");
    setResult(null);
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
  }

  function removeFile() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to analyze this image.");
      setResult(data);
    } catch (requestError) {
      setError(requestError.message || "Unable to reach the detection service.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    removeFile();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="app-shell">
      <nav className="navbar">
        <a className="brand" href="/">
          <span className="brand-mark">C</span>
          <span>Clone Catcher</span>
        </a>
        <span className="nav-note">Image authenticity screening</span>
      </nav>

      <main>
        {!result ? (
          <section className="hero">
            <p className="eyebrow">Experimental detection model</p>
            <h1>Is this image <em>real?</em></h1>
            <p className="intro">
              Upload a face image and let Clone Catcher check it against patterns learned by our
              Xception model.
            </p>

            <div
              className={`upload-card ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                chooseFile(event.dataTransfer.files[0]);
              }}
            >
              {file ? (
                <div className="file-preview">
                  <img src={preview} alt="Selected upload preview" />
                  <div className="file-details">
                    <strong>{file.name}</strong>
                    <span>{formatBytes(file.size)} · {file.type.split("/")[1].toUpperCase()}</span>
                    <div className="file-actions">
                      <button className="text-button" onClick={() => inputRef.current?.click()}>
                        Replace image
                      </button>
                      <button className="text-button muted" onClick={removeFile}>Remove</button>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="upload-icon">↑</div>
                  <h2>Drop an image here</h2>
                  <p>or choose a file from your device</p>
                  <button className="browse-button" onClick={() => inputRef.current?.click()}>
                    Browse image
                  </button>
                  <small>JPG, JPEG, or PNG · up to 10 MB</small>
                </>
              )}
              <input
                ref={inputRef}
                type="file"
                accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                onChange={(event) => chooseFile(event.target.files[0])}
                hidden
              />
            </div>

            {file && (
              <button className="analyze-button" onClick={analyze} disabled={loading}>
                {loading ? <><span className="spinner" />Analyzing image...</> : "Analyze image"}
              </button>
            )}
            {error && <p className="error-message" role="alert">{error}</p>}
            <p className="disclaimer">
              Clone Catcher is experimental and does not provide definitive deepfake verification.
            </p>
          </section>
        ) : (
          <section className="results-section">
            <p className="eyebrow">Analysis complete</p>
            <h1>This image appears <em>{result.prediction === "REAL" ? "real" : "synthetic"}.</em></h1>
            <div className={`result-panel ${result.prediction.toLowerCase()}`}>
              <div className="result-badge">{result.prediction === "REAL" ? "✓" : "!"}</div>
              <div>
                <span className="result-label">Prediction</span>
                <strong>{result.prediction}</strong>
              </div>
              <div className="model-score">
                <span>Model score</span>
                <strong>{result.model_score.toFixed(2)}%</strong>
              </div>
            </div>
            <div className="scores">
              <Score label="Real score" value={result.real_score} tone="real" />
              <Score label="Fake score" value={result.fake_score} tone="fake" />
            </div>
            <button className="analyze-button secondary" onClick={reset}>Analyze another image</button>
            <p className="disclaimer">
              This result is an experimental model estimate, not definitive proof of image authenticity.
            </p>
          </section>
        )}
      </main>
      <footer>Clone Catcher <span>·</span> Experimental Xception model</footer>
    </div>
  );
}

function Score({ label, value, tone }) {
  return (
    <div className="score-card">
      <div className="score-heading"><span>{label}</span><strong>{value.toFixed(2)}%</strong></div>
      <div className="score-track"><div className={`score-fill ${tone}`} style={{ width: `${value}%` }} /></div>
    </div>
  );
}
