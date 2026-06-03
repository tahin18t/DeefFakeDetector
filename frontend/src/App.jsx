import { useState, useRef, useCallback } from "react";

const API_URL = "https://deeffakedetector.onrender.com";

const formatPercent = (v) => `${Number(v).toFixed(1)}%`;

function UploadZone({ onFile, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) onFile(file);
  }, [onFile]);

  return (
    <div
      onClick={() => !disabled && inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${dragging ? "#00ff88" : "#2a2a2a"}`,
        borderRadius: "12px",
        padding: "48px 24px",
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "all 0.2s",
        background: dragging ? "rgba(0,255,136,0.04)" : "rgba(255,255,255,0.02)",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
      <div style={{ fontSize: "40px", marginBottom: "12px" }}>⬆</div>
      <div style={{ color: "#888", fontFamily: "'Courier New', monospace", fontSize: "13px", letterSpacing: "0.08em" }}>
        DROP IMAGE HERE OR CLICK TO BROWSE
      </div>
      <div style={{ color: "#444", fontFamily: "'Courier New', monospace", fontSize: "11px", marginTop: "8px" }}>
        JPG · PNG · WEBP
      </div>
    </div>
  );
}

function ProbBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: "16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
        <span style={{ fontFamily: "'Courier New', monospace", fontSize: "12px", color: "#888", letterSpacing: "0.1em" }}>
          {label}
        </span>
        <span style={{ fontFamily: "'Courier New', monospace", fontSize: "13px", color, fontWeight: "bold" }}>
          {formatPercent(value)}
        </span>
      </div>
      <div style={{ height: "6px", background: "#1a1a1a", borderRadius: "3px", overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${value}%`,
          background: color,
          borderRadius: "3px",
          transition: "width 1s cubic-bezier(0.4,0,0.2,1)",
          boxShadow: `0 0 10px ${color}55`
        }} />
      </div>
    </div>
  );
}

function ImagePanel({ title, src, label }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{
        fontFamily: "'Courier New', monospace",
        fontSize: "10px",
        color: "#444",
        letterSpacing: "0.15em",
        marginBottom: "8px",
        textTransform: "uppercase"
      }}>
        {title}
      </div>
      <div style={{
        borderRadius: "8px",
        overflow: "hidden",
        border: "1px solid #1e1e1e",
        background: "#0d0d0d",
        aspectRatio: "1",
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      }}>
        {src
          ? <img src={`data:image/png;base64,${src}`} alt={title} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          : <div style={{ color: "#222", fontFamily: "monospace", fontSize: "11px" }}>—</div>
        }
      </div>
      {label && (
        <div style={{ fontFamily: "'Courier New', monospace", fontSize: "10px", color: "#555", marginTop: "6px", textAlign: "center" }}>
          {label}
        </div>
      )}
    </div>
  );
}

function Verdict({ label, confidence }) {
  const isFake = label?.toLowerCase() === "fake";
  const color = isFake ? "#ff4444" : "#00ff88";
  const bg = isFake ? "rgba(255,68,68,0.08)" : "rgba(0,255,136,0.08)";

  return (
    <div style={{
      border: `1px solid ${color}33`,
      borderRadius: "10px",
      padding: "20px 24px",
      background: bg,
      textAlign: "center",
      marginBottom: "24px",
    }}>
      <div style={{
        fontFamily: "'Courier New', monospace",
        fontSize: "10px",
        color: "#555",
        letterSpacing: "0.2em",
        marginBottom: "8px"
      }}>
        ANALYSIS RESULT
      </div>
      <div style={{
        fontFamily: "'Courier New', monospace",
        fontSize: "32px",
        fontWeight: "900",
        color,
        letterSpacing: "0.15em",
        textShadow: `0 0 30px ${color}44`,
        lineHeight: 1
      }}>
        {label?.toUpperCase()}
      </div>
      <div style={{
        fontFamily: "'Courier New', monospace",
        fontSize: "12px",
        color: "#666",
        marginTop: "8px"
      }}>
        {formatPercent(confidence)} confidence
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeView, setActiveView] = useState("overlay");

  const handleFile = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "Server error");
      setResult(data);
      setActiveView("overlay");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const viewSrc = result
    ? (activeView === "overlay" ? result.overlay_b64 : activeView === "heatmap" ? result.heatmap_b64 : result.original_b64)
    : null;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#080808",
      color: "#e0e0e0",
      fontFamily: "'Courier New', monospace",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: "1px solid #141414",
        padding: "20px 32px",
        display: "flex",
        alignItems: "center",
        gap: "16px",
      }}>
        <div style={{
          width: "8px", height: "8px",
          borderRadius: "50%",
          background: "#00ff88",
          boxShadow: "0 0 8px #00ff88"
        }} />
        <div style={{ fontSize: "13px", letterSpacing: "0.25em", color: "#666" }}>
          DEEPFAKE DETECTION SYSTEM
        </div>
        <div style={{ marginLeft: "auto", fontSize: "10px", color: "#333", letterSpacing: "0.1em" }}>
          MODEL: RESNET50 · GRAD-CAM
        </div>
      </div>

      {/* Main */}
      <div style={{
        maxWidth: "1100px",
        margin: "0 auto",
        padding: "48px 32px",
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "40px",
        alignItems: "start",
      }}>

        {/* LEFT PANEL */}
        <div>
          <div style={{ fontSize: "10px", color: "#444", letterSpacing: "0.2em", marginBottom: "16px" }}>
            INPUT
          </div>

          {!preview ? (
            <UploadZone onFile={handleFile} disabled={loading} />
          ) : (
            <div>
              <div style={{
                borderRadius: "10px",
                overflow: "hidden",
                border: "1px solid #1e1e1e",
                marginBottom: "16px",
                aspectRatio: "1",
                background: "#0d0d0d"
              }}>
                <img src={preview} alt="preview" style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
              </div>
              <div style={{ fontSize: "11px", color: "#444", marginBottom: "16px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {file?.name}
              </div>
            </div>
          )}

          <div style={{ display: "flex", gap: "10px", marginTop: "16px" }}>
            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              style={{
                flex: 1,
                padding: "13px",
                background: (!file || loading) ? "#141414" : "#00ff88",
                color: (!file || loading) ? "#333" : "#000",
                border: "none",
                borderRadius: "8px",
                fontFamily: "'Courier New', monospace",
                fontSize: "12px",
                letterSpacing: "0.15em",
                fontWeight: "bold",
                cursor: (!file || loading) ? "not-allowed" : "pointer",
                transition: "all 0.2s",
              }}
            >
              {loading ? "ANALYZING..." : "ANALYZE IMAGE"}
            </button>
            {(file || result) && (
              <button
                onClick={handleReset}
                disabled={loading}
                style={{
                  padding: "13px 18px",
                  background: "transparent",
                  color: "#555",
                  border: "1px solid #222",
                  borderRadius: "8px",
                  fontFamily: "'Courier New', monospace",
                  fontSize: "12px",
                  letterSpacing: "0.1em",
                  cursor: "pointer",
                }}
              >
                RESET
              </button>
            )}
          </div>

          {error && (
            <div style={{
              marginTop: "16px",
              padding: "12px 16px",
              background: "rgba(255,68,68,0.08)",
              border: "1px solid #ff444433",
              borderRadius: "8px",
              color: "#ff6666",
              fontSize: "12px",
            }}>
              ERROR: {error}
            </div>
          )}

          {/* Loading indicator */}
          {loading && (
            <div style={{ marginTop: "20px", textAlign: "center" }}>
              <div style={{ fontSize: "11px", color: "#444", letterSpacing: "0.15em", animation: "pulse 1.5s infinite" }}>
                RUNNING INFERENCE...
              </div>
              <style>{`@keyframes pulse { 0%,100%{opacity:0.3} 50%{opacity:1} }`}</style>
            </div>
          )}
        </div>

        {/* RIGHT PANEL */}
        <div>
          <div style={{ fontSize: "10px", color: "#444", letterSpacing: "0.2em", marginBottom: "16px" }}>
            ANALYSIS
          </div>

          {!result ? (
            <div style={{
              border: "1px solid #141414",
              borderRadius: "12px",
              padding: "64px 24px",
              textAlign: "center",
              color: "#2a2a2a",
              fontSize: "12px",
              letterSpacing: "0.15em"
            }}>
              AWAITING INPUT
            </div>
          ) : (
            <div>
              {/* Verdict */}
              <Verdict label={result.label} confidence={result.confidence} />

              {/* Probabilities */}
              <div style={{
                border: "1px solid #141414",
                borderRadius: "10px",
                padding: "20px",
                marginBottom: "24px",
                background: "rgba(255,255,255,0.01)"
              }}>
                <div style={{ fontSize: "10px", color: "#444", letterSpacing: "0.15em", marginBottom: "16px" }}>
                  PROBABILITY BREAKDOWN
                </div>
                <ProbBar label="FAKE" value={result.fake_prob} color="#ff4444" />
                <ProbBar label="REAL" value={result.real_prob} color="#00ff88" />
              </div>

              {/* View toggle */}
              <div style={{ display: "flex", gap: "6px", marginBottom: "12px" }}>
                {["overlay", "heatmap", "original"].map((v) => (
                  <button
                    key={v}
                    onClick={() => setActiveView(v)}
                    style={{
                      flex: 1,
                      padding: "8px",
                      background: activeView === v ? "#1a1a1a" : "transparent",
                      color: activeView === v ? "#e0e0e0" : "#444",
                      border: `1px solid ${activeView === v ? "#2a2a2a" : "#141414"}`,
                      borderRadius: "6px",
                      fontFamily: "'Courier New', monospace",
                      fontSize: "10px",
                      letterSpacing: "0.1em",
                      cursor: "pointer",
                      transition: "all 0.15s",
                    }}
                  >
                    {v.toUpperCase()}
                  </button>
                ))}
              </div>

              {/* Image viewer */}
              <div style={{
                borderRadius: "10px",
                overflow: "hidden",
                border: "1px solid #1e1e1e",
                aspectRatio: "1",
                background: "#0d0d0d"
              }}>
                {viewSrc && (
                  <img
                    src={`data:image/png;base64,${viewSrc}`}
                    alt={activeView}
                    style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
                  />
                )}
              </div>
              <div style={{ fontSize: "10px", color: "#333", marginTop: "8px", textAlign: "center", letterSpacing: "0.1em" }}>
                {activeView === "overlay" && "GRAD-CAM OVERLAY — RED/YELLOW = HIGH ATTENTION REGIONS"}
                {activeView === "heatmap" && "RAW GRAD-CAM HEATMAP — MODEL ATTENTION VISUALIZATION"}
                {activeView === "original" && "ORIGINAL INPUT IMAGE — NORMALIZED FOR DISPLAY"}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}