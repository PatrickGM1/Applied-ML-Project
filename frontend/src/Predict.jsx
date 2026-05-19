import { useState } from "react"
import { Link } from "react-router-dom"

const DEFAULT_FORM = {
  statement: "",
  subjects: "",
  party: "",
  state: "",
  speaker_job: "",
  hist1: 0, hist2: 0, hist3: 0, hist4: 0, hist5: 0,
}

const HISTORY_FIELDS = [
  { name: "hist1", label: "Barely true" },
  { name: "hist2", label: "False" },
  { name: "hist3", label: "Half true" },
  { name: "hist4", label: "Mostly true" },
  { name: "hist5", label: "Pants on fire" },
]

function normalizeString(v) { return v && v.trim() ? v.trim() : "missing" }

function getApiBaseUrl() {
  const c = localStorage.getItem("API_BASE_URL")
  if (c && c.trim()) return c.trim().replace(/\/$/, "")
  return ""
}

export default function Predict() {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  function handleChange(e) {
    const { name, value, type } = e.target
    setForm(prev => ({ ...prev, [name]: type === "number" ? Number(value) : value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!form.statement.trim()) { setResult({ error: "Claim text is required." }); return }
    setLoading(true); setResult(null)
    const payload = {
      statement: form.statement.trim(),
      subjects: form.subjects.trim(),
      party: normalizeString(form.party),
      state: normalizeString(form.state),
      speaker_job: normalizeString(form.speaker_job),
      hist1: Number(form.hist1 || 0), hist2: Number(form.hist2 || 0),
      hist3: Number(form.hist3 || 0), hist4: Number(form.hist4 || 0),
      hist5: Number(form.hist5 || 0),
    }
    try {
      const res = await fetch(`${getApiBaseUrl()}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      const data = await res.json()
      if (!res.ok) { setResult({ error: data.detail || "Prediction failed." }); return }
      setResult({ data })
    } catch {
      setResult({ error: "Could not reach the API. Make sure the backend is running." })
    } finally { setLoading(false) }
  }

  function handleReset() { setForm(DEFAULT_FORM); setResult(null) }

  return (
    <div className="container">
      <header className="hero">
        <div className="hero-domain">
          <Link to="/">aml<span className="hero-slash">.</span>guba<span className="hero-slash">.</span>dev</Link>
        </div>
        <p>Base Model</p>
      </header>

      <section>
        <span className="section-label">Try Prediction</span>
        <form className="predict-form" onSubmit={handleSubmit}>
          <label htmlFor="statement">Claim text</label>
          <textarea id="statement" name="statement" rows={4} placeholder="Enter a political claim&hellip;"
            value={form.statement} onChange={handleChange} required />

          <div className="grid-2">
            {[["subjects","Subjects (comma-separated)","economy,budget","text"],
              ["party","Party","republican","text"],
              ["state","State","texas","text"],
              ["speaker_job","Speaker job","senator","text"]].map(([name,label,placeholder,type]) => (
              <div key={name}>
                <label htmlFor={name}>{label}</label>
                <input id={name} name={name} type={type} placeholder={placeholder}
                  value={form[name]} onChange={handleChange} />
              </div>
            ))}
          </div>

          <details>
            <summary>Speaker history (optional) — how many times has this speaker been rated each label?</summary>
            <div className="grid-5">
              {HISTORY_FIELDS.map(({ name, label }) => (
                <div key={name}>
                  <label htmlFor={name}>{label}</label>
                  <input id={name} name={name} type="number" min="0" step="1"
                    value={form[name]} onChange={handleChange} />
                </div>
              ))}
            </div>
          </details>

          <div className="form-actions">
            <button type="submit" disabled={loading}>{loading ? "Predicting\u2026" : "Predict"}</button>
            {(result || form.statement) && (
              <button type="button" className="btn-ghost" onClick={handleReset}>Clear</button>
            )}
          </div>
        </form>

        <PredictResult result={result} loading={loading} />
      </section>

      <footer>&copy; 2026 <a href="https://guba.dev">guba.dev</a></footer>
    </div>
  )
}

function PredictResult({ result, loading }) {
  if (loading) return (
    <div className="result result--loading">
      <span className="spinner" /><p className="muted">Running inference&hellip;</p>
    </div>
  )
  if (!result) return <div className="result"><p className="muted">Submit a claim to see model output.</p></div>
  if (result.error) return <div className="result"><p className="error">{result.error}</p></div>

  const { data } = result
  const probs = Object.entries(data.class_probabilities || {}).sort(([,a],[,b]) => b - a)
  const confPct = data.confidence ? (data.confidence * 100).toFixed(1) : null

  return (
    <div className="result result--success">
      <div className="result-header">
        <div className="result-meta">
          <span className="result-verdict">{data.predicted_label}</span>
          {confPct && <p className="muted result-confidence-text">Confidence: <strong style={{color:"var(--text)"}}>{confPct}%</strong></p>}
        </div>
        {data.confidence != null && <ConfidenceMeter value={data.confidence} />}
      </div>
      {data.is_low_confidence && (
        <p className="warn">Low confidence &mdash; treat this as a weak signal, not a verdict.</p>
      )}
      <div className="prob-list">
        {probs.map(([label, value]) => {
          const pct = (value * 100).toFixed(1)
          const isTop = label === data.predicted_label
          return (
            <div key={label} className={`prob-row${isTop ? " prob-row--top" : ""}`}>
              <span className="prob-label">{label}</span>
              <div className="prob-bar-track"><div className="prob-bar-fill" style={{width:`${pct}%`}} /></div>
              <span className="prob-value">{pct}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ConfidenceMeter({ value }) {
  const pct = value * 100
  const color = pct >= 75 ? "#4ade80" : pct >= 60 ? "#facc15" : "#f87171"
  return (
    <div className="confidence-meter" title={`${pct.toFixed(1)}% confidence`}>
      <svg viewBox="0 0 36 36" className="confidence-svg">
        <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--border)" strokeWidth="3" />
        <circle cx="18" cy="18" r="15.9" fill="none" stroke={color} strokeWidth="3"
          strokeDasharray={`${pct.toFixed(1)} 100`} strokeLinecap="round"
          transform="rotate(-90 18 18)" style={{transition:"stroke-dasharray 0.5s ease"}} />
      </svg>
      <span className="confidence-value" style={{color}}>{Math.round(pct)}%</span>
    </div>
  )
}
