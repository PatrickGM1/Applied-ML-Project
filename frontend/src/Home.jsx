import { Link } from "react-router-dom"

const ClockIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 8 12 12 14 14" />
  </svg>
)

const DocsIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
)

const PlayIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="5 3 19 12 5 21 5 3" />
  </svg>
)

export default function Home() {
  return (
    <div className="container">
      <header className="hero">
        <div className="hero-domain">
          <a href="https://aml.guba.dev">
            aml<span className="hero-slash">.</span>guba<span className="hero-slash">.</span>dev
          </a>
        </div>
        <p>Applied Machine Learning project</p>
        <p className="hero-sub">
          Fake News Detection &mdash;{" "}
          <a href="https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset" target="_blank" rel="noopener noreferrer">
            LIAR Dataset
          </a>
        </p>
      </header>

      <section>
        <span className="section-label">API</span>
        <div className="links">
          <a href="/health" target="_blank" rel="noopener noreferrer" className="link-btn">
            <ClockIcon />
            Test endpoint
          </a>
          <a href="/docs" target="_blank" rel="noopener noreferrer" className="link-btn">
            <DocsIcon />
            Swagger UI
          </a>
        </div>
      </section>

      <section>
        <span className="section-label">Demo</span>
        <div className="links">
          <Link to="/demo" className="link-btn">
            <PlayIcon />
            Test base model
          </Link>
        </div>
      </section>

      <footer>
        &copy; 2026 <a href="https://guba.dev">guba.dev</a>
      </footer>
    </div>
  )
}
