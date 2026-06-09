import streamlit as st

st.set_page_config(
    page_title="aml.guba.dev — Fake News Detection",
    page_icon="🔍",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0a0a0a;
        --surface: #141414;
        --surface-2: #1a1a1a;
        --border: #222;
        --border-hover: #7c6af7;
        --text: #e8e8e8;
        --text-dim: #aaa;
        --muted: #666;
        --accent: #7c6af7;
        --accent-dim: rgba(124, 106, 247, 0.10);
        --accent-mid: rgba(124, 106, 247, 0.20);
        --accent-glow: rgba(124, 106, 247, 0.35);
        --radius: 10px;
        --radius-sm: 6px;
    }

    /* ── Global ── */
    *:not([class*="material"]):not([data-testid="stIconMaterial"]) {
        font-family: "Inter", system-ui, -apple-system, sans-serif !important;
    }

    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background-color: var(--bg) !important;
    }
    [data-testid="stHeader"] {
        background-color: var(--bg) !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppDeployButton"] { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Typography ── */
    body, p, li, span { color: var(--text); }
    h1 { color: var(--text) !important; font-weight: 700 !important; letter-spacing: -0.02em; }
    h2 { color: var(--text) !important; font-weight: 600 !important; letter-spacing: -0.01em; }
    h3, h4, h5, h6 { color: var(--text) !important; font-weight: 600 !important; }
    [data-testid="stCaptionContainer"], .stCaption, small { color: var(--muted) !important; }

    /* ── Links ── */
    a { color: var(--accent) !important; text-decoration: none !important; transition: color .15s; }
    a:hover { color: #9d90fa !important; }

    /* ── Dividers ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: var(--radius-sm) !important;
        transition: border-color .2s, box-shadow .2s;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-dim), 0 0 12px var(--accent-dim) !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: var(--muted) !important;
        opacity: 0.6;
    }
    .stTextInput label, .stTextArea label, .stNumberInput label, .stSelectbox label {
        color: var(--text-dim) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ── Buttons ── */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        transition: all .2s ease;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--accent) !important;
        background: var(--accent-dim) !important;
        color: var(--text) !important;
        box-shadow: 0 0 16px var(--accent-dim) !important;
    }
    .stButton > button:active,
    [data-testid="stFormSubmitButton"] > button:active {
        background: var(--accent-mid) !important;
        transform: scale(0.98);
    }

    /* ── Link buttons ── */
    .stLinkButton > a {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500 !important;
        transition: all .2s ease;
    }
    .stLinkButton > a:hover {
        border-color: var(--accent) !important;
        background: var(--accent-dim) !important;
        box-shadow: 0 0 16px var(--accent-dim) !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--surface) !important;
        transition: border-color .2s;
    }
    [data-testid="stExpander"]:hover {
        border-color: #333 !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-dim) !important;
        font-size: 0.8rem;
    }

    /* ── Form container ── */
    [data-testid="stForm"] {
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        border-radius: var(--radius) !important;
        padding: 1.5rem !important;
    }

    /* ── Alerts / warnings ── */
    .stAlert {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div { color: var(--accent) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        min-width: 180px !important;
        max-width: 210px !important;
        width: 200px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        padding-top: 1.2rem;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        margin: 2px 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        font-size: 0.82rem !important;
        padding: 0.45rem 0.8rem !important;
        color: var(--text-dim) !important;
        border-radius: var(--radius-sm);
        transition: all .15s;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        color: var(--text) !important;
        background: var(--accent-dim) !important;
        border-color: var(--accent) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        color: var(--text) !important;
        background: var(--accent-dim) !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        color: var(--muted) !important;
    }

    /* ── Prob bars (shared results component) ── */
    .prob-row {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 8px; font-size: 0.85rem;
    }
    .prob-label { width: 80px; text-align: right; color: var(--muted); font-weight: 500; }
    .prob-label.top { color: var(--text); font-weight: 600; }
    .prob-bar-track {
        flex: 1; height: 6px; background: var(--border);
        border-radius: 999px; overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%; background: var(--accent); border-radius: 999px;
        transition: width .4s ease;
    }
    .prob-bar-fill.top {
        box-shadow: 0 0 8px var(--accent-glow);
    }
    .prob-value {
        width: 48px; text-align: right; color: var(--muted);
        font-variant-numeric: tabular-nums; font-weight: 500;
    }
    .prob-value.top { color: var(--text); font-weight: 600; }

    /* ── Result card ── */
    .result-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0 1rem;
    }

    /* ── Confidence ring ── */
    .conf-ring {
        text-align: center; padding-top: 4px;
    }
    .conf-ring svg { filter: drop-shadow(0 0 6px var(--accent-dim)); }

    /* ── Section headers ── */
    .section-label {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

home = st.Page("pages/home.py", title="Home", icon=":material/home:")
predict = st.Page("pages/predict.py", title="Base Model", icon=":material/search:")
final = st.Page("pages/final.py", title="BERT Model", icon=":material/model_training:")
results = st.Page("pages/results.py", title="Results", icon=":material/bar_chart:")

pg = st.navigation([home, predict, final, results])
pg.run()
