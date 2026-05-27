import streamlit as st

st.markdown(
    """
    <style>
    :root {
        --bg: #0e0e0e;
        --surface: #161616;
        --border: #2a2a2a;
        --border-hover: #7c6af7;
        --text: #e8e8e8;
        --muted: #888;
        --accent: #7c6af7;
        --accent-dim: rgba(124, 106, 247, 0.12);
    }

    /* App background */
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background-color: var(--bg) !important;
    }
    [data-testid="stHeader"] {
        background-color: var(--bg) !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppDeployButton"] { display: none; }

    /* Text */
    body, p, li { color: var(--text); font-family: "Inter", system-ui, sans-serif; }
    h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
    [data-testid="stCaptionContainer"], .stCaption, small { color: var(--muted) !important; }

    /* Links */
    a { color: var(--accent) !important; }
    a:hover { color: var(--text) !important; text-decoration: underline; }

    /* Dividers */
    hr { border-color: var(--border) !important; }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--border-hover) !important;
        box-shadow: 0 0 0 1px var(--border-hover) !important;
    }
    .stTextInput label, .stTextArea label, .stNumberInput label {
        color: var(--muted) !important;
        font-size: 0.75rem;
    }

    /* Buttons */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
        font-weight: 600;
        transition: border-color .15s, background .15s;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--border-hover) !important;
        background: var(--accent-dim) !important;
        color: var(--text) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border: 1px dashed var(--border) !important;
        border-radius: 8px !important;
        background: var(--surface) !important;
    }
    [data-testid="stExpander"] summary { color: var(--muted); font-size: 0.8rem; }

    /* Form container */
    [data-testid="stForm"] {
        border-color: var(--border) !important;
        background: var(--surface) !important;
        border-radius: 10px !important;
    }

    /* Alerts / warnings */
    .stAlert { background: var(--surface) !important; border-color: var(--border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

home = st.Page("pages/home.py", title="Home", icon=":material/home:")
predict = st.Page("pages/predict.py", title="Demo", icon=":material/search:")

pg = st.navigation([home, predict], position="hidden")
pg.run()
