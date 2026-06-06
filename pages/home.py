import os
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", API_BASE_URL)

st.markdown(
    """
    <style>
    /* Model card buttons */
    div[data-testid="stColumn"] .stButton > button {
        background: #141414 !important;
        border: 1px solid #222 !important;
        border-left: 3px solid #7c6af7 !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        text-align: left !important;
        min-height: 80px !important;
        transition: all .2s ease !important;
    }
    div[data-testid="stColumn"] .stButton > button:hover {
        border-color: #7c6af7 !important;
        background: rgba(124, 106, 247, 0.08) !important;
        box-shadow: 0 0 16px rgba(124, 106, 247, 0.1) !important;
    }
    div[data-testid="stColumn"] .stButton > button p {
        text-align: left !important;
    }
    div[data-testid="stColumn"] .stButton > button p:first-child {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #e8e8e8 !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stColumn"] .stButton > button p:last-child {
        font-weight: 400 !important;
        font-size: 0.78rem !important;
        color: #666 !important;
    }
    /* API link buttons */
    .stLinkButton > a {
        background: #141414 !important;
        border: 1px solid #222 !important;
        border-radius: 8px !important;
        color: #aaa !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1rem !important;
        transition: all .2s ease !important;
    }
    .stLinkButton > a:hover {
        border-color: #7c6af7 !important;
        color: #e8e8e8 !important;
        background: rgba(124, 106, 247, 0.08) !important;
        box-shadow: 0 0 12px rgba(124, 106, 247, 0.1) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0.2rem; font-size: 1.8rem;">Fake News Detection</h1>
        <p style="color: #666; font-size: 0.9rem; margin: 0;">
            Applied Machine Learning &mdash;
            <a href="https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset"
               target="_blank">LIAR Dataset</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">API</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.link_button("v1 Health", f"{PUBLIC_BASE_URL}/v1/health", use_container_width=True)
with c2:
    st.link_button("v2 Health", f"{PUBLIC_BASE_URL}/v2/health", use_container_width=True)
with c3:
    st.link_button("Swagger UI", f"{PUBLIC_BASE_URL}/docs", use_container_width=True)

st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("TF-IDF + Logistic Regression\n\nText + metadata + history · v1", use_container_width=True):
        st.switch_page("pages/predict.py")

with col2:
    if st.button("BERT + Metadata Fusion\n\nBERT encoder + metadata · v2", use_container_width=True):
        st.switch_page("pages/final.py")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("[guba.dev](https://guba.dev)")
