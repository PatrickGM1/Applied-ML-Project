import os
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", API_BASE_URL)

st.markdown(
    "Fake News Detection -> [LIAR Dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)"
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
