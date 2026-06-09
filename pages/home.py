import os
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", API_BASE_URL)

st.markdown(
    """
    <div style="margin-bottom: 1.8rem;">
        <h1 style="margin-bottom: 0.2rem; font-size: 1.8rem;">Fake News Detection</h1>
        <p style="color: #666; font-size: 0.85rem; margin: 0; line-height: 1.6;">
            Classifying political statements as fake or real using the
            <a href="https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset"
               target="_blank">LIAR</a> dataset.
            Applied Machine Learning &mdash; University of Groningen.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="background:#141414; border:1px solid #222; border-radius:10px;
                    padding:1.2rem 1.3rem;">
            <div style="font-size:0.6rem; font-weight:600; text-transform:uppercase;
                        letter-spacing:0.08em; color:#666; margin-bottom:0.5rem;">v1 · Base Model</div>
            <div style="font-size:0.95rem; font-weight:600; color:#e8e8e8; margin-bottom:0.3rem;">
                TF-IDF + Logistic Regression
            </div>
            <div style="color:#555; font-size:0.75rem;">
                Text + metadata + speaker history
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Try v1", use_container_width=True, key="btn_v1"):
        st.switch_page("pages/predict.py")

with col2:
    st.markdown(
        """
        <div style="background:#141414; border:1px solid #deff9a; border-radius:10px;
                    padding:1.2rem 1.3rem;
                    box-shadow:0 0 20px rgba(222,255,154,0.06);">
            <div style="font-size:0.6rem; font-weight:600; text-transform:uppercase;
                        letter-spacing:0.08em; color:#deff9a; margin-bottom:0.5rem;">v2 · Final Model</div>
            <div style="font-size:0.95rem; font-weight:600; color:#e8e8e8; margin-bottom:0.3rem;">
                BERT + Metadata Fusion
            </div>
            <div style="color:#555; font-size:0.75rem;">
                BERT encoder + categorical metadata
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Try v2", use_container_width=True, key="btn_v2"):
        st.switch_page("pages/final.py")

st.markdown("---")

st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
st.markdown(
    """
    <p style="color:#555; font-size:0.8rem; margin-bottom:0.8rem;">
        Benchmarked on
        <a href="https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset" target="_blank">LIAR</a>
        and
        <a href="https://github.com/chengxuphd/liar2" target="_blank">LIAR2</a>
        — binary and multiclass, with cross-dataset generalization.
    </p>
    """,
    unsafe_allow_html=True,
)
if st.button("View Results", use_container_width=True, key="btn_results"):
    st.switch_page("pages/results.py")

st.markdown("---")

with st.expander("API"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("v1 Health", f"{PUBLIC_BASE_URL}/v1/health", use_container_width=True)
    with c2:
        st.link_button("v2 Health", f"{PUBLIC_BASE_URL}/v2/health", use_container_width=True)
    with c3:
        st.link_button("Swagger UI", f"{PUBLIC_BASE_URL}/docs", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("[guba.dev](https://guba.dev)")
