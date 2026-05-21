import streamlit as st

st.set_page_config(
    page_title="aml.guba.dev — Fake News Detection",
    page_icon="🔍",
    layout="centered",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stAppDeployButton"] { display: none; }
    .link-card {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 7px 14px;
        border: 1px solid #222; border-radius: 6px;
        color: #666; text-decoration: none;
        font-size: 0.82rem; font-weight: 500;
        transition: color .15s, border-color .15s, background .15s;
    }
    .link-card:hover { color: #efefef; border-color: #7c6af7; background: rgba(124,106,247,.12); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## aml.guba.dev")
st.markdown("Applied Machine Learning project")
st.markdown(
    "Fake News Detection — [LIAR Dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset)"
)

st.divider()

st.markdown("##### API")
col1, col2, _ = st.columns([1, 1, 2])
with col1:
    st.markdown(
        '<a class="link-card" href="http://localhost:8000/health" target="_blank">⏱ Test endpoint</a>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<a class="link-card" href="http://localhost:8000/docs" target="_blank">📄 Swagger UI</a>',
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("##### Demo")
if st.button("▶  Test base model", use_container_width=False):
    st.switch_page("pages/predict.py")

st.divider()
st.caption("© 2026 [guba.dev](https://guba.dev)")
