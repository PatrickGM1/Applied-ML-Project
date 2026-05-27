import streamlit as st

st.set_page_config(
    page_title="aml.guba.dev — Fake News Detection",
    page_icon="🔍",
    layout="centered",
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
    st.link_button("⏱ Test endpoint", "http://localhost:8000/health")
with col2:
    st.link_button("📄 Swagger UI", "http://localhost:8000/docs")

st.divider()

st.markdown("##### Demo")
if st.button("Test base model", use_container_width=False):
    st.switch_page("pages/predict.py")

st.divider()
st.caption("© 2026 [guba.dev](https://guba.dev)")
