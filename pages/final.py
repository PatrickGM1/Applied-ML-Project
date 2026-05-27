import base64
import os

import streamlit as st

if st.button("← Home"):
    st.switch_page("pages/home.py")

st.divider()

st.markdown("## Final Model")
st.caption("BERT")

st.divider()

_img_path = "pages/sign.jpg"
_img_b64 = ""
if os.path.exists(_img_path):
    with open(_img_path, "rb") as _f:
        _img_b64 = base64.b64encode(_f.read()).decode()

st.markdown(
    f"""
    <div style="text-align:center; padding: 48px 0 32px;">
        {"" if not _img_b64 else f'<img src="data:image/jpeg;base64,{_img_b64}" style="width:160px; border-radius:10px; margin-bottom:16px;" />'}
        <h3 style="color:#e8e8e8; margin-bottom:8px;">Work in progress</h3>
        <p style="color:#888; font-size:0.9rem; max-width:360px; margin:0 auto 24px;">
            The final model is being trained, evaluated, and very dramatically
            prepared for its grand arrival.
        </p>
        <p style="color:#7c6af7; font-size:0.8rem; font-weight:600; letter-spacing:0.05em;">
            ARRIVING SOON™ (Hopefully)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.caption("© 2026 [guba.dev](https://guba.dev)")
