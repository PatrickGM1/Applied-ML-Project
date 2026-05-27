import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Demo — Fake News Detection",
    page_icon="🔍",
    layout="centered",
)

st.markdown(
    """
    <style>
    .prob-row {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 6px; font-size: 0.9rem;
    }
    .prob-label { width: 110px; text-align: right; color: #888; }
    .prob-label.top { color: #e8e8e8; font-weight: 600; }
    .prob-bar-track {
        flex: 1; height: 6px; background: #2a2a2a;
        border-radius: 999px; overflow: hidden;
    }
    .prob-bar-fill { height: 100%; background: #7c6af7; border-radius: 999px; }
    .prob-bar-fill.top { box-shadow: 0 0 6px rgba(124,106,247,0.6); }
    .prob-value { width: 48px; text-align: right; color: #888; font-variant-numeric: tabular-nums; }
    .prob-value.top { color: #e8e8e8; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("← Home"):
    st.switch_page("pages/home.py")

st.markdown("## Base Model")
st.caption("Enter a political claim and optional speaker metadata.")
st.divider()

with st.form("predict_form"):
    statement = st.text_area("Claim text", placeholder="Enter a political claim…", height=120)

    col1, col2 = st.columns(2)
    with col1:
        subjects = st.text_input("Subjects (comma-separated)", placeholder="economy,budget")
        state = st.text_input("State", placeholder="texas")
    with col2:
        party = st.text_input("Party", placeholder="republican")
        speaker_job = st.text_input("Speaker job", placeholder="senator")

    with st.expander("Speaker history (optional) — prior label counts for this speaker"):
        h1, h2, h3, h4, h5 = st.columns(5)
        hist1 = h1.number_input("Barely true", min_value=0, step=1, value=0)
        hist2 = h2.number_input("False", min_value=0, step=1, value=0)
        hist3 = h3.number_input("Half true", min_value=0, step=1, value=0)
        hist4 = h4.number_input("Mostly true", min_value=0, step=1, value=0)
        hist5 = h5.number_input("Pants on fire", min_value=0, step=1, value=0)

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    if not statement.strip():
        st.error("Claim text is required.")
    else:
        def normalize(v):
            return v.strip() if v and v.strip() else "missing"

        payload = {
            "statement": statement.strip(),
            "subjects": subjects.strip(),
            "party": normalize(party),
            "state": normalize(state),
            "speaker_job": normalize(speaker_job),
            "hist1": float(hist1), "hist2": float(hist2),
            "hist3": float(hist3), "hist4": float(hist4),
            "hist5": float(hist5),
        }

        with st.spinner("Running inference…"):
            try:
                resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(f"Could not reach the API at `{API_BASE_URL}`. Make sure the backend is running.")
                st.stop()
            except requests.exceptions.HTTPError as exc:
                st.error(f"API error: {exc.response.json().get('detail', str(exc))}")
                st.stop()

        st.divider()

        confidence = data.get("confidence", 0)
        conf_pct = round(confidence * 100, 1)
        label = data.get("predicted_label", "—")

        res_col, meter_col = st.columns([3, 1])
        with res_col:
            st.markdown(f"### {label.capitalize()}")
            st.caption(f"Confidence: **{conf_pct}%**")
        with meter_col:
            color = "#4ade80" if conf_pct >= 75 else "#facc15" if conf_pct >= 60 else "#f87171"
            st.markdown(
                f"""<div style="text-align:center;padding-top:8px">
                <svg viewBox="0 0 36 36" width="64" height="64">
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="#222" stroke-width="3"/>
                    <circle cx="18" cy="18" r="15.9" fill="none" stroke="{color}" stroke-width="3"
                        stroke-dasharray="{conf_pct:.1f} 100" stroke-linecap="round"
                        transform="rotate(-90 18 18)"/>
                </svg>
                <div style="font-size:.75rem;font-weight:700;color:{color}">{conf_pct}%</div>
                </div>""",
                unsafe_allow_html=True,
            )

        if data.get("is_low_confidence"):
            st.warning("Low confidence — treat this as a weak signal, not a verdict.")

        st.markdown("**Class probabilities**")
        probs = sorted(data.get("class_probabilities", {}).items(), key=lambda x: x[1], reverse=True)
        bars = ""
        for lbl, val in probs:
            pct = val * 100
            top = lbl == label
            tc = " top" if top else ""
            bars += f"""<div class="prob-row">
                <span class="prob-label{tc}">{lbl}</span>
                <div class="prob-bar-track"><div class="prob-bar-fill{tc}" style="width:{pct:.1f}%"></div></div>
                <span class="prob-value{tc}">{pct:.1f}%</span>
            </div>"""
        st.markdown(bars, unsafe_allow_html=True)

st.divider()
st.caption("© 2026 [guba.dev](https://guba.dev)")
