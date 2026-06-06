import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.markdown(
    """
    <div style="margin-bottom: 0.3rem;">
        <h2 style="margin-bottom: 0.1rem;">BERT Model</h2>
        <span style="color:#666; font-size:0.82rem;">BERT + Metadata Fusion &middot; text encoder + categorical metadata</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

with st.form("bert_predict_form"):
    statement = st.text_area("Statement", placeholder="Enter a political claim…", height=100)

    col1, col2 = st.columns(2)
    with col1:
        subjects = st.text_input("Subjects (comma-separated)", placeholder="economy,budget")
        speaker = st.text_input("Speaker", placeholder="barack-obama")
        state = st.text_input("State", placeholder="texas")
    with col2:
        party = st.text_input("Party", placeholder="republican")
        speaker_job = st.text_input("Speaker job", placeholder="senator")
        context = st.text_input("Context", placeholder="a speech")

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted and statement and statement.strip():
    def normalize(v):
        return v.strip() if v and v.strip() else "unknown"

    payload = {
        "statement": statement.strip(),
        "subjects": subjects.strip(),
        "speaker": normalize(speaker),
        "party": normalize(party),
        "state": normalize(state),
        "speaker_job": normalize(speaker_job),
        "context": normalize(context),
    }

    with st.spinner("Running BERT inference…"):
        try:
            resp = requests.post(f"{API_BASE_URL}/v2/predictions", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error(f"Could not reach the API at `{API_BASE_URL}`. Make sure the backend is running.")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
            st.error(f"API error: {detail}")
            st.stop()

    st.markdown("---")

    confidence = data.get("confidence", 0)
    conf_pct = round(confidence * 100, 1)
    label = data.get("label", "-")
    color = "#4ade80" if conf_pct >= 75 else "#facc15" if conf_pct >= 60 else "#f87171"
    icon = "&#10003;" if label == "real" else "&#10007;"

    st.markdown(
        f"""
        <div class="result-card" style="border-left: 3px solid {color};">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="font-size:1.6rem; color:{color};">{icon}</div>
                    <div>
                        <div style="font-size:1.3rem; font-weight:700; color:{color};">{label.capitalize()}</div>
                        <div style="color:#555; font-size:0.78rem; margin-top:2px;">
                            {conf_pct}% confidence
                        </div>
                    </div>
                </div>
                <div class="conf-ring">
                    <svg viewBox="0 0 36 36" width="52" height="52">
                        <circle cx="18" cy="18" r="15.9" fill="none" stroke="#1a1a1a" stroke-width="2.5"/>
                        <circle cx="18" cy="18" r="15.9" fill="none" stroke="{color}" stroke-width="2.5"
                            stroke-dasharray="{conf_pct:.1f} 100" stroke-linecap="round"
                            transform="rotate(-90 18 18)"/>
                    </svg>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data.get("is_low_confidence"):
        st.markdown(
            """
            <div style="background:rgba(250,204,21,0.08); border:1px solid rgba(250,204,21,0.2);
                        border-radius:8px; padding:0.6rem 1rem; margin:0.5rem 0; font-size:0.82rem; color:#facc15;">
                Low confidence — treat this as a weak signal, not a verdict.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Class probabilities</div>', unsafe_allow_html=True)
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

st.markdown("<br>", unsafe_allow_html=True)
st.caption("[guba.dev](https://guba.dev)")
