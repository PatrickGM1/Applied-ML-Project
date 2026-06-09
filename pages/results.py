import streamlit as st
import pandas as pd

st.markdown(
    """
    <div style="margin-bottom: 0.3rem;">
        <h2 style="margin-bottom: 0.1rem;">Results</h2>
        <span style="color:#666; font-size:0.82rem;">
            Model evaluation on
            <a href="https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset" target="_blank">LIAR</a>
            &amp;
            <a href="https://github.com/chengxuphd/liar2" target="_blank">LIAR2</a>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Binary results table ──
st.markdown('<div class="section-label">LIAR — Binary Classification</div>', unsafe_allow_html=True)
st.caption("All models evaluated on the same 802-sample test split (train: 6,489 samples)")

df_binary = pd.DataFrame([
    {"Model": "TF-IDF + LogReg", "Features": "text only",              "Accuracy": "63.22%", "F1 (macro)": "60.54%"},
    {"Model": "TF-IDF + LogReg", "Features": "text + metadata",        "Accuracy": "66.33%", "F1 (macro)": "65.01%"},
    {"Model": "TF-IDF + LogReg", "Features": "text + meta + history",  "Accuracy": "68.70%", "F1 (macro)": "67.39%"},
    {"Model": "BERT",            "Features": "text only",              "Accuracy": "68.08%", "F1 (macro)": "66.89%"},
    {"Model": "BERT + Meta Fusion", "Features": "text + metadata",     "Accuracy": "68.83%", "F1 (macro)": "67.60%"},
])
st.dataframe(df_binary, use_container_width=True, hide_index=True)

# ── Best model detail ──
st.markdown("---")
st.markdown('<div class="section-label">Best Model — BERT + Metadata Fusion</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", "68.83%")
with col2:
    st.metric("F1 (macro)", "67.60%")
with col3:
    st.metric("Test samples", 802)

st.markdown("")
st.caption("Per-class breakdown")

df_class = pd.DataFrame([
    {"Class": "Fake", "Precision": "65.1%", "Recall": "57.9%", "F1": "61.3%", "Support": 342},
    {"Class": "Real", "Precision": "71.1%", "Recall": "77.0%", "F1": "73.9%", "Support": 460},
])
st.dataframe(df_class, use_container_width=True, hide_index=True)

st.caption("Confusion matrix")
cm_df = pd.DataFrame(
    [[198, 144], [106, 354]],
    index=["True Fake", "True Real"],
    columns=["Pred Fake", "Pred Real"],
)
st.dataframe(cm_df, use_container_width=True)

# ── Multiclass ──
st.markdown("---")
st.markdown('<div class="section-label">LIAR — Multiclass (6-way) Classification</div>', unsafe_allow_html=True)
st.caption("Original labels: pants-fire, false, barely-true, half-true, mostly-true, true")

df_mc = pd.DataFrame([
    {"Model": "TF-IDF + LogReg", "Features": "text only",              "Accuracy": "23.54%", "F1 (macro)": "20.74%"},
    {"Model": "TF-IDF + LogReg", "Features": "text + metadata",        "Accuracy": "24.47%", "F1 (macro)": "23.53%"},
    {"Model": "TF-IDF + LogReg", "Features": "text + meta + history",  "Accuracy": "31.23%", "F1 (macro)": "31.14%"},
    {"Model": "BERT",            "Features": "text only",              "Accuracy": "27.28%", "F1 (macro)": "27.15%"},
    {"Model": "BERT + Meta Fusion", "Features": "text + metadata",     "Accuracy": "25.95%", "F1 (macro)": "24.94%"},
])
st.dataframe(df_mc, use_container_width=True, hide_index=True)
st.caption("6-way classification is much harder — fine-grained labels have overlapping semantics")

# ── LIAR2 generalization ──
st.markdown("---")
st.markdown('<div class="section-label">LIAR2 — Generalization (out-of-distribution)</div>', unsafe_allow_html=True)
st.markdown(
    """
    <span style="color:#aaa; font-size:0.82rem;">
        Best model (BERT + Meta Fusion) evaluated on
        <a href="https://github.com/chengxuphd/liar2" target="_blank">LIAR2</a>
        test set — <b>no retraining</b>.
    </span>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", "80.06%", delta="+11.23% vs LIAR")
with col2:
    st.metric("F1 (macro)", "79.52%", delta="+11.92% vs LIAR")
with col3:
    st.metric("Test samples", 1565)

st.markdown("")

df_comp = pd.DataFrame([
    {"Metric": "Accuracy",      "LIAR": "68.83%", "LIAR2": "80.06%"},
    {"Metric": "F1 (macro)",    "LIAR": "67.60%", "LIAR2": "79.52%"},
    {"Metric": "F1 (weighted)", "LIAR": "68.53%", "LIAR2": "80.29%"},
])
st.dataframe(df_comp, use_container_width=True, hide_index=True)

st.caption("Per-class breakdown (LIAR2)")
df_liar2_class = pd.DataFrame([
    {"Class": "Fake", "Precision": "88.0%", "Recall": "78.3%", "F1": "82.9%", "Support": 963},
    {"Class": "Real", "Precision": "70.5%", "Recall": "82.9%", "F1": "76.2%", "Support": 602},
])
st.dataframe(df_liar2_class, use_container_width=True, hide_index=True)

st.caption("Confusion matrix (LIAR2)")
cm2_df = pd.DataFrame(
    [[754, 209], [103, 499]],
    index=["True Fake", "True Real"],
    columns=["Pred Fake", "Pred Real"],
)
st.dataframe(cm2_df, use_container_width=True)

# ── Key findings ──
st.markdown("---")
st.markdown('<div class="section-label">Key Findings</div>', unsafe_allow_html=True)
st.markdown(
    """
    - **80% accuracy on LIAR2** without retraining — strong cross-dataset generalization
    - Performance **higher on LIAR2 than LIAR** (80.06% vs 68.83%), likely due to cleaner examples
    - BERT + metadata slightly beats TF-IDF + history (68.83% vs 68.70%), but TF-IDF uses speaker history — an unfair advantage
    - On equal features (no history), BERT wins by a bigger margin (68.83% vs 66.33%)
    - 6-way multiclass remains very hard (~31% best) — consistent with published LIAR benchmarks
    """,
)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("[guba.dev](https://guba.dev)")
