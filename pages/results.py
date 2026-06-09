import json
import os
import streamlit as st
import pandas as pd

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "fake_news_detection", "artifacts")

def load_json(path):
    full = os.path.join(ARTIFACTS, path)
    if os.path.exists(full):
        with open(full) as f:
            return json.load(f)
    return None

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

# ── Load all metrics ──
binary_tfidf_text = load_json("comparisons_no_history/binary_tfidf_text_only_metrics.json")
binary_tfidf_meta = load_json("comparisons_no_history/binary_tfidf_meta_no_hist_metrics.json")
binary_tfidf_full = load_json("final/binary_text_metadata_final_metrics.json")
binary_bert_text = load_json("comparisons_no_history/binary_bert_text_only_metrics.json")
binary_bert_meta = load_json("final/bert_text_metadata/binary_bert_metadata_metrics.json")
liar2_metrics = load_json("liar2/binary_bert_metadata_liar2_test_metrics.json")

# ── Binary results table ──
st.markdown('<div class="section-label">LIAR — Binary Classification</div>', unsafe_allow_html=True)
st.caption("All models evaluated on the same 802-sample test split (train: 6,489 samples)")

binary_rows = []
for name, feats, m in [
    ("TF-IDF + LogReg", "text only", binary_tfidf_text),
    ("TF-IDF + LogReg", "text + metadata", binary_tfidf_meta),
    ("TF-IDF + LogReg", "text + meta + history", binary_tfidf_full),
    ("BERT", "text only", binary_bert_text),
    ("BERT + Meta Fusion", "text + metadata", binary_bert_meta),
]:
    if m:
        binary_rows.append({
            "Model": name,
            "Features": feats,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "F1 (macro)": f"{m['f1_macro'] * 100:.2f}%",
        })

if binary_rows:
    df_binary = pd.DataFrame(binary_rows)
    st.dataframe(df_binary, use_container_width=True, hide_index=True)

# ── Best model detail ──
if binary_bert_meta:
    st.markdown("---")
    st.markdown('<div class="section-label">Best Model — BERT + Metadata Fusion</div>', unsafe_allow_html=True)

    report = binary_bert_meta["classification_report"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{binary_bert_meta['accuracy'] * 100:.2f}%")
    with col2:
        st.metric("F1 (macro)", f"{binary_bert_meta['f1_macro'] * 100:.2f}%")
    with col3:
        st.metric("Test samples", binary_bert_meta["eval_rows"])

    st.markdown("")
    st.caption("Per-class breakdown")

    class_rows = []
    for label, name in [("0", "Fake"), ("1", "Real")]:
        r = report[label]
        class_rows.append({
            "Class": name,
            "Precision": f"{r['precision'] * 100:.1f}%",
            "Recall": f"{r['recall'] * 100:.1f}%",
            "F1": f"{r['f1-score'] * 100:.1f}%",
            "Support": int(r["support"]),
        })
    st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

    cm = binary_bert_meta["confusion_matrix"]
    st.caption("Confusion matrix")
    cm_df = pd.DataFrame(cm, index=["True Fake", "True Real"], columns=["Pred Fake", "Pred Real"])
    st.dataframe(cm_df, use_container_width=True)

# ── Multiclass ──
st.markdown("---")
st.markdown('<div class="section-label">LIAR — Multiclass (6-way) Classification</div>', unsafe_allow_html=True)
st.caption("Original labels: pants-fire, false, barely-true, half-true, mostly-true, true")

mc_tfidf_text = load_json("comparisons_no_history/multiclass_tfidf_text_only_metrics.json")
mc_tfidf_meta = load_json("comparisons_no_history/multiclass_tfidf_meta_no_hist_metrics.json")
mc_tfidf_full = load_json("baselines/multiclass_text_metadata_metrics.json")
mc_bert_text = load_json("final/bert_text_only/multiclass_bert_text_only_metrics.json")
mc_bert_meta = load_json("final/bert_text_metadata/multiclass_bert_metadata_metrics.json")

mc_rows = []
for name, feats, m in [
    ("TF-IDF + LogReg", "text only", mc_tfidf_text),
    ("TF-IDF + LogReg", "text + metadata", mc_tfidf_meta),
    ("TF-IDF + LogReg", "text + meta + history", mc_tfidf_full),
    ("BERT", "text only", mc_bert_text),
    ("BERT + Meta Fusion", "text + metadata", mc_bert_meta),
]:
    if m:
        mc_rows.append({
            "Model": name,
            "Features": feats,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "F1 (macro)": f"{m.get('f1_macro', 0) * 100:.2f}%",
        })

if mc_rows:
    st.dataframe(pd.DataFrame(mc_rows), use_container_width=True, hide_index=True)
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

if liar2_metrics and binary_bert_meta:
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        delta_acc = (liar2_metrics["accuracy"] - binary_bert_meta["accuracy"]) * 100
        st.metric("Accuracy", f"{liar2_metrics['accuracy'] * 100:.2f}%", delta=f"{delta_acc:+.2f}% vs LIAR")
    with col2:
        delta_f1 = (liar2_metrics["f1_macro"] - binary_bert_meta["f1_macro"]) * 100
        st.metric("F1 (macro)", f"{liar2_metrics['f1_macro'] * 100:.2f}%", delta=f"{delta_f1:+.2f}% vs LIAR")
    with col3:
        st.metric("Test samples", liar2_metrics["rows_evaluated"])

    st.markdown("")

    comp_rows = [
        {"Metric": "Accuracy", "LIAR": f"{binary_bert_meta['accuracy'] * 100:.2f}%", "LIAR2": f"{liar2_metrics['accuracy'] * 100:.2f}%"},
        {"Metric": "F1 (macro)", "LIAR": f"{binary_bert_meta['f1_macro'] * 100:.2f}%", "LIAR2": f"{liar2_metrics['f1_macro'] * 100:.2f}%"},
        {"Metric": "F1 (weighted)", "LIAR": f"{binary_bert_meta['f1_weighted'] * 100:.2f}%", "LIAR2": f"{liar2_metrics['f1_weighted'] * 100:.2f}%"},
    ]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.caption("Per-class breakdown (LIAR2)")
    liar2_report = liar2_metrics["classification_report"]
    liar2_class_rows = []
    for label in ["fake", "real"]:
        r = liar2_report[label]
        liar2_class_rows.append({
            "Class": label.capitalize(),
            "Precision": f"{r['precision'] * 100:.1f}%",
            "Recall": f"{r['recall'] * 100:.1f}%",
            "F1": f"{r['f1-score'] * 100:.1f}%",
            "Support": int(r["support"]),
        })
    st.dataframe(pd.DataFrame(liar2_class_rows), use_container_width=True, hide_index=True)

    cm2 = liar2_metrics["confusion_matrix"]
    st.caption("Confusion matrix (LIAR2)")
    cm2_df = pd.DataFrame(cm2, index=["True Fake", "True Real"], columns=["Pred Fake", "Pred Real"])
    st.dataframe(cm2_df, use_container_width=True)

elif not liar2_metrics:
    st.warning("LIAR2 metrics not found. Run evaluate_liar2_best_model.py first.")

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
