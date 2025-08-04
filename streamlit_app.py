# ───────────────────────────────────────────────────────────
#  streamlit_app.py   (Aug‑2025, stable UX version)
#  ----------------------------------------------------------
#  • Step 4: build / edit dictionary
#  • Step 5‑A: Run Classification  → stores predictions
#  • Step 5‑B: Optional ground‑truth  → Compute Metrics
#  • Downloads include predictions, ground‑truth, tactic_flag
# ───────────────────────────────────────────────────────────
import ast, re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Tactic Classifier + Metrics", layout="wide")
st.title("📊 Marketing‑Tactic Text Classifier + Metrics")

# ────────────────── built‑in dictionaries ──────────────────
DEFAULT_TACTICS = {
    "urgency_marketing":  ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof":       ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"],
    "Classic_Timeless_Luxury_style": [
        'elegance', 'heritage', 'sophistication', 'refined', 'timeless', 'grace',
        'legacy', 'opulence', 'bespoke', 'tailored', 'understated', 'prestige',
        'quality', 'craftsmanship', 'heirloom', 'classic', 'tradition', 'iconic',
        'enduring', 'rich', 'authentic', 'luxury', 'fine', 'pure', 'exclusive',
        'elite', 'mastery', 'immaculate', 'flawless', 'distinction', 'noble',
        'chic', 'serene', 'clean', 'minimal', 'poised', 'balanced', 'eternal',
        'neutral', 'subtle', 'grand', 'timelessness', 'tasteful', 'quiet', 'sublime'
    ]
}
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(DEFAULT_TACTICS.keys()))
st.write(f"Chosen tactic: **{tactic}**")

# ───────────────────────── helpers ─────────────────────────
def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

def classify(txt: str, dct):
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]

def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        try: return ast.literal_eval(x)
        except Exception: return []
    return []

def safe_bool(x):
    """Accept 0/1, True/False, '0'/'1' as booleans."""
    if isinstance(x, (int, float)): return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes"}
    return False
# ───────────────────────────────────────────────────────────

# initialise session objects
defaults = {
    "dict_ready": False,
    "dictionary": {},
    "top_words":  pd.Series(dtype=int),
    "raw_df":     pd.DataFrame(),  # uploaded raw data
    "pred_df":    pd.DataFrame(),  # predictions stored here
    "gt_df":      pd.DataFrame()   # ground‑truth (if any)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ────────────────── STEP 2 – upload raw CSV ─────────────────
raw_file = st.file_uploader("📁 Step 2 — upload raw CSV", type="csv")
if raw_file:
    st.session_state.raw_df = pd.read_csv(raw_file)
    if "ID" not in st.session_state.raw_df.columns:
        st.session_state.raw_df.insert(0, "ID", st.session_state.raw_df.index.astype(str))
    st.dataframe(st.session_state.raw_df.head())

# need raw_df for everything else
if st.session_state.raw_df.empty:
    st.stop()

text_col = st.selectbox("📋 Step 3 — select text column",
                        st.session_state.raw_df.columns)

# ─────────── STEP 4 – generate / refine dictionary ─────────
if st.button("🧠 Step 4 — Generate / refine dictionary"):
    df = st.session_state.raw_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    base_terms = set(DEFAULT_TACTICS[tactic])
    df["row_matches_tactic"] = df["cleaned"].apply(
        lambda x: any(tok in x.split() for tok in base_terms)
    )
    pos_df = df[df["row_matches_tactic"]]

    stop_words = {'the','is','in','on','and','a','for','you','i','are','of',
                  'your','to','my','with','it','me','this','that','or'}

    if pos_df.empty:
        contextual_terms, contextual_freq = [], pd.Series(dtype=int)
        st.warning("No rows matched seed words; using default list only.")
    else:
        word_freq = (pos_df["cleaned"]
                     .str.split(expand=True)
                     .stack()
                     .value_counts())
        contextual_terms = [w for w in word_freq.index
                            if w not in stop_words and w not in base_terms][:30]
        contextual_freq  = word_freq.loc[contextual_terms]

    auto_dict = {tactic: sorted(base_terms.union(contextual_terms))}

    st.subheader("Contextual keywords")
    if not contextual_freq.empty:
        st.dataframe(contextual_freq.rename("Freq"))
    else:
        st.write("‑‑ none found ‑‑")

    dict_text = st.text_area("✏️ Edit dictionary (Python dict syntax)",
                             value=str(auto_dict), height=150)
    try:
        st.session_state.dictionary = ast.literal_eval(dict_text)
        st.success("Dictionary saved.")
    except Exception:
        st.session_state.dictionary = auto_dict
        st.error("Bad format → reverted to auto dictionary.")

    st.session_state.top_words  = contextual_freq
    st.session_state.dict_ready = True

# ───────── STEP 5‑A – RUN CLASSIFICATION (predictions) ─────
st.subheader("Step 5‑A — Classification")
if st.button("🔹 1. Run Classification",
             disabled=not st.session_state.dict_ready):
    df = st.session_state.raw_df.copy()
    df["cleaned"] = df[text_col].apply(clean)

    dct = st.session_state.dictionary
    df["categories"]   = df["cleaned"].apply(lambda x: classify(x, dct))
    df["tactic_flag"]  = df["categories"].apply(lambda cats: int(tactic in cats))

    st.session_state.pred_df = df.copy()   # keep for later
    st.success("Predictions generated and stored.")

    # quick view
    st.dataframe(df.head())

# show word‑stats if predictions exist
if not st.session_state.pred_df.empty:
    counts = pd.Series(
        [c for cats in st.session_state.pred_df["categories"] for c in cats]
    ).value_counts()
    st.markdown("##### Category frequencies")
    st.table(counts)

# ───────── STEP 5‑B – GROUND‑TRUTH + METRICS ───────────────
st.subheader("Step 5‑B — Ground‑truth & Metrics (optional)")

with st.expander("📥  Provide ground‑truth labels"):
    gt_source = st.radio("Choose method", ["None", "Upload CSV", "Manual entry"],
                         horizontal=True)
    if gt_source == "Upload CSV":
        gt_file = st.file_uploader("Upload CSV with ID & true_label (or truth_flag)",
                                   type="csv")
        if gt_file:
            df_gt = pd.read_csv(gt_file)
            if "ID" not in df_gt.columns:
                st.error("Ground‑truth file must contain an ID column.")
            else:
                st.session_state.gt_df = df_gt
                st.success("Ground‑truth file loaded.")
    elif gt_source == "Manual entry":
        if st.session_state.pred_df.empty:
            st.info("Run classification first, then you can edit ground‑truth here.")
        else:
            if "true_label" not in st.session_state.pred_df.columns:
                st.session_state.pred_df["true_label"] = "[]"
            editable = st.data_editor(
                st.session_state.pred_df[["ID", text_col, "true_label"]],
                num_rows="dynamic", height=400, key="manual_gt")
            st.session_state.pred_df["true_label"] = editable["true_label"]

# ---------- COMPUTE METRICS ----------
if st.button("🔹 2. Compute Metrics",
             disabled=st.session_state.pred_df.empty):
    df_pred = st.session_state.pred_df.copy()

    # merge gt (if provided via upload)
    if not st.session_state.gt_df.empty:
        gt = st.session_state.gt_df.copy()
        # allow boolean flag column (same name as tactic + '_flag') or true_label list
        if f"{tactic}_flag" in gt.columns:
            gt["true_label"] = gt[f"{tactic}_flag"].apply(
                lambda x: [tactic] if safe_bool(x) else [])
        elif "true_label" not in gt.columns:
            st.error("Ground‑truth must have a 'true_label' or "
                     f"'{tactic}_flag' column.")
            st.stop()
        df_pred = df_pred.merge(gt[["ID", "true_label"]], on="ID", how="left")

    # verify we have some truth
    if "true_label" not in df_pred.columns or df_pred["true_label"].isna().all():
        st.warning("No ground‑truth labels present → cannot compute metrics.")
    else:
        df_pred["__gt_list__"]   = df_pred["true_label"].apply(to_list)
        df_pred["__pred_list__"] = df_pred["categories"]

        results = []
        for tac in st.session_state.dictionary.keys():
            df_pred["__pred_flag__"] = df_pred["__pred_list__"].apply(lambda lst: tac in lst)
            df_pred["__gt_flag__"]   = df_pred["__gt_list__"].apply(lambda lst: tac in lst)

            TP = int(((df_pred["__pred_flag__"]) & (df_pred["__gt_flag__"])).sum())
            FP = int(((df_pred["__pred_flag__"]) & (~df_pred["__gt_flag__"])).sum())
            FN = int((~df_pred["__pred_flag__"] & (df_pred["__gt_flag__"])).sum())

            prec = TP / (TP + FP) if TP + FP else 0.0
            recall = TP / (TP + FN) if TP + FN else 0.0
            f1 = 2*prec*recall / (prec + recall) if prec + recall else 0.0

            results.append({"tactic": tac,
                            "TP": TP, "FP": FP, "FN": FN,
                            "precision": prec, "recall": recall, "f1": f1})

        mdf = pd.DataFrame(results).set_index("tactic")
        st.markdown("##### Precision / Recall / F1")
        st.dataframe(mdf.style.format({"precision":"{:.3f}",
                                       "recall":"{:.3f}",
                                       "f1":"{:.3f}"}))

        # store back the merged truth
        st.session_state.pred_df = df_pred

# ───────── DOWNLOADS ───────────────────────────────────────
if not st.session_state.pred_df.empty:
    st.markdown("### 📥 Download results")
    st.download_button("classified_results.csv",
                       st.session_state.pred_df.to_csv(index=False).encode(),
                       "classified_results.csv", "text/csv")
