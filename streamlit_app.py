# ──────────────────────────────────────────────────────────────
#  streamlit_app.py   (end‑to‑end version, Aug‑2025)
#  -------------------------------------------------------------
#  1.  Builds / edits a tactic‑aware dictionary
#  2.  Classifies the uploaded text
#  3.  Lets you add ground‑truth labels
#  4.  Shows precision, recall, F1 (per tactic)
#  5.  Exports a single CSV containing predictions + truth
# ──────────────────────────────────────────────────────────────
import ast
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Marketing‑Tactic Text Classifier", layout="wide")
st.title("📊 Marketing‑Tactic Text Classifier + Metrics")

# ───────────────── STEP 1 – choose tactic ────────────────────
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

# ───────────────── STEP 2 – upload CSV ───────────────────────
file = st.file_uploader("📁 Step 2 — upload CSV", type="csv")

# ---------- helper functions ---------------------------------
def clean(txt: str) -> str:
    """Lower‑case & remove punctuation/digits for simple tokenisation."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

def classify(txt: str, dct):
    """Return list of categories whose term list appears at least once."""
    toks = txt.split()
    return [cat for cat, terms in dct.items() if any(w in toks for w in terms)] or ["uncategorized"]

def safe_literal_eval(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []
# -------------------------------------------------------------

# ────────── Initialise session placeholders ──────────────────
for key, default in {
    "dict_ready": False,
    "dictionary": {},
    "top_words":  pd.Series(dtype=int),
    "df":         pd.DataFrame()
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────
#                        MAIN APP
# ──────────────────────────────────────────────────────────────
if file:
    df = pd.read_csv(file)

    # ensure an ID column exists for later merges / editing
    if "ID" not in df.columns:
        df.insert(0, "ID", df.index.astype(str))

    st.dataframe(df.head())

    text_col = st.selectbox("📋 Step 3 — select text column", df.columns)

    # ────────── STEP 4 – generate / refine dictionary ─────────
    if st.button("🧠 Step 4 — Generate Keywords & Dictionary"):
        df["cleaned"] = df[text_col].apply(clean)

        base_terms = set(DEFAULT_TACTICS[tactic])
        df["row_matches_tactic"] = df["cleaned"].apply(
            lambda x: any(tok in x.split() for tok in base_terms)
        )
        pos_df = df[df["row_matches_tactic"]]

        stop_words = {
            'the', 'is', 'in', 'on', 'and', 'a', 'for', 'you', 'i', 'are', 'of',
            'your', 'to', 'my', 'with', 'it', 'me', 'this', 'that', 'or'
        }

        if pos_df.empty:
            st.warning(
                "No rows contained seed words for this tactic — using default list only."
            )
            contextual_terms = []
            contextual_freq  = pd.Series(dtype=int)
        else:
            all_pos_words = " ".join(pos_df["cleaned"]).split()
            word_freq = pd.Series(all_pos_words).value_counts()
            contextual_terms = [
                w for w in word_freq.index
                if w not in stop_words and w not in base_terms
            ][:30]
            contextual_freq = word_freq.loc[contextual_terms]

        auto_dict = {tactic: sorted(base_terms.union(contextual_terms))}

        st.subheader("Top contextual keywords")
        if not contextual_freq.empty:
            st.dataframe(contextual_freq.rename("Frequency"))
        else:
            st.write("‑‑ none found ‑‑")

        st.write("Auto‑generated dictionary:", auto_dict)

        dict_text = st.text_area(
            "✏️ Edit dictionary (Python dict syntax)",
            value=str(auto_dict),
            height=150
        )
        try:
            st.session_state.dictionary = ast.literal_eval(dict_text)
            st.success("Dictionary saved.")
        except Exception:
            st.error("Bad format → using auto dict.")
            st.session_state.dictionary = auto_dict

        st.session_state.df         = df
        st.session_state.top_words  = contextual_freq
        st.session_state.dict_ready = True

    # ─────────── STEP 5 – classify & score (if ready) ────────
    if st.session_state.dict_ready:
        if st.button("🧪 Step 5 — Run Classification"):
            df         = st.session_state.df.copy()
            top_words  = st.session_state.top_words
            dictionary = st.session_state.dictionary

            # ---------- predictions ----------
            df["categories"] = df["cleaned"].apply(lambda x: classify(x, dictionary))
            df["tactic_flag"] = df["categories"].apply(
                lambda cats: 1 if tactic in cats else 0
            )

            # ---------- OPTIONAL: ground‑truth input ----------
            st.markdown("### 🖍 Add ground‑truth labels (optional)")
            with st.expander("Provide the correct label for each row"):
                gt_mode = st.radio(
                    "Ground‑truth source",
                    ["None", "Manual entry", "Upload CSV"],
                    horizontal=True,
                    index=0
                )

                if gt_mode == "Manual entry":
                    if "true_label" not in df.columns:
                        df["true_label"] = "[]"
                    df_edit = st.data_editor(
                        df[["ID", text_col, "true_label"]],
                        num_rows="dynamic",
                        height=400,
                        key="gt_editor"
                    )
                    df["true_label"] = df_edit["true_label"]

                elif gt_mode == "Upload CSV":
                    gt_file = st.file_uploader(
                        "Upload CSV with ID and true_label columns",
                        type="csv",
                        key="gt_upload"
                    )
                    if gt_file:
                        df_gt = pd.read_csv(gt_file)
                        if {"ID", "true_label"}.issubset(df_gt.columns):
                            df = df.merge(df_gt[["ID", "true_label"]],
                                          on="ID", how="left")
                            st.success("Ground‑truth merged.")
                        else:
                            st.error("CSV must contain ID and true_label columns.")

            # ---------- display word‑level info ----------
            counts = pd.Series(
                [c for cats in df["categories"] for c in cats]
            ).value_counts()

            st.subheader("📊 Category frequencies")
            st.table(counts)

            st.subheader("🔑 Top contextual keywords")
            if not top_words.empty:
                st.table(top_words)
            else:
                st.write("‑‑ none to display ‑‑")

            fig, ax = plt.subplots(figsize=(10, 4))
            if not top_words.empty:
                top_words.sort_values(ascending=False).plot.bar(ax=ax)
                ax.set_title("Top contextual keyword frequencies")
            else:
                ax.text(0.5, 0.5, "No contextual keywords", ha="center", va="center")
                ax.set_axis_off()
            st.pyplot(fig)

            # ---------- precision / recall / F1 ----------
            if "true_label" in df.columns and df["true_label"].notna().any():
                df["__gt_list__"]   = df["true_label"].apply(safe_literal_eval)
                df["__pred_list__"] = df["categories"]

                metric_rows = []
                for tac in dictionary.keys():
                    df["__pred_flag__"] = df["__pred_list__"].apply(lambda lst: tac in lst)
                    df["__gt_flag__"]   = df["__gt_list__"  ].apply(lambda lst: tac in lst)

                    TP = int(((df["__pred_flag__"]) & (df["__gt_flag__"])).sum())
                    FP = int(((df["__pred_flag__"]) & (~df["__gt_flag__"])).sum())
                    FN = int((~df["__pred_flag__"] & (df["__gt_flag__"])).sum())

                    precision = TP / (TP + FP) if (TP + FP) else 0.0
                    recall    = TP / (TP + FN) if (TP + FN) else 0.0
                    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

                    metric_rows.append({
                        "tactic": tac,
                        "TP": TP, "FP": FP, "FN": FN,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1
                    })

                mdf = pd.DataFrame(metric_rows).set_index("tactic")
                st.subheader("📏 Classifier metrics")
                st.dataframe(
                    mdf.style.format({"precision":"{:.3f}",
                                      "recall":"{:.3f}",
                                      "f1":"{:.3f}"})
                )
            else:
                st.info("Add ground‑truth labels above to see precision / recall / F1.")

            # ---------- downloads ----------
            st.download_button(
                "📥 classified_results.csv",
                df.to_csv(index=False).encode(),
                "classified_results.csv",
                "text/csv"
            )
            st.download_button(
                "📥 category_frequencies.csv",
                counts.to_csv().encode(),
                "category_frequencies.csv",
                "text/csv"
            )
            if not top_words.empty:
                st.download_button(
                    "📥 top_keywords.csv",
                    top_words.to_csv().encode(),
                    "top_keywords.csv",
                    "text/csv"
                )
else:
    st.info("Upload a CSV to begin.")
