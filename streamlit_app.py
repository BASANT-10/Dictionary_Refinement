# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re, ast

st.title("📊 Marketing‑Tactic Text Classifier")

# ───────────────── STEP 1 – choose tactic ─────────────────
default_tactics = {
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
tactic = st.selectbox("🎯 Step 1 — choose a tactic", list(default_tactics.keys()))
st.write(f"Chosen tactic: **{tactic}**")

# ───────────────── STEP 2 – upload CSV ───────────────────
file = st.file_uploader("📁 Step 2 — upload CSV", type="csv")

# ---------- helper functions ----------
def clean(txt: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

def classify(txt: str, d):
    return [
        cat for cat, terms in d.items()
        if any(word in txt.split() for word in terms)
    ] or ["uncategorized"]
# --------------------------------------

if "dict_ready" not in st.session_state:
    st.session_state.dict_ready = False

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())   # preview

    text_col = st.selectbox("📋 Step 3 — select text column", df.columns)

    # ─────────── STEP 4 – generate / refine dictionary ───────────
    if st.button("🧠 Step 4 — Generate Keywords & Dictionary"):
        df["cleaned"] = df[text_col].apply(clean)
        all_words = " ".join(df["cleaned"]).split()
        word_freq = pd.Series(all_words).value_counts()
        top = word_freq[word_freq > 1].head(50)

        stop_words = set([
            'the', 'is', 'in', 'on', 'and', 'a', 'for', 'you', 'i', 'are', 'of', 'your',
            'to', 'my', 'with', 'it', 'me', 'this', 'that', 'or'
        ])
        filtered_top = [w for w in top.index if w not in stop_words]

        use_default = st.checkbox("✅ Use default dictionary for this tactic")
        if use_default:
            auto_dict = {tactic: default_tactics[tactic]}
        else:
            auto_dict = {tactic: set(filtered_top)}

        st.subheader("Top keywords (filtered)")
        st.dataframe(pd.Series(filtered_top, name="Keyword"))

        st.write("Auto‑generated dictionary:", auto_dict)

        dict_text = st.text_area(
            "✏️ Edit dictionary (Python dict syntax)",
            value=str(auto_dict),
            height=150
        )
        try:
            final_dict = ast.literal_eval(dict_text)
            st.session_state.dictionary = final_dict  # ✅ Updated to use edited dict
            st.success("Dictionary saved.")
        except Exception:
            st.error("Bad format → using auto dict.")
            st.session_state.dictionary = auto_dict

        st.session_state.df         = df
        st.session_state.top_words  = pd.Series(filtered_top, name="Keyword")
        st.session_state.dict_ready = True

    # ─────────── STEP 5 – run classifier (only if ready) ───────────
    if st.session_state.dict_ready:
        if st.button("🧪 Step 5 — Run Classification"):
            df         = st.session_state.df.copy()
            top_words  = st.session_state.top_words
            dictionary = st.session_state.dictionary

            df["categories"] = df["cleaned"].apply(lambda x: classify(x, dictionary))

            # -----------------------------▼ NEW COLUMN ▼-----------------------------
            df["tactic_flag"] = df["categories"].apply(
                lambda cats: 1 if tactic in cats else 0
            )
            # -------------------------------------------------------------------------

            counts = pd.Series(
                [c for cats in df["categories"] for c in cats]
            ).value_counts()

            st.subheader("📊 Category frequencies")
            st.table(counts)

            st.subheader("🔑 Top keywords")
            st.table(top_words)

            fig, ax = plt.subplots(figsize=(10, 4))
            pd.Series(top_words).value_counts().sort_values(ascending=False).plot.bar(ax=ax)
            ax.set_title("Top keyword frequencies")
            st.pyplot(fig)

            # Downloads
            st.download_button(
                "📥 classified_results.csv",
                df.to_csv(index=False).encode(),
                "classified_results.csv",
                "text/csv",
            )
            st.download_button(
                "📥 category_frequencies.csv",
                counts.to_csv().encode(),
                "category_frequencies.csv",
                "text/csv",
            )
            st.download_button(
                "📥 top_keywords.csv",
                pd.Series(top_words).to_csv().encode(),
                "top_keywords.csv",
                "text/csv",
            )
else:
    st.info("Upload a CSV to begin.")
