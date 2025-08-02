# streamlit_app.py   —  run with:  streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re, ast

st.title("📊 Marketing-Tactic Text Classifier")

# ───────────────────────  STEP 1: choose tactic  ───────────────────────
default_tactics = {
    "urgency_marketing":  ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof":       ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer'],
}
tactic_name = st.selectbox("🎯 Step 1 — Choose a marketing tactic", list(default_tactics.keys()))
st.write(f"Selected tactic: **{tactic_name}**")

# ───────────────────────  STEP 2: upload CSV  ──────────────────────────
uploaded_file = st.file_uploader("📁 Step 2 — Upload your CSV file", type="csv")

if uploaded_file:
    # (Re)initialise session state when a new file is uploaded
    if "file_id" not in st.session_state or st.session_state.file_id != uploaded_file.id:
        st.session_state.clear()               # drop anything left from a previous run
        st.session_state.file_id = uploaded_file.id

    df = pd.read_csv(uploaded_file)
    st.success("File uploaded.")
    st.dataframe(df.head())

    # ─────────────────  STEP 3: select text column  ────────────────────
    text_col = st.selectbox("📋 Step 3 — Select the text column", df.columns)

    # ─────────────────  STEP 4: dictionary refinement  ─────────────────
    if st.button("🧠 Step 4 — Generate Keywords & Dictionary"):
        # clean text & extract keywords
        def clean(txt):
            return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())
        df["cleaned_text"] = df[text_col].apply(clean)
        all_words = " ".join(df["cleaned_text"]).split()
        word_freq = pd.Series(all_words).value_counts()
        top_words = word_freq[word_freq > 1].head(20)

        st.write("🔍 **Top keywords:**")
        st.dataframe(top_words)

        # auto dictionary
        auto_dict = {tactic_name: set(top_words.index)}
        st.write("🧠 Auto-generated dictionary:", auto_dict)

        # let user edit
        dict_str = st.text_area("✏️ Refine dictionary (Python format)", value=str(auto_dict))
        try:
            final_dict = ast.literal_eval(dict_str)
            st.success("Dictionary parsed successfully.")
        except Exception:
            st.error("Invalid format – using auto dictionary.")
            final_dict = auto_dict

        # store everything for the next step
        st.session_state.df          = df
        st.session_state.top_words   = top_words
        st.session_state.dictionary  = final_dict
        st.session_state.ready       = True          # flag: Step 4 completed

    # ────────────────  STEP 5: dictionary classifier  ─────────────────
    if st.session_state.get("ready"):
        if st.button("🧪 Step 5 — Run Classification"):
            df          = st.session_state.df.copy()
            top_words   = st.session_state.top_words
            dictionary  = st.session_state.dictionary

            def classify(txt, d):
                return [cat for cat, terms in d.items() if any(t in txt.split() for t in terms)] or ["uncategorized"]

            df["categories"] = df["cleaned_text"].apply(lambda x: classify(x, dictionary))
            cat_counts = pd.Series([c for cats in df["categories"] for c in cats]).value_counts()

            st.subheader("📊 Category frequencies")
            st.table(cat_counts)

            st.subheader("🔑 Top keywords")
            st.table(top_words)

            fig, ax = plt.subplots(figsize=(10, 4))
            top_words.sort_values(ascending=False).plot(kind="bar", ax=ax)
            ax.set_title("Top Keyword Frequencies")
            st.pyplot(fig)

            # downloads
            def to_csv(obj, index=False): return obj.to_csv(index=index).encode()
            st.download_button("📥 classified_results.csv", to_csv(df), "classified_results.csv", "text/csv")
            st.download_button("📥 category_frequencies.csv",  to_csv(cat_counts, index=True), "category_frequencies.csv", "text/csv")
            st.download_button("📥 top_keywords.csv",          to_csv(top_words, index=True), "top_keywords.csv", "text/csv")
else:
    st.info("Upload a CSV to begin.")


