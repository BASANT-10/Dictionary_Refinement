# streamlit_app.py   —  run with:  streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, ast
from io import BytesIO
from collections import Counter

st.title("📊 Marketing-Tactic Text Classifier")

# ────────────────────────────────── STEP 2: Choose tactic ──────────────────────────────────
default_tactics = {
    "urgency_marketing":  ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof":       ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer']
}

st.subheader("🎯 Choose a marketing tactic")
tactic_name = st.selectbox("Select tactic:", list(default_tactics.keys()))
st.write(f"✅ Selected tactic: **{tactic_name}**")

# ────────────────────────────────── STEP 3: Upload CSV ─────────────────────────────────────
st.subheader("📁 Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV", type=("csv",))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded")
    st.dataframe(df.head())

    # ───────────────────────────── STEP 4: Select text column ──────────────────────────────
    st.subheader("📋 Select text column")
    text_col = st.selectbox("Column containing text to analyze:", df.columns)
    st.write(f"✅ Selected column: **{text_col}**")

    # Run button triggers everything else
    if st.button("Run Analysis"):
        # ────────────────────── STEP 5: Clean text & extract top keywords ───────────────────
        def clean_text(txt):
            return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

        df["cleaned_text"] = df[text_col].apply(clean_text)
        all_words = " ".join(df["cleaned_text"]).split()
        word_freq = pd.Series(all_words).value_counts()
        top_words = word_freq[word_freq > 1].head(20)

        st.write("🔍 **Top keywords in your data:**")
        st.dataframe(top_words)

        # ────────────────────────── STEP 6: Build (editable) dictionary ────────────────────
        generated_dict = {tactic_name: set(top_words.index)}
        st.write("🧠 *Auto-generated dictionary:*", generated_dict)

        if st.checkbox("✏️ Edit dictionary?"):
            custom_dict_str = st.text_area(
                "Paste your dictionary here "
                "(e.g. {'urgency_marketing': {'now', 'hurry'}})",
                value=str(generated_dict),
            )
            dictionary = ast.literal_eval(custom_dict_str)
        else:
            dictionary = generated_dict

        st.write("✅ **Final dictionary used:**", dictionary)

        # ────────────────────────────── STEP 7: Classify text ──────────────────────────────
        def classify(txt: str, search_dict):
            return [
                cat
                for cat, terms in search_dict.items()
                if any(term in txt.split() for term in terms)
            ] or ["uncategorized"]

        df["categories"] = df["cleaned_text"].apply(lambda x: classify(x, dictionary))

        # ──────────────────────────────── STEP 8: Results ─────────────────────────────────
        category_counts = (
            pd.Series([c for cats in df["categories"] for c in cats]).value_counts()
        )

        st.subheader("📊 Category frequencies")
        st.table(category_counts)

        st.subheader("🔑 Top keywords")
        st.table(top_words)

        # ─────────────────────────────── STEP 9: Bar chart ───────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        top_words.sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_xlabel("Keywords")
        ax.set_ylabel("Frequency")
        ax.set_title("Top Keyword Frequencies")
        st.pyplot(fig)

        # ────────────────────────────── STEP 10: Downloads ────────────────────────────────
        def to_csv_bytes(frame):
            return frame.to_csv(index=False).encode()

        st.subheader("💾 Download results")
        st.download_button(
            label="classified_results.csv",
            data=to_csv_bytes(df),
            file_name="classified_results.csv",
            mime="text/csv",
        )
        st.download_button(
            label="category_frequencies.csv",
            data=category_counts.to_csv().encode(),
            file_name="category_frequencies.csv",
            mime="text/csv",
        )
        st.download_button(
            label="top_keywords.csv",
            data=top_words.to_csv().encode(),
            file_name="top_keywords.csv",
            mime="text/csv",
        )
else:
    st.info("Awaiting CSV upload…")
