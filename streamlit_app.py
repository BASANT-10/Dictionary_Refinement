import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re, ast
from collections import Counter

st.title("📊 Marketing-Tactic Text Classifier")

# ───────────────────── STEP 1: Choose Tactic ─────────────────────
default_tactics = {
    "urgency_marketing":  ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof":       ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer']
}

st.subheader("🎯 Step 1: Choose a Marketing Tactic")
tactic_name = st.selectbox("Select tactic:", list(default_tactics.keys()))
st.write(f"✅ Selected tactic: **{tactic_name}**")

# ───────────────────── STEP 2: Upload CSV ─────────────────────
st.subheader("📁 Step 2: Upload Your CSV File")
uploaded_file = st.file_uploader("Choose a CSV", type=("csv",))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    # ───────────────────── STEP 3: Select Text Column ─────────────────────
    st.subheader("📋 Step 3: Select Text Column")
    text_col = st.selectbox("Column containing text to analyze:", df.columns)
    st.write(f"✅ Selected column: **{text_col}**")

    # ───────────────────── STEP 4: Dictionary Refinement ─────────────────────
    st.header("🔧 Step 4: Dictionary Refinement")

    # Clean text and extract keywords
    def clean_text(txt):
        return re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower())

    df["cleaned_text"] = df[text_col].apply(clean_text)
    all_words = " ".join(df["cleaned_text"]).split()
    word_freq = pd.Series(all_words).value_counts()
    top_words = word_freq[word_freq > 1].head(20)

    st.write("🔍 Top keywords in your data:")
    st.dataframe(top_words)

    # Auto-generate dictionary
    generated_dict = {tactic_name: set(top_words.index)}
    st.write("🧠 Auto-generated dictionary:", generated_dict)

    # Let user edit it
    custom_dict_str = st.text_area(
        "✏️ Refine your dictionary below (Python format):",
        value=str(generated_dict),
        height=150,
    )

    try:
        refined_dict = ast.literal_eval(custom_dict_str)
        st.success("✅ Dictionary parsed successfully.")
    except Exception:
        st.error("⚠️ Invalid format. Reverting to generated dictionary.")
        refined_dict = generated_dict

    # ───────────────────── STEP 5: Dictionary Classifier Creation ─────────────────────
    st.header("🧪 Step 5: Dictionary Classifier Creation")

    if st.button("🔍 Run Classification"):
        # Apply classification
        def classify(txt: str, search_dict):
            return [
                cat
                for cat, terms in search_dict.items()
                if any(term in txt.split() for term in terms)
            ] or ["uncategorized"]

        df["categories"] = df["cleaned_text"].apply(lambda x: classify(x, refined_dict))

        # Show results
        category_counts = (
            pd.Series([c for cats in df["categories"] for c in cats]).value_counts()
        )

        st.subheader("📊 Category Frequencies")
        st.table(category_counts)

        st.subheader("🔑 Top Keywords")
        st.table(top_words)

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        top_words.sort_values(ascending=False).plot(kind="bar", ax=ax)
        ax.set_xlabel("Keywords")
        ax.set_ylabel("Frequency")
        ax.set_title("Top Keyword Frequencies")
        st.pyplot(fig)

        # Download buttons
        def to_csv_bytes(frame):
            return frame.to_csv(index=False).encode()

        st.subheader("💾 Download Results")
        st.download_button("📥 classified_results.csv", to_csv_bytes(df), "classified_results.csv", "text/csv")
        st.download_button("📥 category_frequencies.csv", category_counts.to_csv().encode(), "category_frequencies.csv", "text/csv")
        st.download_button("📥 top_keywords.csv", top_words.to_csv().encode(), "top_keywords.csv", "text/csv")
else:
    st.info("🕐 Please upload a CSV file to get started.")

