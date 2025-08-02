# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re, ast

# STEP 1: Title
st.title("🧠 Dictionary-Based Text Classification")

# STEP 2: Tactic selection
st.header("Step 1: Choose a Marketing Tactic")

default_tactics = {
    "urgency_marketing": ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof": ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer']
}

tactic_name = st.selectbox("Select a tactic", list(default_tactics.keys()))
st.success(f"Selected tactic: {tactic_name}")

# STEP 3: Upload CSV
st.header("Step 2: Upload Your CSV File")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.subheader("First few rows")
    st.dataframe(df.head())

    # STEP 4: Choose text column
    st.header("Step 3: Select Text Column")
    text_col = st.selectbox("Choose a column for text analysis", df.columns)

    # STEP 5: Clean text & extract top keywords
    def clean_text(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

    df['cleaned_text'] = df[text_col].apply(clean_text)
    all_words = ' '.join(df['cleaned_text']).split()
    word_freq = pd.Series(all_words).value_counts()
    top_words = word_freq[word_freq > 1].head(20)

    st.subheader("🔍 Top Keywords in Your Data")
    st.dataframe(top_words)

    # STEP 6: Auto-generate dictionary
    st.header("Step 4: Review or Edit Generated Dictionary")
    generated_dict = {tactic_name: set(top_words.index.tolist())}
    default_dict_str = str(generated_dict)

    custom_dict_str = st.text_area(
        "You can edit the dictionary below (Python format):",
        value=default_dict_str,
        height=150
    )

    try:
        dictionary = ast.literal_eval(custom_dict_str)
        st.success("✅ Final dictionary parsed successfully.")
    except:
        st.error("⚠️ Invalid dictionary format. Using auto-generated one.")
        dictionary = generated_dict

    # STEP 7: Classification
    def classify(text, search_dict):
        categories = []
        for cat, terms in search_dict.items():
            if any(term in text.split() for term in terms):
                categories.append(cat)
        return categories if categories else ['uncategorized']

    df['categories'] = df['cleaned_text'].apply(lambda x: classify(x, dictionary))

    # STEP 8: Results
    st.header("Step 5: Classification Results")

    st.subheader("📊 Category Frequencies")
    all_cats = [cat for cats in df['categories'] for cat in cats]
    category_counts = pd.Series(all_cats).value_counts()
    st.dataframe(category_counts)

    st.subheader("🔑 Top Keywords")
    st.dataframe(top_words)

    # STEP 9: Word Cloud
    st.subheader("☁️ Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # STEP 10: Save & download results
    st.header("Step 6: Download Results")

    df.to_csv("classified_results.csv", index=False)
    category_counts.to_csv("category_frequencies.csv")
    top_words.to_csv("top_keywords.csv")

    with open("classified_results.csv", "rb") as f:
        st.download_button("📥 Download Classified Data", f, file_name="classified_results.csv", mime="text/csv")

    with open("category_frequencies.csv", "rb") as f:
        st.download_button("📥 Download Category Frequencies", f, file_name="category_frequencies.csv", mime="text/csv")

    with open("top_keywords.csv", "rb") as f:
        st.download_button("📥 Download Top Keywords", f, file_name="top_keywords.csv", mime="text/csv")

