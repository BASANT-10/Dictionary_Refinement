# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import ast
import nltk

# Ensure stopwords are available
nltk.download('stopwords')
from nltk.corpus import stopwords

# Title
st.title("🧠 Dictionary-Based Text Classification")

# Step 1: Choose tactic
st.header("Step 1: Choose a Marketing Tactic")

default_tactics = {
    "urgency_marketing": ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof": ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer']
}

tactic_name = st.selectbox("Select a tactic", list(default_tactics.keys()))
st.success(f"Selected tactic: {tactic_name}")

# Step 2: Upload CSV
st.header("Step 2: Upload Your CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Step 3: Select text column
    st.header("Step 3: Select the Text Column")
    text_col = st.selectbox("Select column to analyze", df.columns)

    # Step 4: Clean text and extract top words
    def clean_text(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

    df['cleaned_text'] = df[text_col].apply(clean_text)
    all_words = ' '.join(df['cleaned_text']).split()
    filtered_words = [word for word in all_words if word not in stopwords.words('english')]
    word_freq = pd.Series(filtered_words).value_counts()
    top_words = word_freq[word_freq > 1].head(20)

    st.subheader("🔍 Top Keywords from Text")
    st.dataframe(top_words)

    # Step 5: Build or edit dictionary
    st.header("Step 4: Review or Edit the Suggested Dictionary")
    generated_dict = {tactic_name: set(top_words.index.tolist())}
    default_dict_str = str(generated_dict)

    custom_dict_str = st.text_area(
        "Edit the dictionary (in Python dict format):",
        value=default_dict_str,
        height=200
    )

    try:
        dictionary = ast.literal_eval(custom_dict_str)
        st.success("✅ Dictionary loaded successfully.")
    except Exception as e:
        st.error(f"Invalid dictionary format. Using auto-generated one.\n{e}")
        dictionary = generated_dict

    # Step 6: Classification
    st.header("Step 5: Classify Text")

    def classify(text, search_dict):
        categories = []
        for cat, terms in search_dict.items():
            if any(term in text.split() for term in terms):
                categories.append(cat)
        return categories if categories else ['uncategorized']

    df['categories'] = df['cleaned_text'].apply(lambda x: classify(x, dictionary))

    st.subheader("📋 Sample Classification Results")
    st.dataframe(df[['original_text', 'categories']] if 'original_text' in df else df[['cleaned_text', 'categories']])

    # Step 7: Category Frequencies
    st.subheader("📊 Category Frequency Summary")
    all_cats = [cat for cats in df['categories'] for cat in cats]
    category_counts = pd.Series(all_cats).value_counts()
    st.dataframe(category_counts)

    # Step 8: Word Cloud
    st.subheader("☁️ Word Cloud of Filtered Words")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Step 9: Download Results
    st.header("Step 6: Download Results")

    df.to_csv("classified_results.csv", index=False)
    category_counts.to_csv("category_frequencies.csv")
    top_words.to_csv("top_keywords.csv")

    with open("classified_results.csv", "rb") as f:
        st.download_button("📥 Download Full Classification", f, file_name="classified_results.csv")

    with open("category_frequencies.csv", "rb") as f:
        st.download_button("📥 Download Category Frequencies", f, file_name="category_frequencies.csv")

    with open("top_keywords.csv", "rb") as f:
        st.download_button("📥 Download Top Keywords", f, file_name="top_keywords.csv")
