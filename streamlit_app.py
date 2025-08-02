import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import ast

# STEP 1: Choose tactic
st.title("🎯 Dictionary-Based Text Classifier")

st.header("Step 1: Choose a Marketing Tactic")
default_tactics = {
    "urgency_marketing": ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof": ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer']
}
tactic_name = st.selectbox("Select a tactic", list(default_tactics.keys()))
st.success(f"✅ Selected tactic: {tactic_name}")

# STEP 2: Upload CSV
st.header("Step 2: Upload Your CSV File")
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
    st.subheader("First few rows")
    st.dataframe(df.head())

    # STEP 3: Select column
    st.header("Step 3: Choose the Text Column")
    text_col = st.selectbox("Select column to analyze", df.columns)

    # STEP 4: Clean text and extract top keywords
    def clean_text(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

    df['cleaned_text'] = df[text_col].apply(clean_text)

    all_words = ' '.join(df['cleaned_text']).split()
    word_freq = pd.Series(all_words).value_counts()
    top_words = word_freq[word_freq > 1].head(20)

    st.subheader("🔍 Top Keywords in Your Data")
    st.dataframe(top_words)

    # STEP 5: Build editable dictionary
    st.header("Step 4: Review or Edit Dictionary")
    generated_dict = {tactic_name: set(top_words.index.tolist())}
    dict_string = st.text_area("Edit your dictionary here (Python format)", value=str(generated_dict))

    try:
        dictionary = ast.literal_eval(dict_string)
        st.success("✅ Dictionary parsed successfully.")
    except:
        st.error("⚠️ Invalid dictionary format. Using auto-generated version.")
        dictionary = generated_dict

    st.write("✅ Final dictionary used:", dictionary)

    # STEP 6: Classify text
    def classify(text, search_dict):
        categories = []
        for cat, terms in search_dict.items():
            if any(term in text.split() for term in terms):
                categories.append(cat)
        return categories if categories else ['uncategorized']

    df['categories'] = df['cleaned_text'].apply(lambda x: classify(x, dictionary))

    # STEP 7: Show results
    st.header("Step 5: Classification Results")
    st.subheader("📊 Category Frequencies")
    category_counts = pd.Series([cat for cats in df['categories'] for cat in cats]).value_counts()
    st.dataframe(category_counts)

    st.subheader("🔑 Top Keywords")
    st.dataframe(top_words)

    # STEP 8: WordCloud
    st.header("Step 6: Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
    st.pyplot(plt.figure(figsize=(12, 6)))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Text Data")
    st.pyplot()

    # STEP 9: Save results
    st.header("Step 7: Download Results")
    df.to_csv("classified_results.csv", index=False)
    category_counts.to_csv("category_frequencies.csv")
    top_words.to_csv("top_keywords.csv")

    with open("classified_results.csv", "rb") as f:
        st.download_button("📥 Download Classified Results", f, file_name="classified_results.csv")

    with open("category_frequencies.csv", "rb") as f:
        st.download_button("📥 Download Category Frequencies", f, file_name="category_frequencies.csv")

    with open("top_keywords.csv", "rb") as f:
        st.download_button("📥 Download Top Keywords", f, file_name="top_keywords.csv")

