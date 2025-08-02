import streamlit as st
import pandas as pd
import re
import ast
from collections import Counter

# 🧠 Hardcoded common stopwords (to avoid nltk issues)
stopword_list = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "you", "your", "i",
    "we", "they", "but", "or", "if", "not", "so", "my", "me", "our"
}

# Title
st.title("📊 Dictionary-Based Text Classifier (No External Libraries)")

# Step 1: Choose tactic
st.header("Step 1: Select a Marketing Tactic")
default_tactics = {
    "urgency_marketing": ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof": ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer']
}
tactic_name = st.selectbox("Choose tactic", list(default_tactics.keys()))
base_keywords = set(default_tactics[tactic_name])

# Step 2: Upload file
st.header("Step 2: Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # Step 3: Select column
    st.header("Step 3: Select Text Column")
    text_col = st.selectbox("Select text column", df.columns)

    # Step 4: Clean text
    def clean(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

    df['cleaned_text'] = df[text_col].apply(clean)

    # Step 5: Get top keywords
    all_words = ' '.join(df['cleaned_text']).split()
    filtered_words = [w for w in all_words if w not in stopword_list]
    word_counts = pd.Series(filtered_words).value_counts()
    top_words = word_counts[word_counts > 1].head(20)

    st.header("Step 4: Top Keywords in Your Data")
    st.dataframe(top_words)

    # Step 6: Build/edit dictionary
    st.header("Step 5: Review or Edit Dictionary")
    generated_dict = {tactic_name: set(top_words.index.tolist()) | base_keywords}
    user_input_dict = st.text_area("Edit dictionary below (Python format):", str(generated_dict), height=200)

    try:
        final_dict = ast.literal_eval(user_input_dict)
        st.success("Dictionary parsed successfully.")
    except Exception as e:
        st.error("Invalid dictionary format. Using auto-generated one.")
        final_dict = generated_dict

    # Step 7: Classify
    st.header("Step 6: Classify Text")

    def classify(text, dictionary):
        tokens = text.split()
        matches = []
        for category, keywords in dictionary.items():
            if any(word in tokens for word in keywords):
                matches.append(category)
        return matches if matches else ['uncategorized']

    df['categories'] = df['cleaned_text'].apply(lambda x: classify(x, final_dict))

    st.subheader("Sample Classification")
    st.dataframe(df[[text_col, 'categories']].head())

    # Step 8: Show frequency
    st.header("Step 7: Category Frequency")
    all_categories = [cat for sublist in df['categories'] for cat in sublist]
    cat_freq = pd.Series(all_categories).value_counts()
    st.dataframe(cat_freq)

    # Step 9: Download
    st.header("Step 8: Download Results")
    df.to_csv("classified_output.csv", index=False)
    cat_freq.to_csv("category_frequency.csv")

    with open("classified_output.csv", "rb") as f:
        st.download_button("⬇️ Download Classified Data", f, file_name="classified_output.csv")

    with open("category_frequency.csv", "rb") as f:
        st.download_button("⬇️ Download Category Frequencies", f, file_name="category_frequency.csv")
