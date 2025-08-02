import streamlit as st
import pandas as pd
import re
import ast
from collections import Counter
import nltk

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# App title
st.title("🧠 Dictionary-Based Text Classifier (No WordCloud)")

# Step 1: Tactic selection
st.header("Step 1: Choose a Marketing Tactic")
tactic_dict = {
    "urgency_marketing": ["now", "today", "limited", "hurry", "exclusive"],
    "social_proof": ["bestseller", "popular", "trending", "recommended"],
    "discount_marketing": ["sale", "discount", "deal", "free", "offer"]
}
tactic_name = st.selectbox("Select a tactic", list(tactic_dict.keys()))
base_keywords = set(tactic_dict[tactic_name])

# Step 2: Upload CSV
st.header("Step 2: Upload a CSV File")
uploaded_file = st.file_uploader("Upload your dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    # Step 3: Select text column
    st.header("Step 3: Select the Text Column")
    text_col = st.selectbox("Choose a text column", df.columns)

    # Step 4: Clean text and show top keywords
    def clean_text(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

    df["cleaned_text"] = df[text_col].apply(clean_text)
    all_words = " ".join(df["cleaned_text"]).split()
    filtered_words = [word for word in all_words if word not in stopwords.words("english")]
    word_counts = pd.Series(filtered_words).value_counts()
    top_words = word_counts.head(20)

    st.header("Step 4: Top Keywords from Text")
    st.dataframe(top_words)

    # Step 5: Edit dictionary
    st.header("Step 5: Review or Edit the Dictionary")
    auto_dict = {tactic_name: set(top_words.index.tolist()) | base_keywords}
    dict_str = str(auto_dict)
    user_dict_input = st.text_area("Edit dictionary (Python format)", value=dict_str, height=200)

    try:
        final_dict = ast.literal_eval(user_dict_input)
        st.success("✅ Dictionary loaded.")
    except Exception as e:
        st.error("❌ Invalid dictionary format. Using generated dictionary.")
        final_dict = auto_dict

    # Step 6: Classification
    st.header("Step 6: Run Classification")

    def classify(text, dictionary):
        matches = []
        words = text.split()
        for category, terms in dictionary.items():
            if any(term in words for term in terms):
                matches.append(category)
        return matches if matches else ["uncategorized"]

    df["categories"] = df["cleaned_text"].apply(lambda x: classify(x, final_dict))

    st.subheader("📊 Classification Results")
    st.dataframe(df[[text_col, "categories"]].head())

    # Step 7: Show category frequencies
    st.header("Step 7: Category Frequencies")
    flat_cats = [cat for sublist in df["categories"] for cat in sublist]
    cat_freq = pd.Series(flat_cats).value_counts()
    st.dataframe(cat_freq)

    # Step 8: Download results
    st.header("Step 8: Download Outputs")

    df.to_csv("classified_output.csv", index=False)
    cat_freq.to_csv("category_frequencies.csv")

    with open("classified_output.csv", "rb") as f:
        st.download_button("⬇️ Download Classified Data", f, file_name="classified_output.csv")

    with open("category_frequencies.csv", "rb") as f:
        st.download_button("⬇️ Download Category Frequencies", f, file_name="category_frequencies.csv")

