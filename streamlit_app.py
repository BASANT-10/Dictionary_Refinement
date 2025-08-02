# STEP 0: Install required packages
!pip install -q pandas matplotlib wordcloud

# STEP 1: Imports
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re, ast
from google.colab import files
import io

# STEP 2: Choose tactic
print("🎯 Choose a marketing tactic from the options below:")
default_tactics = {
    "urgency_marketing": ['now', 'today', 'limited', 'hurry', 'exclusive'],
    "social_proof": ['bestseller', 'popular', 'trending', 'recommended'],
    "discount_marketing": ['sale', 'discount', 'deal', 'free', 'offer']
}

for i, tactic in enumerate(default_tactics):
    print(f"{i}. {tactic}")

tactic_idx = int(input("Enter the number of your chosen tactic: "))
tactic_name = list(default_tactics.keys())[tactic_idx]
print(f"✅ Selected tactic: {tactic_name}")

# STEP 3: Upload CSV
print("\n📁 Upload your CSV file:")
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(list(uploaded.values())[0]))
print("✅ File uploaded. First few rows:")
display(df.head())

# STEP 4: Select column
print("\n📋 Available columns:")
for i, col in enumerate(df.columns):
    print(f"{i}. {col}")
col_idx = int(input("Enter the column number containing the text to analyze: "))
text_col = df.columns[col_idx]
print(f"✅ Selected column: {text_col}")

# STEP 5: Clean text and extract top keywords
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

df['cleaned_text'] = df[text_col].apply(clean_text)

# Generate top keywords
all_words = ' '.join(df['cleaned_text']).split()
word_freq = pd.Series(all_words).value_counts()
top_words = word_freq[word_freq > 1].head(20)
print("\n🔍 Top keywords in your data:")
print(top_words)

# STEP 6: Build editable dictionary
print("\n🧠 Auto-suggesting dictionary from top words for tactic:", tactic_name)
generated_dict = {tactic_name: set(top_words.index.tolist())}
print("🛠️ Generated dictionary:", generated_dict)

edit_dict = input("✏️ Would you like to edit the dictionary? (y/n): ").strip().lower()
if edit_dict == 'y':
    print("Enter your custom dictionary in Python format like:")
    print("{'urgency_marketing': {'now', 'hurry'}}")
    custom_dict_str = input("Paste your dictionary here:\n")
    dictionary = ast.literal_eval(custom_dict_str)
else:
    dictionary = generated_dict

print("✅ Final dictionary used:", dictionary)

# STEP 7: Classify text
def classify(text, search_dict):
    categories = []
    for cat, terms in search_dict.items():
        if any(term in text.split() for term in terms):
            categories.append(cat)
    return categories if categories else ['uncategorized']

df['categories'] = df['cleaned_text'].apply(lambda x: classify(x, dictionary))

# STEP 8: Show results
print("\n📊 Category frequencies:")
category_counts = pd.Series([cat for cats in df['categories'] for cat in cats]).value_counts()
print(category_counts)

print("\n🔑 Top keywords:")
print(top_words)

# STEP 9: WordCloud
wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Text Data")
plt.show()

# STEP 10: Save results
df.to_csv("classified_results.csv", index=False)
category_counts.to_csv("category_frequencies.csv")
top_words.to_csv("top_keywords.csv")
files.download("classified_results.csv")
files.download("category_frequencies.csv")
files.download("top_keywords.csv")
