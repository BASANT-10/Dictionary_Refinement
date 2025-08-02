# Step control flags
keyword_generated = False
dictionary_finalized = False

# ─────────── NEW STEP 2 BUTTON ───────────
if st.button("🧠 Step 2: Generate Keywords & Dictionary"):
    df["cleaned_text"] = df[text_col].apply(lambda txt: re.sub(r"[^a-zA-Z0-9\s]", "", str(txt).lower()))
    all_words = " ".join(df["cleaned_text"]).split()
    word_freq = pd.Series(all_words).value_counts()
    top_words = word_freq[word_freq > 1].head(20)

    st.write("🔍 **Top keywords in your data:**")
    st.dataframe(top_words)

    generated_dict = {tactic_name: set(top_words.index)}
    st.write("🧠 *Auto-generated dictionary:*", generated_dict)

    if st.checkbox("✏️ Edit dictionary?", key="edit_dict"):
        custom_dict_str = st.text_area(
            "Paste your dictionary here (e.g. {'urgency_marketing': {'now', 'hurry'}})",
            value=str(generated_dict),
            key="custom_dict_input"
        )
        try:
            dictionary = ast.literal_eval(custom_dict_str)
            st.success("✅ Final dictionary parsed successfully.")
        except:
            st.error("❌ Invalid dictionary format. Using auto-generated one.")
            dictionary = generated_dict
    else:
        dictionary = generated_dict

    st.session_state["top_words"] = top_words
    st.session_state["dictionary"] = dictionary
    st.session_state["df"] = df
    st.session_state["step2_done"] = True

# ─────────── NEW STEP 3 BUTTON ───────────
if st.session_state.get("step2_done") and st.button("🧪 Step 3: Run Classification"):
    df = st.session_state["df"]
    dictionary = st.session_state["dictionary"]
    top_words = st.session_state["top_words"]

    def classify(txt, search_dict):
        return [
            cat
            for cat, terms in search_dict.items()
            if any(term in txt.split() for term in terms)
        ] or ["uncategorized"]

    df["categories"] = df["cleaned_text"].apply(lambda x: classify(x, dictionary))

    category_counts = pd.Series([c for cats in df["categories"] for c in cats]).value_counts()

    st.subheader("📊 Category Frequencies")
    st.table(category_counts)

    st.subheader("🔑 Top Keywords")
    st.table(top_words)

    fig, ax = plt.subplots(figsize=(10, 5))
    top_words.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_xlabel("Keywords")
    ax.set_ylabel("Frequency")
    ax.set_title("Top Keyword Frequencies")
    st.pyplot(fig)

    def to_csv_bytes(frame):
        return frame.to_csv(index=False).encode()

    st.subheader("💾 Download Results")
    st.download_button("📥 classified_results.csv", to_csv_bytes(df), "classified_results.csv", "text/csv")
    st.download_button("📥 category_frequencies.csv", category_counts.to_csv().encode(), "category_frequencies.csv", "text/csv")
    st.download_button("📥 top_keywords.csv", top_words.to_csv().encode(), "top_keywords.csv", "text/csv")

